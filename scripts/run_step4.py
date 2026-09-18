"""Step 4: alert layer + caregiver demo page.

    python scripts/run_step4.py     (needs scripts/run_step1.py and scripts/run_step3.py first; takes a few minutes)
Writes docs/demo/prefall_demo.html: open it in any browser, no server needed.

Everything is computed with the deployable pieces: the threshold detector, the SVM, and the
INT8 .tflite CNN-BiLSTM file exported by step 3, run per sample on the held-out test trials.
"""
import _bootstrap  # noqa: F401  (repo root on sys.path, cwd = repo root)
import os
import pickle
import numpy as np

from prefall.alert import pack_alert
from prefall.classical import make_models
from prefall.config import FS
from prefall.dataset import SCALE
from prefall.demo import build_html
from prefall.deploy import TFLiteRunner
from prefall.dsp import condition
from prefall.evaluate import calibrate
from prefall.features import sliding_windows, window_features
from prefall.hybrid import confirmed_alarms, episodes, run_policy
from prefall.threshold import trace, alarm_mask

for f in ("data/step1_windows.npz", "export/cnn_bilstm_int8.tflite"):
    if not os.path.exists(f):
        raise SystemExit(f"missing {f}: run scripts/run_step1.py and scripts/run_step3.py first")

with open("data/step1_trials.pkl", "rb") as f:
    trials = pickle.load(f)
half = len(trials) // 2
train_tr, test_tr = trials[:half], trials[half:]
p = calibrate(train_tr, [trace(condition(t.raw)) for t in train_tr], mode="first_pass")

d = np.load("data/step1_windows.npz")
svm = make_models()["SVM (RBF)"].fit(d["X"][d["train_mask"]], d["y"][d["train_mask"]])
runner = TFLiteRunner(open("export/cnn_bilstm_int8.tflite", "rb").read())

print(f"running the detectors over {len(test_tr)} held-out trials ...")
S = []                                     # per trial: signals, masks, per-sample decisions
for k, tr in enumerate(test_tr):
    x = condition(tr.raw)
    th, vd = trace(x)
    W, _ = sliding_windows(x, 50, 1)
    prob = np.zeros(len(x))
    prob[49:] = runner.predict_proba(W / SCALE)[:, 1]
    svm_pred = np.zeros(len(x), bool)
    svm_pred[49:] = svm.predict(np.stack([window_features(w) for w in W])).astype(bool)
    S.append(dict(x=x, th=th, vd=vd, mask=alarm_mask(th, vd, p), prob=prob, svm=svm_pred, mlpos=prob > 0.5))
    if (k + 1) % 50 == 0:
        print(f"  {k + 1}/{len(test_tr)}")


def summary(r):
    return dict(sensitivity=r["sensitivity"], specificity=r["specificity"], lead_ms=r["lead_mean_ms"],
                useful=r["useful_rate"], n_fall=r["n_fall"], n_adl=r["n_adl"],
                fa=r["fa_by_kind"])


rA = run_policy(test_tr, [s["mask"] for s in S], [s["svm"] for s in S], 0, 0)
rB = run_policy(test_tr, [s["mlpos"] for s in S], [s["mlpos"] for s in S], 300, 300)
print(f"A: sens {100*rA['sensitivity']:.1f}% spec {100*rA['specificity']:.1f}% lead {rA['lead_mean_ms']:.0f} ms")
print(f"B: sens {100*rB['sensitivity']:.1f}% spec {100*rB['specificity']:.1f}% lead {rB['lead_mean_ms']:.0f} ms")

LABELS = {"fall": "Fall (trip)", "stumble": "Stumble, recovered", "sit_fast": "Sitting down fast",
          "jump": "Jump", "stairs": "Walking down stairs", "walk": "Normal walking"}
PER_KIND, DEVICE_ID, N = 3, 0x0A01, 300
scenarios = []
for kind in LABELS:
    idx = [i for i, t in enumerate(test_tr) if t.kind == kind][:PER_KIND]      # first ones, not hand-picked
    for j, i in enumerate(idx, 1):
        tr, s = test_tr[i], S[i]
        alarms = {"A": confirmed_alarms(s["mask"], s["svm"], 0, 0),
                  "B": confirmed_alarms(s["mlpos"], s["mlpos"], 300, 300)}
        first = (alarms["A"] or alarms["B"] or episodes(s["mask"]) or [300])[0]
        centre = tr.t_onset if kind == "fall" else first / FS
        a0 = int(max(0, min(len(s["x"]) - N, round((centre - 3.0) * FS))))
        sl = slice(a0, a0 + N)
        rel = lambda a: round((a - a0) / FS, 3)
        keep = lambda v: [a for a in v if a0 <= a < a0 + N]
        scenarios.append(dict(
            label=f"{LABELS[kind]} #{j}", kind=kind, fs=FS, n=N,
            a=[round(float(v), 3) for v in np.linalg.norm(s["x"][sl, :3], axis=1)],
            th=[round(float(v), 2) for v in s["th"][sl]],
            vd=[round(float(v), 3) for v in s["vd"][sl]],
            p=[round(float(v), 3) for v in s["prob"][sl]],
            onset=None if kind != "fall" else round(tr.t_onset - a0 / FS, 3),
            impact=None if kind != "fall" else round(tr.t_impact - a0 / FS, 3),
            alarms={k: [rel(a) for a in keep(v)] for k, v in alarms.items()},
            payloads={k: [pack_alert(a * 1000 // FS, DEVICE_ID, s["prob"][a], 0 if k == "A" else 1).hex()
                          for a in keep(v)] for k, v in alarms.items()}))

os.makedirs("docs/demo", exist_ok=True)
html = build_html(dict(scenarios=scenarios, summary=dict(A=summary(rA), B=summary(rB)),
                       params=dict(theta_crit=p.theta_crit, v_thr=p.v_thr, tau=p.tau)))
open("docs/demo/prefall_demo.html", "w", encoding="utf-8").write(html)
print(f"wrote docs/demo/prefall_demo.html ({len(html)/1024:.0f} KB, {len(scenarios)} scenarios)")
