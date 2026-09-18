"""Step 2c: hybrid threshold + ML confirmation on the held-out test trials.

    python scripts/run_step2c.py     (needs scripts/run_step1.py and scripts/run_step2b.py first)
"""
import _bootstrap  # noqa: F401  (repo root on sys.path, cwd = repo root)
import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prefall.classical import make_models
from prefall.dsp import condition
from prefall.evaluate import calibrate
from prefall.features import window_features
from prefall.hybrid import (POLICIES, all_windows, per_stream, feature_pred, deep_pred, run_policy)
from prefall.threshold import trace, alarm_mask

for f in ("data/step1_windows.npz", "data/cnn_bilstm.keras", "data/convlstm.keras"):
    if not os.path.exists(f):
        raise SystemExit(f"missing {f}: run scripts/run_step1.py and scripts/run_step2b.py first")
import keras

d = np.load("data/step1_windows.npz")
with open("data/step1_trials.pkl", "rb") as f:
    trials = pickle.load(f)
half = len(trials) // 2
train_tr, test_tr = trials[:half], trials[half:]

# stage 1: first-pass threshold, calibrated on training trials only (as in step 1)
streams_tr = [condition(t.raw) for t in train_tr]
p = calibrate(train_tr, [trace(x) for x in streams_tr], mode="first_pass")
streams = [condition(t.raw) for t in test_tr]
masks = [alarm_mask(*trace(x), p) for x in streams]
print(f"stage 1: theta_crit={p.theta_crit} deg, v_thr={p.v_thr} m/s, tau={p.tau}")

# stage 2: per-sample model decisions on the test trials
W, owner = all_windows(streams)
print(f"scoring {len(W)} windows per model ...")
Xf, y, tid, train_mask = d["X"], d["y"], d["trial_id"], d["train_mask"]
clf = make_models()
clf["Random Forest"].fit(Xf[train_mask], y[train_mask])
clf["SVM (RBF)"].fit(Xf[train_mask], y[train_mask])
feats = np.stack([window_features(w) for w in W])
preds = {
    "Random Forest": per_stream(clf["Random Forest"].predict(feats), owner, streams),
    "SVM (RBF)": per_stream(clf["SVM (RBF)"].predict(feats), owner, streams),
    "1D-CNN + BiLSTM": per_stream(deep_pred(keras.saving.load_model("data/cnn_bilstm.keras"), W), owner, streams),
    "ConvLSTM": per_stream(deep_pred(keras.saving.load_model("data/convlstm.keras"), W), owner, streams),
}


def line(r):
    return (f"sens {100*r['sensitivity']:5.1f}%  spec {100*r['specificity']:5.1f}%  "
            f"lead {r['lead_mean_ms']:4.0f} ms  lead>=200ms {100*r['useful_rate']:5.1f}%")


base = run_policy(test_tr, masks, [None] * len(test_tr))
print(f"\nthreshold only (no ML)                {line(base)}")
results = {"threshold only": base}
for name, pr in preds.items():
    print(f"\n[{name}]   policy = (wait ms, model must be positive for last ms)")
    results[name] = {}
    for dd, mm in POLICIES:
        r = run_policy(test_tr, masks, pr, dd, mm)
        results[name][f"{dd},{mm}"] = r
        print(f"  ({dd:3d},{mm:3d})  {line(r)}   FA: sit {100*r['fa_by_kind'].get('sit_fast',0):3.0f}% "
              f"stumble {100*r['fa_by_kind'].get('stumble',0):3.0f}% jump {100*r['fa_by_kind'].get('jump',0):3.0f}%")

# alternative architecture: no threshold gate. The model's own rising edge starts the clock and it
# must stay positive for d ms before alarming.
print("\nML only, no threshold (clock starts when the model first says pre-fall; must stay positive for d ms)")
results["ML only"] = {}
for name, pr in preds.items():
    for dd in (0, 100, 200, 300):
        r = run_policy(test_tr, pr, pr, dd, dd)
        results["ML only"][f"{name}|{dd}"] = r
        print(f"  {name:16s} d={dd:3d}  {line(r)}   FA: sit {100*r['fa_by_kind'].get('sit_fast',0):3.0f}% "
              f"stumble {100*r['fa_by_kind'].get('stumble',0):3.0f}% jump {100*r['fa_by_kind'].get('jump',0):3.0f}% "
              f"walk {100*r['fa_by_kind'].get('walk',0):3.0f}%")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
xs = [0, 100, 200, 300]
for name in preds:
    pol = ["0,0", "100,100", "200,200", "300,300"]
    ax[0].plot(xs, [100 * results[name][k]["specificity"] for k in pol], marker="o", label=name)
    ax[1].plot(xs, [results[name][k]["lead_mean_ms"] for k in pol], marker="o", label=name)
ax[0].axhline(100 * base["specificity"], ls=":", c="gray"); ax[1].axhline(200, ls=":", c="green")
ax[0].set_xlabel("confirmation wait (ms), model positive throughout"); ax[0].set_ylabel("trial specificity (%)")
ax[1].set_xlabel("confirmation wait (ms), model positive throughout"); ax[1].set_ylabel("mean lead time (ms)")
ax[0].legend(); fig.tight_layout(); fig.savefig("docs/images/hybrid_tradeoff.png", dpi=130)
with open("data/step2c_results.json", "w") as f:
    json.dump(results, f, indent=1)
