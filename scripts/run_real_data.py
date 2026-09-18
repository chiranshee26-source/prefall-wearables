"""Train and evaluate the pipeline on a REAL dataset (SisFall or KFall). See docs/real-data.md.

    python scripts/run_real_data.py --dataset kfall   --path data/real/KFall
    python scripts/run_real_data.py --dataset sisfall --path data/real/SisFall
    python scripts/run_real_data.py --dataset sisfall --path data/real/SisFall --inspect     (look at the raw files)
    python scripts/run_real_data.py ... --limit-subjects 6                                     (quick trial run)

Everything is split by SUBJECT (nobody appears in both train and test). Models are trained on the training
subjects only; the threshold is calibrated on them too. Classical models only (Random Forest and SVM on the
12 features): they need no TensorFlow and are the fastest way to see how real data behaves.
"""
import _bootstrap  # noqa: F401  (repo root on sys.path, cwd = repo root)
import argparse
import json
import os
import time

import numpy as np

from prefall.classical import make_models
from prefall.config import FS, WIN
from prefall.dataset import build_windows
from prefall.dsp import condition
from prefall.evaluate import calibrate
from prefall.features import sliding_windows, window_features
from prefall.hybrid import run_policy
from prefall.realdata import inspect, load_kfall, load_sisfall, split_subjects
from prefall.threshold import alarm_mask, trace

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", choices=["sisfall", "kfall"], required=True)
ap.add_argument("--path", required=True, help="SisFall: folder with the subject folders. KFall: folder with sensor_data/ and label_data/")
ap.add_argument("--test-frac", type=float, default=0.3, help="share of SUBJECTS held out for testing")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit-subjects", type=int, default=None, help="use only the first N subjects (quick test)")
ap.add_argument("--max-neg", type=int, default=30000, help="cap on normal-activity training windows (keeps the SVM fast)")
ap.add_argument("--eval-hop", type=int, default=2, help="score the models every N samples on test recordings (2 = every 40 ms)")
ap.add_argument("--onset-window", type=float, default=1.0, help="SisFall only: assumed seconds of fall before the estimated impact")
ap.add_argument("--inspect", action="store_true", help="print the first lines of a few raw files and exit")
args = ap.parse_args()

if args.inspect:
    inspect(args.path, args.dataset)
    raise SystemExit(0)

t0 = time.time()
print(f"loading {args.dataset} from {args.path} ...")
if args.dataset == "sisfall":
    trials, amap = load_sisfall(args.path, limit_subjects=args.limit_subjects, onset_window=args.onset_window)
else:
    trials, amap = load_kfall(args.path, limit_subjects=args.limit_subjects)
train, test = split_subjects(trials, args.test_frac, args.seed)


def comp(ts):
    return f"{len({t.subject['id'] for t in ts})} subjects, {sum(t.kind == 'fall' for t in ts)} falls, {sum(t.kind != 'fall' for t in ts)} normal-activity recordings"


print(f"train: {comp(train)}\ntest:  {comp(test)}")
if args.dataset == "sisfall":
    print("NOTE: SisFall has no timing labels. Impact = estimated acceleration peak, fall onset = impact minus "
          f"{args.onset_window:.1f} s. Lead times below are APPROXIMATE.")

# ---- threshold (stage 1), calibrated on training subjects only --------------------------------------------
print("calibrating the threshold on the training subjects ...")
train_traces = [trace(condition(t.raw, t.fs_raw)) for t in train]
p_first = calibrate(train, train_traces, mode="first_pass")
print(f"  first-pass threshold: theta_crit={p_first.theta_crit} deg, v_thr={p_first.v_thr} m/s, tau={p_first.tau}")
del train_traces

# ---- classical models on windows ------------------------------------------------------------------------------
print("building training windows and fitting Random Forest and SVM ...")
X, y, tid, _ = build_windows(train, hop=10)
rng = np.random.default_rng(args.seed)
neg = np.flatnonzero(y == 0)
if len(neg) > args.max_neg:
    keep = np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, args.max_neg, replace=False)])
    X, y = X[keep], y[keep]
print(f"  {int((y == 1).sum())} pre-fall windows, {int((y == 0).sum())} normal windows")
models = make_models(args.seed)
for name in ("Random Forest", "SVM (RBF)"):
    models[name].fit(X, y)
models = {k: models[k] for k in ("Random Forest", "SVM (RBF)")}

# ---- score the held-out subjects --------------------------------------------------------------------------------
print(f"scoring {len(test)} test recordings (every {args.eval_hop} samples) ...")
masks, preds = [], {k: [] for k in models}
for i, t in enumerate(test):
    x = condition(t.raw, t.fs_raw)
    masks.append(alarm_mask(*trace(x), p_first))
    W, ends = sliding_windows(x, WIN, args.eval_hop)
    if len(W) == 0:
        for k in models:
            preds[k].append(np.zeros(len(x), bool))
        continue
    F = np.stack([window_features(w) for w in W])
    for k, m in models.items():
        arr = np.zeros(len(x), bool)
        for e, v in zip(ends, m.predict(F).astype(bool)):
            arr[e - 1:e - 1 + args.eval_hop] = v
        preds[k].append(arr)
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(test)}")

# ---- results ------------------------------------------------------------------------------------------------------
none = [None] * len(test)
configs = [
    ("threshold alone (first-pass setting)", none, masks, 0, 0),
    ("A: threshold, then SVM confirms", preds["SVM (RBF)"], masks, 0, 0),
    ("A: threshold, then Random Forest", preds["Random Forest"], masks, 0, 0),
    ("ML-first: SVM positive for 200 ms", preds["SVM (RBF)"], preds["SVM (RBF)"], 200, 200),
    ("ML-first: SVM positive for 300 ms", preds["SVM (RBF)"], preds["SVM (RBF)"], 300, 300),
    ("ML-first: Random Forest, 300 ms", preds["Random Forest"], preds["Random Forest"], 300, 300),
]
elderly_adl = [i for i, t in enumerate(test) if t.kind != "fall" and t.subject["group"] == "elderly"]
print(f"\nheld-out subjects: {comp(test)}\n")
print(f"{'detector':40s} {'falls caught':>12s} {'no false alarm':>15s} {'mean lead':>10s} {'>=200 ms':>9s}")
results, fall_types = {}, {}
for name, pr, mk, d, m in configs:
    r = run_policy(test, mk, pr, d, m)
    results[name] = r
    extra = ""
    if elderly_adl:
        re = run_policy([test[i] for i in elderly_adl], [mk[i] for i in elderly_adl], [pr[i] for i in elderly_adl], d, m)
        extra = f"  (elderly: {100 * re['specificity']:.0f}%)"
    print(f"{name:40s} {100 * r['sensitivity']:11.1f}% {100 * r['specificity']:14.1f}% {r['lead_mean_ms']:8.0f} ms {100 * r['useful_rate']:8.1f}%{extra}")
    if name.startswith("A: threshold, then SVM"):
        for task in sorted({t.subject["task"] for t in test if t.kind == "fall"}):
            idx = [i for i, t in enumerate(test) if t.kind == "fall" and t.subject["task"] == task]
            rr = run_policy([test[i] for i in idx], [mk[i] for i in idx], [pr[i] for i in idx], d, m)
            fall_types[task] = (len(idx), rr["sensitivity"])
print("\nfalls caught by fall type, detector A (SVM); n = held-out recordings:")
print("  " + "  ".join(f"{k}: {100 * v[1]:.0f}% (n={v[0]})" for k, v in fall_types.items()))
fa = results["A: threshold, then SVM confirms"]["fa_by_kind"]
worst = sorted(fa.items(), key=lambda kv: -kv[1])[:5]
print("most-triggering normal activities for A: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in worst))

os.makedirs("data", exist_ok=True)
out = f"data/real_{args.dataset}_results.json"
with open(out, "w") as f:
    json.dump(dict(dataset=args.dataset, seed=args.seed, train=comp(train), test=comp(test),
                   threshold=vars(p_first), axis_map=amap.describe(), results=results,
                   fall_types={k: dict(n=v[0], sensitivity=v[1]) for k, v in fall_types.items()}), f, indent=1)
print(f"\nsaved {out}  ({time.time() - t0:.0f} s)")
