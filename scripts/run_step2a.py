"""Step 2a: train and score the classical models on the step 1 windows.

    python scripts/run_step2a.py        (run scripts/run_step1.py first; it writes data/step1_windows.npz)
"""
import _bootstrap  # noqa: F401  (repo root on sys.path, cwd = repo root)
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prefall.classical import make_models, forest_kb
from prefall.evaluate import evaluate_window_model
from prefall.features import FEATURE_NAMES

d = np.load("data/step1_windows.npz")
X, y, tid, tend, train_mask = d["X"], d["y"], d["trial_id"], d["t_end"], d["train_mask"]
with open("data/step1_trials.pkl", "rb") as f:
    trials = pickle.load(f)

train_ids = np.unique(tid[train_mask])
test_ids = np.unique(tid[~train_mask])          # same held-out trials as step 1
print(f"train: {len(train_ids)} trials, {train_mask.sum()} windows | "
      f"test: {len(test_ids)} trials, {(~train_mask).sum()} windows")

models = make_models()
results = {}
for name, m in models.items():
    m.fit(X[train_mask], y[train_mask])
    pred = m.predict(X)
    r = evaluate_window_model(pred, y, tid, tend, trials, test_ids)
    results[name] = r
    print(f"\n[{name}]")
    print(f"  trial level : sensitivity {100*r['sensitivity']:.1f}%  specificity {100*r['specificity']:.1f}%  "
          f"lead mean {r['lead_mean_ms']:.0f} ms  lead>=200ms {100*r['useful_rate']:.1f}%")
    print(f"  window level: sensitivity {100*r['win_sensitivity']:.1f}%  specificity {100*r['win_specificity']:.1f}%")
    print("  false alarms by ADL kind:", {k: f"{100*v:.0f}%" for k, v in r["fa_by_kind"].items()})

rf = models["Random Forest"]
nodes, kb = forest_kb(rf)
print(f"\nRandom Forest: {nodes} nodes, roughly {kb:.0f} KB if stored as 12-byte nodes (slide 22 says ~12 KB)")
imp = rf.feature_importances_
order = np.argsort(imp)[::-1]
print("Gini feature importance:", ", ".join(f"{FEATURE_NAMES[i]} {imp[i]:.2f}" for i in order[:6]))

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([FEATURE_NAMES[i] for i in order[::-1]], imp[order[::-1]])
ax.set_xlabel("Gini importance"); ax.set_title("Random Forest feature importance")
fig.tight_layout(); fig.savefig("docs/images/rf_feature_importance.png", dpi=130); plt.close(fig)

with open("data/step2a_results.json", "w") as f:
    json.dump(results, f, indent=1)
