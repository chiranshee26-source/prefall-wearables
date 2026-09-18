"""Step 2b: train the deep models and score them like step 2a.

    python scripts/run_step2b.py [epochs]     (needs scripts/run_step1.py first, and: pip install tensorflow)
"""
import _bootstrap  # noqa: F401  (repo root on sys.path, cwd = repo root)
import json
import pickle
import sys
import time
import numpy as np

from prefall.dataset import build_raw_windows
from prefall.deep import cnn_bilstm, convlstm, train, predict, size_kb
from prefall.evaluate import evaluate_window_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

with open("data/step1_trials.pkl", "rb") as f:
    trials = pickle.load(f)
half = len(trials) // 2                          # same split as steps 1 and 2a
X, y, tid, tend = build_raw_windows(trials, hop=5)
train_ids, test_ids = np.arange(half), np.arange(half, len(trials))
val_ids = train_ids[int(0.8 * half):]            # last 20% of training trials for early stopping
fit_ids = train_ids[:int(0.8 * half)]
m_fit, m_val = np.isin(tid, fit_ids), np.isin(tid, val_ids)
print(f"windows {X.shape}, fit {m_fit.sum()}, val {m_val.sum()}, test trials {len(test_ids)}")

results = {}
for name, build in [("1D-CNN + BiLSTM", cnn_bilstm), ("ConvLSTM", convlstm)]:
    model = build()
    n, fp32, int8 = size_kb(model)
    print(f"\n[{name}] {n} parameters, {fp32:.0f} KB FP32, about {int8:.0f} KB if INT8")
    t0 = time.time()
    h = train(model, X[m_fit], y[m_fit], X[m_val], y[m_val], epochs=EPOCHS, verbose=0)
    print(f"  trained {len(h.history['loss'])} epochs in {time.time()-t0:.0f} s, best val loss {min(h.history['val_loss']):.4f}")
    pred = predict(model, X)
    r = evaluate_window_model(pred, y, tid, tend, trials, test_ids)
    r["params"], r["kb_int8_ideal"] = n, int8
    results[name] = r
    print(f"  trial level : sensitivity {100*r['sensitivity']:.1f}%  specificity {100*r['specificity']:.1f}%  "
          f"lead mean {r['lead_mean_ms']:.0f} ms  lead>=200ms {100*r['useful_rate']:.1f}%")
    print(f"  window level: sensitivity {100*r['win_sensitivity']:.1f}%  specificity {100*r['win_specificity']:.1f}%")
    print("  false alarms by ADL kind:", {k: f"{100*v:.0f}%" for k, v in r["fa_by_kind"].items()})
    model.save(f"data/{model.name}.keras")

with open("data/step2b_results.json", "w") as f:
    json.dump(results, f, indent=1)
