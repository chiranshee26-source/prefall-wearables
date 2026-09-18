"""Step 3: prune, quantize to INT8, convert to TFLite, and measure what was lost.

    python scripts/run_step3.py     (needs scripts/run_step1.py and scripts/run_step2b.py first; pip install tf_keras)
"""
import _bootstrap  # noqa: F401  (repo root on sys.path, cwd = repo root)
import json
import os
import pickle
import numpy as np
import keras

from prefall.dataset import build_raw_windows
from prefall.deep import train, predict
from prefall.deploy import (make_masks, KeepMasks, sparsity_of, to_tf_keras, convert, TFLiteRunner,
                            gzip_kb, to_c_array)
from prefall.evaluate import evaluate_window_model

if not os.path.exists("data/cnn_bilstm.keras"):
    raise SystemExit("missing data/cnn_bilstm.keras: run scripts/run_step2b.py first")
with open("data/step1_trials.pkl", "rb") as f:
    trials = pickle.load(f)
half = len(trials) // 2
X, y, tid, tend = build_raw_windows(trials, hop=5)
fit_ids, val_ids = np.arange(int(0.8 * half)), np.arange(int(0.8 * half), half)
test_ids = np.arange(half, len(trials))
m_fit, m_val, m_test = (np.isin(tid, i) for i in (fit_ids, val_ids, test_ids))
rng = np.random.default_rng(0)
rep_X = X[m_fit][rng.choice(m_fit.sum(), 300, replace=False)]      # calibration windows for INT8


def score(name, pred_test, base=None):
    pred = np.zeros(len(y), int)
    pred[m_test] = pred_test
    r = evaluate_window_model(pred, y, tid, tend, trials, test_ids)
    agree = "" if base is None else f"  agrees with FP32 on {100*np.mean(pred_test == base):.1f}% of windows"
    print(f"  {name:28s} trial sens {100*r['sensitivity']:5.1f}%  spec {100*r['specificity']:5.1f}%  "
          f"lead {r['lead_mean_ms']:3.0f} ms | window sens {100*r['win_sensitivity']:5.1f}%  "
          f"spec {100*r['win_specificity']:5.1f}%{agree}")
    return r


results = {}
model = keras.saving.load_model("data/cnn_bilstm.keras")
print("accuracy on the held-out test trials:")
base = predict(model, X[m_test])
results["fp32"] = score("FP32 (Keras)", base)

# 1. magnitude pruning (slide 13: bottom 30-40% of weights) + short fine-tune with masks held
pruned = keras.models.clone_model(model)
pruned.set_weights(model.get_weights())
pairs = make_masks(pruned, 0.35)
train(pruned, X[m_fit], y[m_fit], X[m_val], y[m_val], epochs=8, lr=3e-4, verbose=0, callbacks=[KeepMasks(pairs)])
print(f"\npruned to {100*sparsity_of(pruned):.0f}% zero weights")
results["fp32_pruned"] = score("FP32 pruned 35% + fine-tune", predict(pruned, X[m_test]), base)

# 2. INT8 conversion of both
blobs = {}
for tag, m in [("dense", model), ("pruned", pruned)]:
    tfk = to_tf_keras(m)
    blobs[tag] = {"float32": convert(tfk, "float32"), "int8": convert(tfk, "int8", rep_X)}
print("\nmodel file sizes (KB):")
for tag in blobs:
    for mode, b in blobs[tag].items():
        print(f"  {tag:7s} {mode:8s} {len(b)/1024:6.1f} KB   gzip {gzip_kb(b):6.1f} KB")

# 3. run the real INT8 models
print("\nINT8 accuracy (the .tflite file itself, one window at a time):")
runner = {tag: TFLiteRunner(blobs[tag]["int8"]) for tag in blobs}
results["int8_dense"] = score("INT8 (no pruning)", runner["dense"].predict(X[m_test]), base)
results["int8_pruned"] = score("INT8 (pruned + INT8)", runner["pruned"].predict(X[m_test]), base)

ops = runner["pruned"].ops()
print("\nops in the INT8 model:", ", ".join(ops))
bad = [o for o in ops if o in ("WHILE", "TensorListReserve", "FlexTensorListReserve") or o.startswith("Flex")]
print("uses Select-TF/Flex ops:", bool(bad))

os.makedirs("export", exist_ok=True)
open("export/cnn_bilstm_int8.tflite", "wb").write(blobs["dense"]["int8"])
open("export/cnn_bilstm_int8_model.h", "w").write(to_c_array(blobs["dense"]["int8"]))
print("\nwrote export/cnn_bilstm_int8.tflite and export/cnn_bilstm_int8_model.h")
results["sizes_kb"] = {t: {m: len(b) / 1024 for m, b in d.items()} for t, d in blobs.items()}
results["ops"] = ops
with open("data/step3_results.json", "w") as f:
    json.dump(results, f, indent=1)
