import numpy as np
import pytest
pytest.importorskip("tf_keras")

from prefall.synth import make_dataset
from prefall.dataset import build_raw_windows
from prefall.deep import cnn_bilstm, train
from prefall.deploy import (make_masks, KeepMasks, sparsity_of, to_tf_keras, convert, TFLiteRunner,
                            to_c_array)


def _data():
    trials = make_dataset(dict(walk=6, fall=10), seed=3)
    X, y, _, _ = build_raw_windows(trials, hop=10)
    return X, y


def test_pruning_reaches_target_and_masks_hold_during_finetune():
    X, y = _data()
    m = cnn_bilstm()
    pairs = make_masks(m, 0.35)
    assert 0.33 < sparsity_of(m) < 0.37
    train(m, X, y, X, y, epochs=1, verbose=0, callbacks=[KeepMasks(pairs)])
    assert sparsity_of(m) >= 0.34                 # fine-tuning did not revive pruned weights


def test_int8_model_fits_budget_uses_builtin_ops_and_matches_fp32():
    X, y = _data()
    m = cnn_bilstm()
    train(m, X, y, X, y, epochs=2, verbose=0)
    blob = convert(to_tf_keras(m), "int8", X[:50])
    assert len(blob) / 1024 < 50                  # slide 13 budget
    r = TFLiteRunner(blob)
    ops = r.ops()
    assert "UNIDIRECTIONAL_SEQUENCE_LSTM" in ops
    assert not any(o.startswith("Flex") or o == "WHILE" for o in ops)
    p8 = r.predict_proba(X[:100])[:, 1]
    p32 = m.predict(X[:100], verbose=0)[:, 1]
    assert np.abs(p8 - p32).mean() < 0.1
    assert to_c_array(blob).count("0x") == len(blob)
