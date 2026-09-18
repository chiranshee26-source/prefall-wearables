import numpy as np
import pytest
pytest.importorskip("keras")

from prefall.synth import make_dataset
from prefall.dataset import build_windows, build_raw_windows
from prefall.deep import cnn_bilstm, convlstm, train, predict, size_kb


def test_raw_windows_align_with_feature_windows():
    trials = make_dataset(dict(walk=3, fall=4), seed=2)
    _, y_f, tid_f, te_f = build_windows(trials, hop=5)
    X, y, tid, te = build_raw_windows(trials, hop=5)
    assert X.shape[1:] == (50, 6)
    assert np.array_equal(y, y_f) and np.array_equal(tid, tid_f) and np.allclose(te, te_f)


@pytest.mark.parametrize("build", [cnn_bilstm, convlstm])
def test_models_train_and_predict(build):
    trials = make_dataset(dict(walk=6, fall=10), seed=3)
    X, y, tid, _ = build_raw_windows(trials, hop=10)
    m = build()
    assert m.output_shape == (None, 2)
    assert size_kb(m)[2] < 50                      # ideal INT8 size must fit the slide 13 budget
    train(m, X, y, X, y, epochs=1, verbose=0)
    p = predict(m, X)
    assert p.shape == y.shape and set(np.unique(p)) <= {0, 1}
