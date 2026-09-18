import numpy as np
from prefall.synth import make_dataset
from prefall.dataset import build_windows
from prefall.classical import make_models, forest_kb
from prefall.evaluate import evaluate_window_model


def test_random_forest_learns_prefall_windows():
    trials = make_dataset(dict(walk=10, jump=10, sit_fast=10, fall=40), seed=1)
    X, y, tid, te = build_windows(trials, hop=5)
    tr = tid < 35
    rf = make_models()["Random Forest"].fit(X[tr], y[tr])
    r = evaluate_window_model(rf.predict(X), y, tid, te, trials, np.unique(tid[~tr]))
    assert r["sensitivity"] > 0.8
    assert r["win_specificity"] > 0.9
    assert forest_kb(rf)[0] > 0
