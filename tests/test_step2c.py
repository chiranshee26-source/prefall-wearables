import numpy as np
from types import SimpleNamespace as NS
from prefall.hybrid import episodes, confirmed_alarms, run_policy
from prefall.config import FS


def _mask(n, on):
    m = np.zeros(n, bool)
    m[on] = True
    return m


def test_episodes_use_rising_edges_and_refractory():
    m = _mask(600, list(range(100, 110)) + list(range(130, 140)) + list(range(300, 310)))
    assert episodes(m) == [100, 300]              # 130 is within 1.5 s of 100


def test_confirmation_delay_and_persistence():
    n = 600
    mask = _mask(n, range(200, 210))
    pred = _mask(n, range(200, 230))              # model positive from 200 to 229
    assert confirmed_alarms(mask, pred, 0, 0) == [200]
    assert confirmed_alarms(mask, pred, 200, 200) == [210]     # 10 samples = 200 ms, positive throughout
    pred2 = _mask(n, list(range(200, 205)) + list(range(206, 230)))   # one negative window inside
    assert confirmed_alarms(mask, pred2, 200, 200) == []
    assert confirmed_alarms(mask, pred2, 200, 0) == [210]


def test_run_policy_lead_and_false_alarm():
    n = 600
    fall = NS(kind="fall", t_onset=180 / FS, t_impact=230 / FS)
    adl = NS(kind="stumble", t_onset=None, t_impact=None)
    mask = _mask(n, range(200, 210))
    on, off = _mask(n, range(150, 300)), np.zeros(n, bool)
    r = run_policy([fall, adl], [mask, mask], [on, off], 0, 0)
    assert r["sensitivity"] == 1.0 and abs(r["lead_mean_ms"] - 600) < 1   # (230-200)/50 s
    assert r["specificity"] == 1.0                                         # model rejects the ADL trigger
    r2 = run_policy([fall, adl], [mask, mask], [on, on], 0, 0)
    assert r2["specificity"] == 0.0
