import numpy as np
from prefall.synth import make_trial
from prefall.dsp import condition
from prefall.features import extract, window_features, FEATURE_NAMES
from prefall.threshold import trace, alarm_mask, Params, Tracker
from prefall.config import FS, WIN

RNG = np.random.default_rng(0)


def test_conditioning_rate_and_no_startup_transient():
    x = condition(make_trial("walk", RNG).raw)
    assert x.shape == (600, 6)
    assert np.linalg.norm(x[0, :3]) > 0.5          # primed filter (walking swings |a| by ~0.3 g); the old bug gave ~0


def test_fall_matches_slide3_physics():
    for _ in range(10):
        tr = make_trial("fall", RNG)
        x = condition(tr.raw)
        a = np.linalg.norm(x[:, :3], axis=1)
        i0, i1 = int(tr.t_onset * FS), int(tr.t_impact * FS)
        assert a[i0:i1 + 1].min() < 0.6            # weightlessness dip, ~0.5 g
        assert a[i1:i1 + 6].max() > 3.0            # impact spike
        th, _ = trace(x)
        assert th[i1 + 5:].max() > 60              # ends near lying orientation


def test_feature_vector_is_12d_and_finite():
    F, ends = extract(condition(make_trial("walk", RNG).raw))
    assert F.shape[1] == len(FEATURE_NAMES) == 12
    assert np.isfinite(F).all()
    assert ends[0] == WIN


def test_tracker_is_causal():
    x = condition(make_trial("fall", RNG).raw)
    th_full, vd_full = trace(x)
    th_cut, vd_cut = trace(x[:300])
    assert np.allclose(th_full[:300], th_cut) and np.allclose(vd_full[:300], vd_cut)


def test_threshold_separates_jump_from_fall():
    p = Params(20, 0.7, 15)
    falls = [alarm_mask(*trace(condition(make_trial("fall", RNG).raw)), p).any() for _ in range(30)]
    jumps = [alarm_mask(*trace(condition(make_trial("jump", RNG).raw)), p).any() for _ in range(30)]
    assert np.mean(falls) > 0.8
    assert np.mean(jumps) < 0.1                    # deep free-fall but no trunk tilt
