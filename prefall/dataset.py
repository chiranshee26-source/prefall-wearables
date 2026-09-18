"""Turn trials into windowed feature tables for the ML stage (step 2)."""
import numpy as np
from .config import FS, WIN
from .dsp import condition
from .features import extract, sliding_windows


# Fixed per-channel scaling (accel / 4 g, gyro / 250 deg/s). Fixed, not learned from data,
# so the exact same constants can be baked into the ESP32 firmware later.
SCALE = np.array([4, 4, 4, 250, 250, 250], dtype=np.float32)


def _labels(tr, te):
    lab = np.zeros(len(te), int)
    if tr.kind == "fall":
        lab[(te >= tr.t_onset + 0.05) & (te <= tr.t_impact)] = 1
        lab[te > tr.t_impact] = -1
    return lab


def build_raw_windows(trials, hop=5):
    """Same windows and labels as build_windows, but the raw scaled (50, 6) signal instead of
    the 12-D features. Row order is identical, so labels line up one to one."""
    X, y, tid, tend = [], [], [], []
    for i, tr in enumerate(trials):
        W, ends = sliding_windows(condition(tr.raw, tr.fs_raw), WIN, hop)
        te = ends / FS
        lab = _labels(tr, te)
        keep = lab >= 0
        X.append((W[keep] / SCALE).astype(np.float32)); y.append(lab[keep])
        tid.append(np.full(keep.sum(), i)); tend.append(te[keep])
    return np.concatenate(X), np.concatenate(y), np.concatenate(tid), np.concatenate(tend)


def build_windows(trials, hop=5):
    """Label 1 = window ends inside [onset+50 ms, impact]; 0 = ADL or pre-onset;
    -1 = post-impact (dropped). hop=5 (100 ms) because the deck's 25-sample hop (500 ms)
    is longer than the whole 200-400 ms pre-impact window."""
    X, y, tid, tend = [], [], [], []
    for i, tr in enumerate(trials):
        F, ends = extract(condition(tr.raw, tr.fs_raw), WIN, hop)
        te = ends / FS
        lab = np.zeros(len(te), int)
        if tr.kind == "fall":
            lab[(te >= tr.t_onset + 0.05) & (te <= tr.t_impact)] = 1
            lab[te > tr.t_impact] = -1
        keep = lab >= 0
        X.append(F[keep]); y.append(lab[keep]); tid.append(np.full(keep.sum(), i)); tend.append(te[keep])
    return np.vstack(X), np.concatenate(y), np.concatenate(tid), np.concatenate(tend)
