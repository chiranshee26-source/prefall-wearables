"""1 s sliding windows and the 12-D feature vector (slides 9 and 11).

Features are orientation-invariant (resultant magnitudes) so the same code serves the
wristband variant. Signal Magnitude Area follows the slide 9 definition.
"""
import numpy as np
from .config import FS, G, WIN, HOP

FEATURE_NAMES = ["acc_mean", "acc_var", "acc_rms", "acc_min", "acc_max", "sma",
                 "zcr", "dom_freq", "gyro_mean", "gyro_max", "tilt_max", "vdesc_max"]


def window_features(w, fs=FS):
    acc, gyr = w[:, :3], w[:, 3:]
    a = np.linalg.norm(acc, axis=1)
    g = np.linalg.norm(gyr, axis=1)
    ac = a - a.mean()
    zcr = np.mean(np.signbit(ac[1:]) != np.signbit(ac[:-1]))
    spec = np.abs(np.fft.rfft(ac))
    freqs = np.fft.rfftfreq(len(a), 1.0 / fs)
    dom = freqs[1 + int(np.argmax(spec[1:]))]
    tilt = np.degrees(np.arctan2(np.hypot(acc[:, 0], acc[:, 1]), acc[:, 2]))
    vd = -np.cumsum((a - 1.0) * G / fs)           # descent-speed proxy, m/s
    return np.array([a.mean(), a.var(), np.sqrt(np.mean(a ** 2)), a.min(), a.max(),
                     np.mean(np.abs(acc).sum(axis=1)), zcr, dom, g.mean(), g.max(),
                     tilt.max(), vd.max()])


def sliding_windows(x, win=WIN, hop=HOP):
    ends = np.arange(win, len(x) + 1, hop)
    return np.stack([x[e - win:e] for e in ends]), ends


def extract(x, win=WIN, hop=HOP):
    W, ends = sliding_windows(x, win, hop)
    return np.stack([window_features(w) for w in W]), ends
