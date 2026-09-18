"""Signal conditioning: anti-alias LPF + decimation, and the wrist band-pass (slide 17)."""
import numpy as np
from scipy import signal
from .config import FS_RAW, FS, LPF_HZ


def butter_lpf(x, fs, fc=LPF_HZ, order=2):
    """Causal 2nd-order Butterworth low-pass (causal on purpose: firmware cannot look ahead)."""
    wn = min(fc / (fs / 2.0), 0.99)
    b, a = signal.butter(order, wn, btype="low")
    zi = signal.lfilter_zi(b, a).reshape((-1,) + (1,) * (x.ndim - 1)) * x[0]   # no start-up transient
    return signal.lfilter(b, a, x, axis=0, zi=zi)[0]


def butter_bandpass(x, fs=FS, lo=0.5, hi=10.0, order=2):
    """0.5-10 Hz band-pass used by the wristband variant to suppress hand gestures."""
    b, a = signal.butter(order, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band")
    zi = signal.lfilter_zi(b, a).reshape((-1,) + (1,) * (x.ndim - 1)) * x[0]
    return signal.lfilter(b, a, x, axis=0, zi=zi)[0]


def condition(raw, fs_raw=FS_RAW, fs=FS):
    """Slide 9 front end: anti-alias LPF at the raw rate, then decimate to 50 Hz."""
    ratio = int(round(fs_raw / fs))
    return butter_lpf(raw, fs_raw)[::ratio]
