"""Synthetic trunk-mounted IMU generator.

Physics follows slide 3: perturbation -> instability -> pre-impact (free-fall dip, tilt > 30 deg)
-> impact spike. ADL classes include hard negatives that the threshold must not fire on
(fast sitting, jumping, stairs) plus a recovered stumble.

Frame: accelerometer reads +1 g on +z when upright. Forward tilt theta gives ax = sin(theta).
Gyro-y is d(theta)/dt in deg/s. Real MPU-6050 axis signs depend on mounting; flip in firmware.
"""
from dataclasses import dataclass
import numpy as np
from .config import FS_RAW

ADL_KINDS = ["walk", "stairs", "sit_fast", "jump", "stumble"]
ALL_KINDS = ADL_KINDS + ["fall"]


@dataclass
class Trial:
    kind: str
    raw: np.ndarray            # (n, 6) at FS_RAW: ax, ay, az [g], gx, gy, gz [deg/s]
    t_onset: float | None      # fall onset (s), falls only
    t_impact: float | None     # impact time (s), falls only
    subject: dict
    fs_raw: float = FS_RAW     # sampling rate of `raw` (real datasets differ: 100, 200, 238 Hz)


def _subject(rng):
    return dict(cadence=rng.uniform(85, 115), amp=rng.uniform(0.15, 0.30),
                tilt0=rng.uniform(0, 6), noise=rng.uniform(0.8, 1.4))


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _bump(t, c, w):
    return np.exp(-0.5 * ((t - c) / w) ** 2)


def _walk(t, s, rng, amp_scale=1.0):
    f = s["cadence"] / 60.0
    ph = rng.uniform(0, 2 * np.pi)
    A = s["amp"] * amp_scale
    az = A * np.sin(2 * np.pi * f * t + ph) + 0.3 * A * np.sin(4 * np.pi * f * t + 2 * ph)
    fx = 0.4 * A * np.sin(2 * np.pi * f * t + ph + 1.0)
    theta = s["tilt0"] + 2.0 * np.sin(np.pi * f * t + ph / 2)
    return theta, fx, az


def _compose(t, theta, fx, az_w, s, rng):
    th = np.deg2rad(theta)
    fz = 1.0 + az_w
    ax = fx * np.cos(th) + fz * np.sin(th)
    az = -fx * np.sin(th) + fz * np.cos(th)
    ay = 0.05 * np.sin(2 * np.pi * 0.85 * t + rng.uniform(0, 6.28))
    gy = np.gradient(theta, t)
    gx = 8.0 * np.sin(2 * np.pi * 0.85 * t + rng.uniform(0, 6.28))
    gz = 4.0 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 6.28))
    n = s["noise"]
    acc = np.stack([ax, ay, az], 1) + rng.normal(0, 0.02 * n, (len(t), 3))
    gyr = np.stack([gx, gy, gz], 1) + rng.normal(0, 1.5 * n, (len(t), 3))
    # MPU-6050 configured range (slide 6): +/-8 g, +/-500 deg/s
    return np.hstack([np.clip(acc, -8, 8), np.clip(gyr, -500, 500)])


def _idx(t, t0):
    return min(int(np.searchsorted(t, t0)), len(t) - 1)


def _ringing(t, t0, amp, decay=0.2, f=6.0):
    d = np.clip(t - t0, 0, None)
    return np.where(t >= t0, amp * np.exp(-d / decay) * np.sin(2 * np.pi * f * d), 0.0)


def _fall(t, s, rng):
    theta, fx, az = _walk(t, s, rng)
    t0 = rng.uniform(4, 8)
    T = rng.uniform(0.45, 0.70)
    ti = t0 + T
    sign = rng.choice([-1, 1])
    dip = rng.uniform(0.5, 0.85)
    th_end = rng.uniform(80, 90)
    tau = np.clip((t - t0) / T, 0, 1)
    th_w = theta[_idx(t, t0)]
    theta = np.where(t < t0, theta, th_w + sign * (th_end - th_w) * tau ** 2)
    theta = theta - sign * 3.0 * np.exp(-np.clip(t - ti, 0, None) / 0.08) * (t >= ti) * np.abs(
        np.sin(2 * np.pi * 4 * np.clip(t - ti, 0, None)))
    az_fall = az * (1 - tau) + (-dip) * tau ** 1.5
    az_w = np.where(t < t0, az, np.where(t < ti, az_fall, 0.0))
    az_w = az_w + rng.uniform(4.0, 7.0) * _bump(t, ti + 0.01, 0.02) + _ringing(t, ti, 0.4)
    fx = fx - 0.5 * sign * _bump(t, t0 + 0.03, 0.03)
    return theta, fx, az_w, t0, ti


def _sit_fast(t, s, rng):
    theta, fx, az = _walk(t, s, rng)
    t0, T, L = rng.uniform(3, 7), rng.uniform(0.5, 0.9), rng.uniform(15, 35)
    u = (t - t0) / T
    theta = theta + L * _ss(u / 0.7) - (L - 8) * _ss((t - (t0 + T)) / 1.2)
    env = 1 - _ss((t - t0) / 0.5)
    az_w = az * env - 0.35 * np.sin(np.pi * np.clip(u / 0.8, 0, 1)) + 0.6 * _bump(t, t0 + T, 0.05)
    return theta, fx * env, az_w


def _stumble(t, s, rng):
    theta, fx, az = _walk(t, s, rng)
    t0, P, T1 = rng.uniform(4, 8), rng.uniform(18, 40), rng.uniform(0.25, 0.4)
    delta = P * _ss((t - t0) / T1) * (1 - _ss((t - t0 - T1) / 0.4))
    dip = rng.uniform(0.2, 0.5)
    az_w = az - dip * np.sin(np.pi * np.clip((t - t0) / (T1 + 0.4), 0, 1))
    fx = fx - 0.4 * _bump(t, t0 + 0.03, 0.03)
    return theta + delta, fx, az_w


def _jump(t, s, rng):
    theta, fx, az = _walk(t, s, rng)
    t0 = rng.uniform(3, 7)
    ta = t0 + 0.45
    F = rng.uniform(0.3, 0.5)
    tl = ta + F
    env = 1 - _ss((t - t0) / 0.3)
    az_w = az * env
    az_w = az_w - 0.3 * np.sin(np.pi * np.clip((t - t0) / 0.3, 0, 1))
    az_w = az_w + 1.0 * np.sin(np.pi * np.clip((t - t0 - 0.3) / 0.15, 0, 1))
    az_w = np.where((t >= ta) & (t < tl), -0.97, az_w)
    az_w = az_w + rng.uniform(2.0, 3.5) * _bump(t, tl + 0.02, 0.02) + _ringing(t, tl, 0.3)
    theta = theta + 4 * np.sin(np.pi * np.clip((t - t0) / (tl - t0 + 0.3), 0, 1))
    return theta, fx * env, az_w


def make_trial(kind, rng, dur=12.0, fs=FS_RAW):
    t = np.arange(int(dur * fs)) / fs
    s = _subject(rng)
    t0 = ti = None
    if kind == "walk":
        theta, fx, az = _walk(t, s, rng)
    elif kind == "stairs":
        theta, fx, az = _walk(t, s, rng, amp_scale=2.2)
        theta = theta + 6.0
    elif kind == "sit_fast":
        theta, fx, az = _sit_fast(t, s, rng)
    elif kind == "jump":
        theta, fx, az = _jump(t, s, rng)
    elif kind == "stumble":
        theta, fx, az = _stumble(t, s, rng)
    elif kind == "fall":
        theta, fx, az, t0, ti = _fall(t, s, rng)
    else:
        raise ValueError(kind)
    return Trial(kind, _compose(t, theta, fx, az, s, rng), t0, ti, s)


def make_dataset(counts, seed=0):
    rng = np.random.default_rng(seed)
    trials = []
    for kind, n in counts.items():
        trials += [make_trial(kind, rng) for _ in range(n)]
    order = rng.permutation(len(trials))
    return [trials[i] for i in order]
