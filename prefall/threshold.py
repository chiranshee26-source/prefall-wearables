"""Method A from slide 14: Triangle Feature threshold, T = theta * Vy.

Sample-by-sample and causal, so it maps 1:1 onto an ESP32 loop. Trunk angle comes from a
complementary filter (gyro integration corrected by accel tilt whenever |a| is near 1 g;
during free-fall the accel is untrustworthy so gyro carries the estimate). Vertical velocity
is the leaky integral of (a_vertical - 1 g), the slide 3 relation.
"""
from dataclasses import dataclass
import numpy as np
from .config import FS, G


class Tracker:
    def __init__(self, fs=FS, alpha=0.98, leak_tau=0.6):
        self.dt, self.alpha, self.leak_tau = 1.0 / fs, alpha, leak_tau
        self.reset()

    def reset(self):
        self.theta = None
        self.vy = 0.0

    def update(self, s):
        ax, ay, az, gx, gy, gz = s
        a = np.sqrt(ax * ax + ay * ay + az * az)
        th_acc = np.degrees(np.arctan2(ax, az))
        if self.theta is None:
            self.theta = th_acc
        else:
            pred = self.theta + gy * self.dt
            self.theta = self.alpha * pred + (1 - self.alpha) * th_acc if 0.7 < a < 1.3 else pred
        r = np.radians(self.theta)
        a_v = ax * np.sin(r) + az * np.cos(r)
        self.vy += (a_v - 1.0) * G * self.dt
        self.vy -= self.vy * self.dt / self.leak_tau
        return self.theta, max(-self.vy, 0.0)      # (trunk angle deg, descent speed m/s)


def trace(x):
    """Run the tracker over a whole (n, 6) stream. Returns |theta| and descent speed arrays."""
    tr = Tracker()
    out = np.array([tr.update(s) for s in x])
    return np.abs(out[:, 0]), out[:, 1]


@dataclass
class Params:
    theta_crit: float = 20.0    # deg
    v_thr: float = 0.7          # m/s
    tau: float = 10.0           # deg * m/s


def alarm_mask(theta, vd, p):
    """Slide 14: IF trunk_angle > theta_crit AND Vy > v_threshold AND T > tau."""
    return (theta > p.theta_crit) & (vd > p.v_thr) & (theta * vd > p.tau)
