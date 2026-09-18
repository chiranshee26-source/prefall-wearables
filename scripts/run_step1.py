"""Step 1 end to end: synth data -> slide 9 front end -> Triangle threshold -> metrics + figures.

    python scripts/run_step1.py
"""
import _bootstrap  # noqa: F401  (repo root on sys.path, cwd = repo root)
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prefall.synth import make_dataset, make_trial
from prefall.dsp import condition
from prefall.threshold import trace, alarm_mask
from prefall.evaluate import evaluate, calibrate
from prefall.dataset import build_windows
from prefall.config import FS

COUNTS = dict(walk=60, stairs=40, sit_fast=60, jump=40, stumble=40, fall=160)

trials = make_dataset(COUNTS, seed=7)
x50 = [condition(t.raw) for t in trials]
traces = [trace(x) for x in x50]

half = len(trials) // 2                      # split by trial so no window leakage
tr_tr, te_tr = trials[:half], trials[half:]
tr_tc, te_tc = traces[:half], traces[half:]

def report(name, p):
    r = evaluate(te_tr, te_tc, p)
    print(f"\n[{name}] theta_crit={p.theta_crit} deg, v_thr={p.v_thr} m/s, tau={p.tau}   "
          f"(calibrated on {len(tr_tr)} trials, tested on {r['n_fall']} falls + {r['n_adl']} ADL)")
    print(f"  sensitivity {100*r['sensitivity']:.1f}%   specificity {100*r['specificity']:.1f}%")
    print(f"  lead time mean {r['lead_mean_ms']:.0f} ms (median {r['lead_median_ms']:.0f} ms), "
          f"falls with lead >= 200 ms: {100*r['useful_rate']:.1f}%")
    print("  false alarms by ADL kind:", {k: f"{100*v:.0f}%" for k, v in r["fa_by_kind"].items()})
    return r


p_solo = calibrate(tr_tr, tr_tc, mode="standalone", min_spec=0.90)
p = calibrate(tr_tr, tr_tc, mode="first_pass", min_sens=0.95)
report("standalone threshold (spec >= 90%)", p_solo)
r = report("first-pass threshold (feeds the ML confirmer)", p)

# ---- figures -------------------------------------------------------------------------------
rng = np.random.default_rng(3)
fig, axes = plt.subplots(4, 2, figsize=(11, 8), sharex="col")
for col, kind in enumerate(["fall", "sit_fast"]):
    tr = make_trial(kind, rng)
    x = condition(tr.raw); th, vd = trace(x); t = np.arange(len(x)) / FS
    m = alarm_mask(th, vd, p)
    a = np.linalg.norm(x[:, :3], axis=1)
    axes[0, col].plot(t, a); axes[0, col].axhline(0.5, ls=":", c="gray"); axes[0, col].set_ylabel("|a| (g)")
    axes[1, col].plot(t, th); axes[1, col].axhline(p.theta_crit, ls=":", c="gray"); axes[1, col].set_ylabel("trunk angle (deg)")
    axes[2, col].plot(t, vd); axes[2, col].axhline(p.v_thr, ls=":", c="gray"); axes[2, col].set_ylabel("descent (m/s)")
    axes[3, col].plot(t, th * vd); axes[3, col].axhline(p.tau, ls=":", c="gray"); axes[3, col].set_ylabel("T = theta*Vy")
    axes[3, col].set_xlabel("time (s)")
    lo, hi = (tr.t_onset - 1.5, tr.t_impact + 1.0) if kind == "fall" else (t[np.argmax(th > 15)] - 1.5, t[np.argmax(th > 15)] + 3)
    for ax in axes[:, col]:
        if m.any():
            ax.axvline(np.argmax(m) / FS, c="red", lw=1)
        if kind == "fall":
            ax.axvspan(tr.t_onset, tr.t_impact, color="orange", alpha=0.15)
        ax.set_xlim(lo, hi)
    axes[0, col].set_title(f"{kind}" + ("  (orange = pre-impact window, red = alarm)" if kind == "fall" else "  (hard negative)"))
fig.tight_layout(); fig.savefig("docs/images/threshold_traces.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(r["leads_ms"], bins=20); ax.axvspan(200, 400, color="green", alpha=0.12)
ax.set_xlabel("lead time before impact (ms)"); ax.set_ylabel("falls"); ax.set_title("Threshold detector lead time (test set)")
fig.tight_layout(); fig.savefig("docs/images/lead_time_hist.png", dpi=130); plt.close(fig)

# ---- hand-off to step 2 (ML) ---------------------------------------------------------------
X, y, tid, tend = build_windows(trials, hop=5)
np.savez_compressed("data/step1_windows.npz", X=X, y=y, trial_id=tid, t_end=tend,
                    train_mask=tid < half)
with open("data/step1_trials.pkl", "wb") as f:
    pickle.dump(trials, f)
print(f"saved {len(X)} windows ({int((y==1).sum())} pre-fall, {int((y==0).sum())} normal) -> data/")
