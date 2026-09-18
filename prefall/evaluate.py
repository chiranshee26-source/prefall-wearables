"""Trial-level metrics: sensitivity, specificity, lead time (slides 2 and 22 definitions)."""
import itertools
import numpy as np
from .config import FS, PREFALL_MIN_LEAD_S
from .threshold import Params, alarm_mask


def first_alarm(mask):
    return int(np.argmax(mask)) if mask.any() else None


def evaluate(trials, traces, p):
    """A fall counts as detected if the first alarm lands in [onset, impact). ADL trials with
    any alarm are false alarms."""
    leads, fall_n, adl_n = [], 0, 0
    adl_by_kind = {}
    for tr, (th, vd) in zip(trials, traces):
        m = alarm_mask(th, vd, p)
        if tr.kind == "fall":
            fall_n += 1
            t = np.arange(len(m)) / FS
            win = m & (t >= tr.t_onset) & (t < tr.t_impact)
            i = first_alarm(win)
            leads.append(tr.t_impact - i / FS if i is not None else None)
        else:
            adl_n += 1
            adl_by_kind.setdefault(tr.kind, [0, 0])
            adl_by_kind[tr.kind][1] += 1
            if m.any():
                adl_by_kind[tr.kind][0] += 1
    det = [l for l in leads if l is not None]
    fa = sum(v[0] for v in adl_by_kind.values())
    return dict(
        sensitivity=len(det) / max(fall_n, 1),
        specificity=1 - fa / max(adl_n, 1),
        lead_mean_ms=1000 * float(np.mean(det)) if det else 0.0,
        lead_median_ms=1000 * float(np.median(det)) if det else 0.0,
        useful_rate=sum(l >= PREFALL_MIN_LEAD_S for l in det) / max(fall_n, 1),
        fa_by_kind={k: v[0] / v[1] for k, v in adl_by_kind.items()},
        leads_ms=[1000 * l for l in det],
        n_fall=fall_n, n_adl=adl_n)


def calibrate(trials, traces, mode="standalone", min_spec=0.90, min_sens=0.95):
    """Grid-search (theta_crit, v_thr, tau), the 'calibrated tau' step on slide 14.

    standalone : maximise sensitivity s.t. specificity >= min_spec (threshold used on its own).
    first_pass : maximise the share of falls caught with >= 200 ms lead s.t. sensitivity >=
                 min_sens, then specificity. Accepts more false alarms because the ML stage
                 (step 2) is what confirms or rejects them.
    """
    best, best_key = None, None
    for tc, v, tau in itertools.product([10, 15, 20, 25, 30], [0.3, 0.5, 0.7, 0.9, 1.1],
                                        [3, 6, 10, 15, 20, 30]):
        p = Params(tc, v, tau)
        r = evaluate(trials, traces, p)
        if mode == "first_pass":
            key = (r["sensitivity"] >= min_sens, r["useful_rate"], r["specificity"])
        else:
            ok = r["specificity"] >= min_spec
            key = (ok, r["sensitivity"] if ok else r["sensitivity"] + r["specificity"], r["lead_mean_ms"])
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


def evaluate_window_model(pred, y, tid, t_end, trials, ids):
    """Score a per-window classifier on whole trials, same definitions as `evaluate`.

    Fall detected = any window labelled pre-fall (ends in [onset+50 ms, impact]) predicted 1;
    lead = impact time minus that window's end time. ADL trial with any predicted 1 = false alarm.
    """
    ids = set(int(i) for i in ids)
    leads, fall_n, adl_n, pre_onset = [], 0, 0, 0
    by = {}
    for i in sorted(ids):
        tr = trials[i]
        m = tid == i
        p, yy, te = pred[m], y[m], t_end[m]
        if tr.kind == "fall":
            fall_n += 1
            hit = (p == 1) & (yy == 1)
            leads.append(tr.t_impact - te[hit][0] if hit.any() else None)
            pre_onset += int(((p == 1) & (yy == 0)).any())
        else:
            adl_n += 1
            by.setdefault(tr.kind, [0, 0])
            by[tr.kind][1] += 1
            by[tr.kind][0] += int((p == 1).any())
    det = [l for l in leads if l is not None]
    sel = np.isin(tid, list(ids))
    P, N = (y[sel] == 1), (y[sel] == 0)
    fa = sum(v[0] for v in by.values())
    return dict(
        sensitivity=len(det) / max(fall_n, 1),
        specificity=1 - fa / max(adl_n, 1),
        lead_mean_ms=1000 * float(np.mean(det)) if det else 0.0,
        lead_median_ms=1000 * float(np.median(det)) if det else 0.0,
        useful_rate=sum(l >= PREFALL_MIN_LEAD_S for l in det) / max(fall_n, 1),
        fa_by_kind={k: v[0] / v[1] for k, v in by.items()},
        pre_onset_fa_falls=pre_onset,
        win_sensitivity=float((pred[sel][P] == 1).mean()),
        win_specificity=float((pred[sel][N] == 0).mean()),
        n_fall=fall_n, n_adl=adl_n)
