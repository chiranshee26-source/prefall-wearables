"""Step 2c: the hybrid detector from slides 14 and 23.

Stage 1  the step 1 threshold runs on every sample. Each rising edge is a *trigger*
         (with a 1.5 s refractory period so one event is not counted twice).
Stage 2  after a trigger the ML model must confirm. Policy (d, m): wait d ms after the trigger,
         and require the model to say "pre-fall" on every window ending in the last m ms.
         (0, 0) = confirm immediately on the trigger window; (200, 200) = the model must stay
         positive for the whole 200 ms wait, which is the slide 23 confirmation window.

Model decisions are computed for a window ending at every sample (hop = 1), so decision times
are exact to 20 ms, unlike the hop = 5 training windows.
"""
import numpy as np
from .config import FS, WIN, PREFALL_MIN_LEAD_S
from .dataset import SCALE
from .features import sliding_windows, window_features

POLICIES = [(0, 0), (100, 0), (200, 0), (300, 0), (100, 100), (200, 200), (300, 300)]


def all_windows(streams):
    """Every 1 s window (hop 1) of every stream, plus which stream each came from."""
    Ws, owner = [], []
    for i, x in enumerate(streams):
        W, _ = sliding_windows(x, WIN, 1)
        Ws.append(W)
        owner.append(np.full(len(W), i))
    return np.concatenate(Ws), np.concatenate(owner)


def per_stream(flat_pred, owner, streams):
    """Split flat window predictions back per stream, aligned to samples:
    out[i][j] is the decision of the window ending at sample j (False before the first full window)."""
    out = []
    for i, x in enumerate(streams):
        p = np.zeros(len(x), bool)
        p[WIN - 1:] = flat_pred[owner == i].astype(bool)
        out.append(p)
    return out


def feature_pred(model, W):
    return model.predict(np.stack([window_features(w) for w in W]))


def deep_pred(model, W, thr=0.5):
    return model.predict((W / SCALE).astype(np.float32), batch_size=2048, verbose=0)[:, 1] > thr


def episodes(mask, refractory=int(1.5 * FS)):
    """Sample indices where the threshold alarm switches on, ignoring re-triggers within 1.5 s."""
    idx = np.flatnonzero(mask[1:] & ~mask[:-1]) + 1
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= refractory:
            out.append(int(i))
            last = i
    return out


def confirmed_alarms(mask, pred, d_ms, m_ms):
    """Alarm sample indices after applying policy (d, m). pred=None means threshold only."""
    if pred is None:
        return episodes(mask)
    d, m = round(d_ms * FS / 1000), round(m_ms * FS / 1000)
    out = []
    for e in episodes(mask):
        a = e + d
        if a < len(pred) and a - m >= WIN - 1 and pred[a - m:a + 1].all():
            out.append(a)
    return out


def run_policy(trials, masks, preds, d_ms=0, m_ms=0):
    leads, fall_n, adl_n, pre_onset, by = [], 0, 0, 0, {}
    for tr, mask, pred in zip(trials, masks, preds):
        t = np.array(confirmed_alarms(mask, pred, d_ms, m_ms)) / FS
        if tr.kind == "fall":
            fall_n += 1
            hit = t[(t >= tr.t_onset) & (t < tr.t_impact)]
            leads.append(tr.t_impact - hit[0] if len(hit) else None)
            pre_onset += int((t < tr.t_onset).any())
        else:
            adl_n += 1
            by.setdefault(tr.kind, [0, 0])
            by[tr.kind][1] += 1
            by[tr.kind][0] += int(len(t) > 0)
    det = [l for l in leads if l is not None]
    fa = sum(v[0] for v in by.values())
    return dict(
        sensitivity=len(det) / max(fall_n, 1),
        specificity=1 - fa / max(adl_n, 1),
        lead_mean_ms=1000 * float(np.mean(det)) if det else 0.0,
        useful_rate=sum(l >= PREFALL_MIN_LEAD_S for l in det) / max(fall_n, 1),
        fa_by_kind={k: v[0] / v[1] for k, v in by.items()},
        pre_onset_fa_falls=pre_onset, n_fall=fall_n, n_adl=adl_n)
