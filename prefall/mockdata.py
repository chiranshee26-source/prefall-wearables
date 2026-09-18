"""Writes FAKE datasets in the SisFall and KFall file formats, from the simulator.

Purpose: test the loaders and rehearse the commands before the real data arrives. The recordings are written
in a deliberately different native axis frame and in raw units, so the loaders' unit conversion and axis-map
estimation are really exercised. Nothing here is real data, and results on it mean nothing about real falls.
"""
import os

import numpy as np

from .realdata import SIS_ACC_G_PER_LSB, SIS_GYR_DPS_PER_LSB
from .synth import make_trial

SIS_CODES = {"walk": "D01", "stairs": "D05", "sit_fast": "D08", "stumble": "D18", "jump": "D19"}
SIS_FALL_CODES = ["F01", "F04"]                       # forward fall while walking (slip / trip)
KF_TASKS = {"walk": 1, "sit_fast": 10, "jump": 17, "stumble": 19}
KF_FALL_TASKS = [20, 21]


def _sis_native(raw):
    """algorithm frame -> a scrambled SisFall-like native frame (gravity on -y, forward on z)."""
    ax, ay, az, gx, gy, gz = raw.T
    acc = np.stack([ay, -az, -ax], 1)
    gyr = np.stack([gx, gz, -gy], 1)
    return acc, gyr


def _kf_native(raw):
    """algorithm frame -> a scrambled KFall-like native frame (gravity on -x, forward on z)."""
    ax, ay, az, gx, gy, gz = raw.T
    acc = np.stack([-az, -ay, ax], 1)
    gyr = np.stack([gz, gy, gx], 1)
    return acc, gyr


def write_mock_sisfall(root, n_subjects=4, seed=0):
    """SisFall layout: <root>/<subject>/<CODE>_<SUBJECT>_R<NN>.txt, raw ADC counts, lines end with ';'."""
    rng = np.random.default_rng(seed)
    meta = []
    subjects = [f"SA{i:02d}" for i in range(1, n_subjects + 1)] + ["SE01"]
    for subj in subjects:
        os.makedirs(os.path.join(root, subj), exist_ok=True)
        plan = [(k, c) for k, c in SIS_CODES.items()]
        if subj.startswith("SA"):                      # elderly subjects only did ADLs, as in the real dataset
            plan += [("fall", c) for c in SIS_FALL_CODES]
        for kind, code in plan:
            for r in (1, 2):
                tr = make_trial(kind, rng)
                acc, gyr = _sis_native(tr.raw)
                mma = np.round(acc / (2 * 8 / 2 ** 14))
                cols = np.hstack([np.round(acc / SIS_ACC_G_PER_LSB), np.round(gyr / SIS_GYR_DPS_PER_LSB), mma]).astype(int)
                name = f"{code}_{subj}_R{r:02d}.txt"
                with open(os.path.join(root, subj, name), "w") as f:
                    for row in cols:
                        f.write(",".join(str(v) for v in row) + ";\n")
                meta.append(dict(file=name, subject=subj, code=code, kind=kind, t_onset=tr.t_onset,
                                 t_impact=tr.t_impact, raw=tr.raw))
    return meta


def write_mock_kfall(root, n_subjects=3, seed=1):
    """KFall layout: <root>/sensor_data/<SUBJECT>/<SUBJECT>T<TT>R<RR>.csv (100 Hz) and
    <root>/label_data/<SUBJECT>_label.xlsx with fall onset / impact frames."""
    import openpyxl
    rng = np.random.default_rng(seed)
    meta = []
    os.makedirs(os.path.join(root, "label_data"), exist_ok=True)
    subjects = [f"SA{i:02d}" for i in range(6, 6 + n_subjects)]
    header = "TimeStamp(s),FrameCounter,AccX,AccY,AccZ,GyrX,GyrY,GyrZ,EulerX,EulerY,EulerZ"
    for si, subj in enumerate(subjects):
        os.makedirs(os.path.join(root, "sensor_data", subj), exist_ok=True)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Task Code (Task ID)", "Description", "Trial ID", "Fall_onset_frame", "Fall_Impact_frame"])

        def write_csv(task, trial, tr):
            acc, gyr = _kf_native(tr.raw[::2])          # 200 Hz simulator -> 100 Hz
            fname = f"{subj}T{task:02d}R{trial:02d}.csv"
            with open(os.path.join(root, "sensor_data", subj, fname), "w") as f:
                f.write(header + "\n")
                for i in range(len(acc)):
                    f.write(f"{i / 100:.3f},{i + 1}," + ",".join(f"{v:.5f}" for v in acc[i]) + ","
                            + ",".join(f"{v:.4f}" for v in gyr[i]) + ",0,0,0\n")
            return fname

        for kind, task in KF_TASKS.items():
            for r in (1, 2):
                fname = write_csv(task, r, tr := make_trial(kind, rng))
                meta.append(dict(file=fname, subject=subj, task=task, kind=kind, t_onset=None, t_impact=None, raw=tr.raw))
        for task in KF_FALL_TASKS:
            for r in (1, 2):
                tr = make_trial("fall", rng)
                fname = write_csv(task, r, tr)
                on, imp = round(tr.t_onset * 100) + 1, round(tr.t_impact * 100) + 1
                ws.append([f"F{task - 19:02d} ({task})" if r == 1 else None, "Forward fall" if r == 1 else None, r, on, imp])
                meta.append(dict(file=fname, subject=subj, task=task, kind="fall", t_onset=tr.t_onset, t_impact=tr.t_impact, raw=tr.raw))
        if si == len(subjects) - 1:                     # a fall recording with no label row: loader must skip it
            tr = make_trial("fall", rng)
            fname = write_csv(KF_FALL_TASKS[-1], 3, tr)
            meta.append(dict(file=fname, subject=subj, task=KF_FALL_TASKS[-1], kind="unlabeled_fall", t_onset=None, t_impact=None, raw=tr.raw))
        wb.save(os.path.join(root, "label_data", f"{subj}_label.xlsx"))
    return meta
