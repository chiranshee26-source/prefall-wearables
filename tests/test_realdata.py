import numpy as np
import pytest

pytest.importorskip("openpyxl")

from prefall.dsp import condition
from prefall.mockdata import write_mock_kfall, write_mock_sisfall
from prefall.realdata import (load_kfall, load_sisfall, parse_sisfall_file, split_subjects, _task_id)
from prefall.threshold import trace


def test_sisfall_units_axes_and_labels(tmp_path):
    meta = write_mock_sisfall(str(tmp_path), n_subjects=3)
    trials, amap = load_sisfall(str(tmp_path), verbose=False)
    assert len(trials) == len(meta)
    assert all(t.fs_raw == 200 for t in trials)
    # native frame put gravity on -y, forward on z: the estimator must undo that
    assert amap.acc_src == (2, 0, 1) and amap.acc_sign[2] == -1.0
    assert amap.gyr_src[1] == 2 and amap.gyr_sign[1] == 1.0
    by = {(t.subject["id"], t.subject["task"], t.subject["trial"]): t for t in trials}
    for m in meta:
        trial = int(m["file"].split("_R")[1][:2])
        t = by[(m["subject"], m["code"], trial)]
        rest = t.raw[:200, :3].mean(0)
        assert abs(np.linalg.norm(rest) - 1.0) < 0.15          # counts converted to g
        assert rest[2] > 0.8                                    # upright = +1 g on az
        if m["kind"] == "fall":
            assert t.kind == "fall"
            assert abs(t.t_impact - m["t_impact"]) < 0.1        # impact = acceleration peak
            th, _ = trace(condition(t.raw, t.fs_raw))
            assert th.max() > 60                                # tracker sees the fall in the recovered frame
        else:
            assert t.kind == m["code"] and t.t_impact is None
    groups = {t.subject["group"] for t in trials}
    assert groups == {"adult", "elderly"}
    assert not any(t.kind == "fall" for t in trials if t.subject["group"] == "elderly")


def test_sisfall_parser_handles_semicolons_and_bad_length(tmp_path):
    p = tmp_path / "F01_SA01_R01.txt"
    p.write_text("1,2,3,4,5,6,7,8,9;\n 10, 11,12,13,14,15,16,17,18 ;\n19,20\n")
    with pytest.warns(UserWarning):
        a = parse_sisfall_file(str(p))
    assert a.shape == (2, 9) and a[1, 0] == 10


def test_kfall_labels_rate_and_unlabeled_falls(tmp_path):
    meta = write_mock_kfall(str(tmp_path), n_subjects=3)
    trials, amap = load_kfall(str(tmp_path), verbose=False)
    assert all(abs(t.fs_raw - 100) < 1 for t in trials)         # rate read from the timestamps
    assert amap.acc_src[2] == 0 and amap.acc_sign[2] == -1.0     # gravity on native -x
    n_unlabeled = sum(m["kind"] == "unlabeled_fall" for m in meta)
    assert n_unlabeled == 1
    assert len(trials) == len(meta) - n_unlabeled               # the unlabeled fall is skipped, not called an ADL
    falls = [t for t in trials if t.kind == "fall"]
    truth = {(m["subject"], m["file"]): m for m in meta if m["kind"] == "fall"}
    assert len(falls) == len(truth)
    for t in falls:
        key = (t.subject["id"], f"{t.subject['id']}{t.subject['task']}R{t.subject['trial']:02d}.csv")
        m = truth[key]
        assert abs(t.t_onset - m["t_onset"]) < 0.03 and abs(t.t_impact - m["t_impact"]) < 0.03
        assert t.t_impact > t.t_onset
        th, _ = trace(condition(t.raw, t.fs_raw))
        assert th.max() > 60
    assert all(t.kind.startswith("T") for t in trials if t.kind != "fall")


def test_task_id_parsing():
    assert _task_id("F01 (20)") == 20 and _task_id(21) == 21 and _task_id(" 7 ") == 7
    assert _task_id(None) is None and _task_id("F01") is None


def test_subject_split_has_no_overlap(tmp_path):
    write_mock_sisfall(str(tmp_path), n_subjects=6)
    trials, _ = load_sisfall(str(tmp_path), verbose=False)
    train, test = split_subjects(trials, test_frac=0.4, seed=1)
    assert {t.subject["id"] for t in train}.isdisjoint({t.subject["id"] for t in test})
    assert len(train) + len(test) == len(trials) and any(t.kind == "fall" for t in test)


def test_condition_handles_non_integer_rates():
    x = np.random.default_rng(0).normal(size=(2380, 6))          # 10 s at 238 Hz
    y = condition(x, 238)
    assert abs(len(y) - 500) <= 2 and y.shape[1] == 6
