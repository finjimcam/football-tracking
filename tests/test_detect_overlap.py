"""Issue #47: detector-side overlap handling (soft-NMS, duplicate counting,
duplicate suppression threshold) on synthetic boxes - no torch needed."""
import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
from detect_frames import box_iou, duplicate_pairs, soft_nms  # noqa: E402
from track_identities import suppress_duplicates  # noqa: E402


def test_box_iou_identical_disjoint_half():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]], float)
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 0.0, 1 / 3], atol=1e-6)


def test_soft_nms_keeps_hidden_player_with_decayed_score():
    # front player, a near-duplicate of it, and a half-hidden player behind
    boxes = np.array([[100, 100, 120, 160],   # front, conf 0.9
                      [101, 100, 121, 161],   # duplicate of front (IoU ~0.92)
                      [108, 98, 128, 156]],   # hidden player (IoU ~0.4 with front)
                     float)
    scores = np.array([0.9, 0.85, 0.6])
    classes = np.zeros(3, int)
    keep, sc = soft_nms(boxes, scores, classes, sigma=0.5, score_floor=0.05)
    assert keep[0] == 0 and sc[0] == pytest.approx(0.9)
    out = dict(zip(keep.tolist(), sc.tolist()))
    # duplicate decays hard (exp(-0.92^2/0.5) ~ 0.18), hidden player only mildly
    assert out[1] < 0.2
    assert 2 in out and out[2] > 0.4
    # with a 0.25 person floor the duplicate goes, the hidden player stays
    assert out[2] >= 0.25 > out[1]


def test_soft_nms_is_per_class_and_handles_empty():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    keep, sc = soft_nms(boxes, np.array([0.9, 0.8]), np.array([0, 32]))
    assert sorted(keep.tolist()) == [0, 1] and np.allclose(sorted(sc), [0.8, 0.9])
    keep, sc = soft_nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
    assert len(keep) == 0 and len(sc) == 0


def _dets(rows):
    return pd.DataFrame(rows, columns=["frame", "class", "x1", "y1", "x2", "y2", "conf"])


def test_duplicate_pairs_counts_only_heavy_same_frame_overlaps():
    det = _dets([(1, "player", 0, 0, 10, 20, 0.9), (1, "player", 0, 0, 10, 21, 0.8),   # dup
                 (1, "player", 6, 0, 16, 20, 0.7),                                     # duel, IoU < .6
                 (2, "player", 0, 0, 10, 20, 0.9), (2, "ball", 0, 0, 10, 20, 0.5)])    # ball ignored
    assert duplicate_pairs(det) == 1


def test_dup_iou_threshold_decides_whether_duel_partner_survives():
    # two players in a duel overlapping with IoU 0.7: dropped at the default
    # 0.6, kept at 0.8 (the threshold a looser detector NMS needs downstream)
    det = _dets([(1, "player", 0, 0, 10, 20, 0.9), (1, "player", 1.765, 0, 11.765, 20, 0.6)])
    iou = box_iou(det[["x1", "y1", "x2", "y2"]].to_numpy()[:1], det[["x1", "y1", "x2", "y2"]].to_numpy()[1:])[0, 0]
    assert iou == pytest.approx(0.7, abs=0.01)
    assert len(suppress_duplicates(det, 0.6)) == 1
    assert len(suppress_duplicates(det, 0.8)) == 2
