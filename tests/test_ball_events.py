"""Issue #7: ball clutter filter (track_ball.py) and event detection (detect_events.py)."""
import numpy as np
import pandas as pd

from track_ball import static_clutter

FPS = 25


def _tracklet(btl, f0, n, x, y, vx=0.0, vy=0.0, conf=0.1):
    f = np.arange(f0, f0 + n)
    return pd.DataFrame({"frame": f, "pitch_x": x + vx * (f - f0) / FPS, "pitch_y": y + vy * (f - f0) / FPS,
                         "conf": conf, "btl": btl})


def test_static_clutter_drops_recurring_still_spots_only():
    parts, btl = [], 0
    # the painted penalty spot: still for 10 s in each of 5 different minutes
    for k in range(5):
        parts.append(_tracklet(btl, k * 60 * FPS + 100, 10 * FPS, 94.2, 34.1))
        btl += 1
    penalty_ids = set(range(btl))
    # a corner: the real ball rests 8 s by the flag, once
    parts.append(_tracklet(btl, 7 * 60 * FPS, 8 * FPS, 104.8, 0.4, conf=0.5)); btl += 1
    # the real ball rolling straight through the penalty spot
    parts.append(_tracklet(btl, 2 * 60 * FPS + 50, 2 * FPS, 85.0, 34.0, vx=10.0, conf=0.5)); btl += 1
    # a long still spare ball on the touchline, re-found every 2 minutes
    for k in range(4):
        parts.append(_tracklet(btl, k * 120 * FPS + 400, 12 * FPS, 80.0, -0.5))
        btl += 1
    cand = pd.concat(parts, ignore_index=True)
    drop, spots = static_clutter(cand, FPS)
    dropped = set(cand.loc[drop, "btl"])
    assert dropped == penalty_ids | set(range(btl - 4, btl))
    assert sorted(map(tuple, spots[["x", "y"]].round().to_numpy().tolist())) == [(80.0, 0.0), (94.0, 34.0)]
    # switched off
    drop0, spots0 = static_clutter(cand, FPS, min_total_s=0)
    assert not drop0.any() and spots0.empty


def test_static_clutter_needs_several_minutes():
    # 40 s still in one minute only (a long stoppage with the ball on the spot) is not clutter
    cand = _tracklet(0, 0, 40 * FPS, 94.0, 34.0)
    drop, spots = static_clutter(cand, FPS)
    assert not drop.any() and spots.empty


# ---------- detect_events.py ----------
from detect_events import control_spells, detect, score_shots  # noqa: E402

L, W = 105.0, 68.0


def _ball(points):
    """points: [(frame, x, y)] -> frame-indexed ball DataFrame, linearly filled per frame."""
    rows = []
    for (f0, x0, y0), (f1, x1, y1) in zip(points[:-1], points[1:]):
        for f in range(f0, f1):
            w = (f - f0) / (f1 - f0)
            rows.append((f, x0 + w * (x1 - x0), y0 + w * (y1 - y0)))
    rows.append(points[-1])
    return pd.DataFrame(rows, columns=["frame", "pitch_x", "pitch_y"]).set_index("frame")


def test_control_spells_bridge_short_gaps_and_need_a_team():
    tl = pd.DataFrame({"frame": range(20),
                       "holder": [5] * 4 + [-1] * 3 + [5] * 3 + [7] * 5 + [99] * 5,
                       "candidate": ["team_a"] * 15 + ["none"] * 5})
    sp = control_spells(tl, {5: "team_a", 7: "team_a"}, bridge=6, min_frames=1)
    assert sp == [(0, 9, 5, "team_a"), (10, 14, 7, "team_a")]  # holder 99 has no team -> dropped


def test_control_spells_drop_flicker_and_rejoin():
    # 5 holds, a 2-frame flicker to marker 104, 5 again: the flicker goes, 5's spells join
    tl = pd.DataFrame({"frame": range(30), "holder": [5] * 12 + [104] * 2 + [-1] * 4 + [5] * 12,
                       "candidate": ["team_a"] * 30})
    sp = control_spells(tl, {5: "team_a", 104: "team_b"}, bridge=2, min_frames=5)
    assert sp == [(0, 29, 5, "team_a")]


def test_pass_turnover_tackle_and_shot():
    goals = {"team_a": L, "team_b": 0.0}
    spells = [(0, 20, 2, "team_a"),       # A2 passes 15 m to A3
              (35, 60, 3, "team_a"),      # A3 loses it 10 m away to B4
              (75, 90, 104, "team_b"),    # B4 is tackled by A5 on the spot
              (92, 110, 5, "team_a"),     # A5 shoots from 20 m: ball reaches the goal line fast
              (200, 220, 101, "team_b")]
    ball = _ball([(0, 50, 30), (20, 52, 30), (35, 67, 30), (60, 70, 30), (75, 80, 30), (90, 82, 30),
                  (92, 83, 30), (110, 85, 33), (125, 104, 35), (200, 104, 35), (220, 90, 34)])
    ev = detect(spells, ball, goals, FPS, L, W)
    got = [(r.type, r.team, r.player, None if pd.isna(r.to_player) else int(r.to_player)) for r in ev.itertuples()]
    assert got == [("pass", "team_a", 2, 3), ("pass_lost", "team_a", 3, 104), ("tackle", "team_a", 5, 104),
                   ("shot", "team_a", 5, None)]
    shot = ev[ev["type"] == "shot"].iloc[0]
    assert shot["end_x"] > 100 and shot["speed_ms"] > 20


def test_no_shot_when_ball_goes_sideways():
    goals = {"team_a": L, "team_b": 0.0}
    spells = [(0, 20, 5, "team_a"), (60, 80, 6, "team_a")]
    ball = _ball([(0, 85, 30), (20, 85, 30), (60, 85, 60), (80, 86, 60)])  # square ball, 30 m across
    ev = detect(spells, ball, goals, FPS, L, W)
    assert list(ev["type"]) == ["pass"]


def test_score_shots_matches_each_detection_once():
    ev = pd.DataFrame({"frame": [1000, 1010, 5000], "type": ["shot"] * 3, "team": ["team_a", "team_a", "team_b"]})
    ref = [{"s": 6, "type": "shot", "team": "team_a"}, {"s": 7, "type": "goal", "team": "team_a"},
           {"s": 150, "type": "shot", "team": "team_a"}, {"s": 300, "type": "foul_won", "team": "team_b"}]
    sc = score_shots(ev, ref, kickoff=850, fps=FPS, window_s=5)
    assert (sc["opta_shots"], sc["found"], sc["unmatched_detections"]) == (3, 2, 1)
    assert sc["team_ok"] == 2


def test_detect_events_stage_in_the_app():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import catalog
    st = catalog.get_stage("b_events")
    args = catalog.build_args(st, {"instances": True}, match_id="m1")
    assert args[0] == "--match-id" and "--instances" in args
    assert "--instances" not in catalog.build_args(st, {}, match_id="m1")


def test_write_instances_replaces_only_auto_ones(tmp_path):
    import json
    from detect_events import write_instances
    p = tmp_path / "m_instances.json"
    p.write_text(json.dumps({"template": "Default", "instances": [
        {"id": "hand", "code": "Press", "t": 5.0, "start": 1.0, "end": 9.0, "labels": {}, "created": 1},
        {"id": "old", "code": "Pass", "t": 6.0, "start": 3.0, "end": 10.0, "labels": {}, "created": 1, "auto": True}]}))
    ev = pd.DataFrame({"frame": [250, 500], "type": ["pass", "shot"], "team": ["team_a", "team_b"], "player": [2, 104]})
    n, kept = write_instances(ev, p, 25, 100.0, {2: "2 Walker"}, {"team_a": "Man City", "team_b": "Tottenham"})
    d = json.loads(p.read_text())
    assert (n, kept) == (2, 1) and d["template"] == "Default"
    got = [(i["code"], i["t"], i.get("auto", False), i["labels"].get("Player")) for i in d["instances"]]
    assert got == [("Press", 5.0, False, None), ("Pass", 10.0, True, "2 Walker"), ("Shot", 20.0, True, None)]
    assert [i["id"] for i in d["instances"]].count("old") == 0 and any(i["id"] == "hand" for i in d["instances"])
    shot = next(i for i in d["instances"] if i["code"] == "Shot")
    assert (shot["start"], shot["end"], shot["labels"]["Team"]) == (12.0, 25.0, "Tottenham")


def test_class_ids_from_model_names():
    import pytest as _pt
    from detect_frames import class_ids
    assert class_ids({0: "person", 1: "bicycle", 32: "sports ball"}) == (0, 32)  # pretrained COCO
    assert class_ids({0: "person", 1: "ball"}) == (0, 1)                        # fine-tuned (#7)
    with _pt.raises(SystemExit):
        class_ids({0: "person"})


def test_ball_labels_add_short_gaps_sized_like_detections():
    from make_ball_dataset import ball_labels
    ball = pd.DataFrame({"frame": [10, 11, 12, 13, 14, 20, 40],
                         "interp": [False, True, True, False, True, False, False],
                         "x1": [100, 0, 0, 110, 0, 200, 300], "y1": [50, 0, 0, 52, 0, 60, 70],
                         "x2": [106, 0, 0, 117, 0, 207, 306], "y2": [56, 0, 0, 59, 0, 67, 76],
                         "foot_x": [103, 105, 107, 113.5, 120, 203.5, 303], "foot_y": [56, 57, 58, 59, 60, 67, 76]})
    lab = ball_labels(ball, interp_gap=4)
    assert set(lab) == {10, 11, 12, 13, 20, 40}  # 14 is in a 7-frame gap (13 -> 20): too long to trust
    assert lab[10] == (103.0, 53.0, 6, 6)
    cx, cy, w, h = lab[11]
    assert (cx, w, h) == (105, 7.0, 7.0) and cy == 57 - 3.5  # sized like the next detection (frame 13)
