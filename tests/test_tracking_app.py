"""Tracking tab back end (app/tracking.py + its API, issue #42) on a small
synthetic broadcast match: per-frame status (tracked / predicted / lost / off
camera), the problem list, lanes, frames, and fixes saved + undone."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MID = "synthetic_trk_1234abcd"
FPS = 25
N = 400
# image = 10 px per metre, pitch origin at the image's top-left: the 640x360
# image shows x 0-64 m, y 0-36 m of the pitch
H_IMG2PITCH = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]


def box(px, py):
    fx, fy = px * 10, py * 10
    return fx - 6, fy - 30, fx + 6, fy


@pytest.fixture
def root(tmp_path):
    mdir = tmp_path / ".tmp" / MID
    mdir.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "videos").mkdir()
    (mdir / "meta.json").write_text(json.dumps({"match_id": MID, "video": str(tmp_path / "videos" / "x.mp4"),
                                                "fps": FPS, "frame_count": N, "duration_s": N / FPS,
                                                "width": 640, "height": 360, "config": {}}))
    (tmp_path / "config" / f"{MID}_match.json").write_text(json.dumps({
        "team_a_name": "Man City", "team_b_name": "Tottenham", "colors": {"team_a": "#6CABDD", "team_b": "#132257"},
        "clock_sync": {"kickoff_frame": 0, "fps": FPS}}))
    (tmp_path / "config" / f"{MID}_calibration.json").write_text(json.dumps({"mode": "static",
                                                                           "homography": H_IMG2PITCH}))
    rows = []
    for f in range(N):
        # 1: always seen; 2: gap of 10 frames at 100-109 (predicted) and 200-299 (lost, in view);
        # 3: walks off the right edge at f>=150 (no rows) -> off camera; 50: unknown fragment
        for tid, (px, py) in {1: (10, 10), 2: (20, 20), 3: (60, 30)}.items():
            if tid == 2 and (100 <= f < 110 or 200 <= f < 300):
                continue
            if tid == 3 and f >= 150:
                continue
            x1, y1, x2, y2 = box(px, py)
            rows.append((f, f / FPS, tid, "player", x1, y1, x2, y2, (x1 + x2) / 2, y2, 0.9, px, py, float(tid),
                         "team_a"))
        if 210 <= f < 230:
            x1, y1, x2, y2 = box(30, 25)
            rows.append((f, f / FPS, 50, "player", x1, y1, x2, y2, (x1 + x2) / 2, y2, 0.6, 30, 25, 9.0, "unknown"))
        rows.append((f, f / FPS, -1, "ball", 100, 100, 104, 104, 102, 104, 0.5, 10.2, 10.4, np.nan, None))
    pd.DataFrame(rows, columns=["frame", "t_s", "track_id", "class", "x1", "y1", "x2", "y2", "foot_x", "foot_y",
                                "conf", "pitch_x", "pitch_y", "tracklet", "kit"]).to_parquet(mdir / "tracks.parquet",
                                                                                            index=False)
    pm = {"players": {"a2": {"team": "team_a", "number": 2, "name": "Walker", "track_ids": [1], "role": "OUT"},
                      "a3": {"team": "team_a", "number": 3, "name": "Dias", "track_ids": [2], "role": "OUT"},
                      "a31": {"team": "team_a", "number": 31, "name": "Ederson", "track_ids": [3], "role": "GK"}},
          "ignored_tracks": []}
    (mdir / "player_map.json").write_text(json.dumps(pm))
    (mdir / "contact_swaps.json").write_text(json.dumps(
        {"team_a": {"swaps": [{"a": 1, "b": 2, "frame": 50, "min_dist_m": 0.4}]}}))
    return tmp_path


def frame_of(mt, f):
    return mt.frames(f, f)["frames"][0]


def status(mt, key, f):
    return next(p[2] for p in frame_of(mt, f)["p"] if p[0] == key)


def test_status_and_problems(root):
    from app.tracking import MatchTracking

    mt = MatchTracking(root, MID)
    assert status(mt, "a2", 5) == "s"
    assert status(mt, "a3", 105) == "p"     # 10-frame gap: predicted
    assert status(mt, "a3", 250) == "l"     # 4 s gap, last seen in view: lost
    p = next(p for p in frame_of(mt, 105)["p"] if p[0] == "a3")
    assert p[3] is not None and p[7] == pytest.approx(20.0)  # interpolated box + spot
    lost = next(p for p in frame_of(mt, 250)["p"] if p[0] == "a3")
    assert lost[3] is None and lost[7] == pytest.approx(20.0)  # no box, last-known spot
    kinds = {(q["kind"], tuple(q["keys"])) for q in mt.summary()["problems"]}
    assert ("swap", ("a2", "a3")) in kinds
    assert ("lost", ("a3",)) in kinds
    fr = frame_of(mt, 215)
    assert fr["o"][0][0] == 50 and fr["b"][2] == pytest.approx(10.2)
    assert fr["v"] == [[0.0, 0.0], [64.0, 0.0], [64.0, 36.0], [0.0, 36.0]]
    lanes = mt.lanes(0, N - 1, 40)["lanes"]
    assert lanes["a2"] == "s" * 40 and "l" in lanes["a3"] and lanes["a31"].endswith("l")


def test_off_camera_vs_lost(root):
    """a31 was last seen at x=60 m: inside the 64 m-wide view, so lost. Move
    the camera model so that spot is outside: off camera."""
    from app.tracking import MatchTracking

    mt = MatchTracking(root, MID)
    assert status(mt, "a31", 250) == "l"
    calib = root / "config" / f"{MID}_calibration.json"
    calib.write_text(json.dumps({"mode": "static", "homography": [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1]]}))
    mt = MatchTracking(root, MID)  # now the view is x 0-32 m
    assert status(mt, "a31", 250) == "o"


def test_fixes_apply_and_resolve(root):
    from app.tracking import MatchTracking, TrackingError

    mt = MatchTracking(root, MID)
    swap = next(q for q in mt.summary()["problems"] if q["kind"] == "swap")
    mt.add_edit({"kind": "swap", "frame": 52, "a": 1, "b": 2})
    assert next(q for q in mt.summary()["problems"] if q["key"] == swap["key"])["fixed"] is not None
    assert next(p for p in frame_of(mt, 60)["p"] if p[0] == "a2")[8] == pytest.approx(20.0)  # swapped
    # after the swap, id 1 (Walker) carries the rows with the gap: his lost run is the problem now
    lost = next(q for q in mt.summary()["problems"] if q["kind"] == "lost" and q["keys"] == ["a2"])
    # hand box in the lost run; the pitch spot comes from the calibration
    e = mt.add_edit({"kind": "box", "frame": 250, "tid": 1, "x1": 194, "y1": 170, "x2": 206, "y2": 200,
                     "problem": lost["key"], "note": lost["text"]})
    assert e["pitch_x"] == pytest.approx(20.0) and e["pitch_y"] == pytest.approx(20.0)
    assert status(mt, "a2", 250) == "h"
    # the hand box filled the whole run (interpolated to both ends): it stays listed, as fixed
    gone = next(q for q in mt.summary()["problems"] if q["key"] == lost["key"])
    assert gone["fixed"] == e["id"] and gone["keys"] == ["a2"]
    # the unknown fragment is Ederson: assign
    mt.add_edit({"kind": "assign", "frame": 215, "tid": 50, "to": 3})
    assert any(p[0] == "a31" and p[2] == "s" for p in frame_of(mt, 220)["p"])
    assert not frame_of(mt, 220)["o"]
    with pytest.raises(TrackingError):
        mt.add_edit({"kind": "swap", "frame": 10, "a": 1, "b": 50})  # 50 isn't identified
    with pytest.raises(TrackingError):
        mt.add_edit({"kind": "box", "frame": 10_000, "tid": 1, "x1": 0, "y1": 0, "x2": 9, "y2": 9})
    saved = json.loads((root / "config" / f"{MID}_track_edits.json").read_text())
    assert [x["kind"] for x in saved["edits"]] == ["swap", "box", "assign"] and saved["base"]
    # every tool reading tracks sees the same fixes
    sys.path.insert(0, str(ROOT / "tools"))
    from _common import load_tracks
    t = load_tracks(MID, root / ".tmp", root / "config")
    assert 50 not in set(t.loc[t["frame"].between(215, 229), "track_id"])
    mt.remove_edit(e["id"])
    assert status(mt, "a2", 250) == "l"


def test_api(root):
    from fastapi.testclient import TestClient

    from app.jobs import JobRunner
    from app.server import create_app

    c = TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))
    s = c.get(f"/api/matches/{MID}/tracking").json()
    assert [p["key"] for p in s["players"]] == ["a31", "a2", "a3"]  # GK first, then by number
    assert s["teams"]["team_b"]["name"] == "Tottenham" and s["open_problems"] >= 2
    fr = c.get(f"/api/matches/{MID}/tracking/frames", params={"start": 0, "end": 2000}).json()
    assert fr["start"] == 0 and fr["end"] == N - 1 and len(fr["frames"]) == N
    assert c.get(f"/api/matches/{MID}/tracking/lanes", params={"start": 0, "end": 99, "bins": 10}).json()["bins"] == 10
    r = c.post(f"/api/matches/{MID}/tracking/edits", json={"edit": {"kind": "split", "frame": 30, "tid": 1}})
    assert r.status_code == 201, r.text
    eid = r.json()["id"]
    assert r.json()["tracklet"] == 1  # filled in by the server
    bad = c.post(f"/api/matches/{MID}/tracking/edits", json={"edit": {"kind": "nope"}})
    assert bad.status_code == 400
    m = c.get(f"/api/matches/{MID}").json()
    assert m["artifacts"]["track_edits"]["exists"]
    assert c.delete(f"/api/matches/{MID}/tracking/edits/{eid}").status_code == 200
    assert c.delete(f"/api/matches/{MID}/tracking/edits/{eid}").status_code == 400
    assert c.get("/api/matches/nope/tracking").status_code == 404
