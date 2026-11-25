"""Tracking tab Auto-fix (app/tracking.py MatchTracking.autofix): while a
player is missing, the free boxes in his colours inside the rectangle from his
last box to his next one are his (unless a missing team-mate's rectangle holds
them too), and the decoys it must leave alone. Plus the two edit kinds it saves
(tools/track_edits.py autoref / autofill)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import track_edits as te  # noqa: E402

MID = "synthetic_fix_1234abcd"
FPS = 25
N = 900
H_IMG2PITCH = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]   # 10 px per metre: the image shows x 0-64, y 0-36


def box(px, py):
    fx, fy = px * 10, py * 10
    return fx - 6, fy - 30, fx + 6, fy


def lerp(f, f0, p0, f1, p1):
    w = (f - f0) / (f1 - f0)
    return p0[0] + w * (p1[0] - p0[0]), p0[1] + w * (p1[1] - p0[1])


def build(tmp_path, extra=()):
    """a2 (id 1) stands at (10, 10). a3 (id 2) walks (20, 20) -> (30, 20) and
    is lost on frames 200-299: unlabelled id 50 (team_a kit) walks his line,
    id 60 (team_b kit) walks it too, id 70 (team_a) runs 25 m away. a4 (id 3)
    walks (40, 30) -> (50, 30) and is lost on frames 400-699 (12
    s): unlabelled id 80 (team_a) walks where he went. b5 (id 4) is
    always seen at (55, 5)."""
    mdir = tmp_path / ".tmp" / MID
    mdir.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "videos").mkdir()
    (mdir / "meta.json").write_text(json.dumps({"match_id": MID, "video": str(tmp_path / "videos" / "x.mp4"),
                                                "fps": FPS, "frame_count": N, "duration_s": N / FPS,
                                                "width": 640, "height": 360, "config": {}}))
    (tmp_path / "config" / f"{MID}_match.json").write_text(json.dumps({
        "team_a_name": "Korea", "team_b_name": "Czechia", "clock_sync": {"kickoff_frame": 0, "fps": FPS}}))
    (tmp_path / "config" / f"{MID}_calibration.json").write_text(json.dumps({"mode": "static",
                                                                           "homography": H_IMG2PITCH}))
    rows = []

    def add(f, tid, p, kit, conf=0.9):
        x1, y1, x2, y2 = box(*p)
        rows.append((f, f / FPS, tid, "player", x1, y1, x2, y2, (x1 + x2) / 2, y2, conf, p[0], p[1],
                     float(tid), kit))

    for f in range(N):
        add(f, 1, (10, 10), "team_a")
        add(f, 4, (55, 5), "team_b")
        a3 = lerp(f, 0, (20, 20), N - 1, (30, 20))
        if not 200 <= f < 300:
            add(f, 2, a3, "team_a")
        elif 205 <= f < 295:
            add(f, 50, a3, "team_a")                                   # him
            add(f, 60, (a3[0], a3[1] + 1.5), "team_b")                 # wrong kit
            add(f, 70, (a3[0] + 25, a3[1] - 10), "team_a")             # nowhere near
        a4 = lerp(f, 0, (40, 30), N - 1, (50, 30))
        if not 400 <= f < 700:
            add(f, 3, a4, "team_a")
        elif 420 <= f < 680:
            add(f, 80, (a4[0], a4[1] - 1), "team_a")                   # him, by elimination
        for e in extra:
            e(f, add)
        rows.append((f, f / FPS, -1, "ball", 100, 100, 104, 104, 102, 104, 0.5, 10.2, 10.4, np.nan, None))
    pd.DataFrame(rows, columns=["frame", "t_s", "track_id", "class", "x1", "y1", "x2", "y2", "foot_x", "foot_y",
                                "conf", "pitch_x", "pitch_y", "tracklet", "kit"]).to_parquet(mdir / "tracks.parquet",
                                                                                            index=False)
    pm = {"players": {"a2": {"team": "team_a", "number": 2, "name": "A", "track_ids": [1], "role": "OUT"},
                      "a3": {"team": "team_a", "number": 3, "name": "B", "track_ids": [2], "role": "OUT"},
                      "a4": {"team": "team_a", "number": 4, "name": "C", "track_ids": [3], "role": "OUT"},
                      "b5": {"team": "team_b", "number": 5, "name": "D", "track_ids": [4], "role": "OUT"}},
          "ignored_tracks": []}
    (mdir / "player_map.json").write_text(json.dumps(pm))
    return tmp_path


@pytest.fixture
def root(tmp_path):
    return build(tmp_path)


def test_autofix_finds_the_boxes_in_his_rectangle(root):
    from app.tracking import MatchTracking

    mt = MatchTracking(root, MID)
    lost_before = mt._lost_seconds()
    r = mt.autofix()
    assert r["box_pieces"] >= 2 and r["ref_learned"] is False
    assert r["lost_s_after"] < lost_before
    ids = mt.df["track_id"].to_numpy()
    fr = mt.df["frame"].to_numpy()
    in_gap = (fr >= 205) & (fr < 295)
    # 50 became a3 (id 2); the wrong-kit 60 and the far-away 70 did not
    assert not np.isin(ids, [50]).any()
    assert ((ids == 60) & in_gap).sum() == 90 and ((ids == 70) & in_gap).sum() == 90
    assert ((ids == 2) & in_gap).sum() == 90
    # 80 became a4 (id 3): the only team_a player missing, and it is in his rectangle
    assert not (ids == 80).any() and ((ids == 3) & (fr >= 420) & (fr < 680)).sum() == 260
    kinds = [e["kind"] for e in te.load_edits(mt.edits_path)["edits"]]
    assert kinds == ["autofill"]


def test_autofix_again_replaces_and_undo_removes(root):
    from app.tracking import MatchTracking, TrackingError

    mt = MatchTracking(root, MID)
    mt.add_edit({"kind": "split", "frame": 30, "tid": 1})
    first = mt.autofix()
    second = mt.autofix()
    assert second["fill_pieces"] == first["fill_pieces"]
    kinds = [e["kind"] for e in te.load_edits(mt.edits_path)["edits"]]
    assert kinds == ["split", "autofill"]      # not stacked
    assert mt.undo_autofix() == {"removed": 1}
    assert [e["kind"] for e in te.load_edits(mt.edits_path)["edits"]] == ["split"]
    assert (mt.df["track_id"] == 50).sum() == 90
    with pytest.raises(TrackingError):
        mt.undo_autofix()


def test_a_box_in_two_missing_rectangles_is_left(tmp_path):
    """A second team_a player (id 5) is missing for the whole of id 80's run
    and his rectangle covers it too: 80 could be either of them, so it is
    left for you."""
    from app.tracking import MatchTracking

    def a6(f, add):
        if not 380 <= f < 720:
            add(f, 5, (46, 29), "team_a")

    build(tmp_path, extra=[a6])
    pm_path = tmp_path / ".tmp" / MID / "player_map.json"
    pm = json.loads(pm_path.read_text())
    pm["players"]["a6"] = {"team": "team_a", "number": 6, "name": "E", "track_ids": [5], "role": "OUT"}
    pm_path.write_text(json.dumps(pm))
    mt = MatchTracking(tmp_path, MID)
    mt.autofix()
    assert (mt.df["track_id"] == 80).sum() == 260


def test_faint_boxes_are_not_searched(tmp_path):
    """Only one player missing, but the box is a faint detection: leave it."""
    from app.tracking import MatchTracking

    build(tmp_path)
    p = tmp_path / ".tmp" / MID / "tracks.parquet"
    df = pd.read_parquet(p)
    df.loc[df["track_id"] == 80, "conf"] = 0.2
    df.to_parquet(p, index=False)
    mt = MatchTracking(tmp_path, MID)
    mt.autofix()
    assert (mt.df["track_id"] == 80).sum() == 260


def test_boxes_off_the_pitch_or_cut_off_are_not_searched(tmp_path):
    """id 80 moved behind the touchline (boards / bench): leave it."""
    from app.tracking import MatchTracking

    build(tmp_path)
    p = tmp_path / ".tmp" / MID / "tracks.parquet"
    df = pd.read_parquet(p)
    df.loc[df["track_id"] == 80, "pitch_y"] = -4.0
    df.to_parquet(p, index=False)
    mt = MatchTracking(tmp_path, MID)
    mt.autofix()
    assert (mt.df["track_id"] == 80).sum() == 260


def test_autofix_api(root):
    from fastapi.testclient import TestClient

    from app.jobs import JobRunner
    from app.server import create_app

    c = TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))
    r = c.post(f"/api/matches/{MID}/tracking/autofix")
    assert r.status_code == 200, r.text
    assert r.json()["box_pieces"] >= 2
    s = c.get(f"/api/matches/{MID}/tracking").json()
    e = next(e for e in s["edits"] if e["kind"] == "autofill")
    assert "items" not in e and e["n_items"] >= 2 and "rectangle" in e["note"]
    assert c.delete(f"/api/matches/{MID}/tracking/autofix").status_code == 200
    assert c.delete(f"/api/matches/{MID}/tracking/autofix").status_code == 400


# ---- the edit kinds ----------------------------------------------------------
def tracks():
    rows = []
    for f in range(20):
        for tid in (1, 2, 50):
            rows.append((f, tid, "player", 0, 0, 10, 10, float(f), 0.0, "team_a"))
    return pd.DataFrame(rows, columns=["frame", "track_id", "class", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y",
                                       "kit"])


def test_autofill_and_autoref_apply():
    df = tracks()
    data = te.empty("m")
    ref = te.add_edit(data, {"kind": "autoref", "frame": 0, "items": [[2, 5, 9]], "batch": "x"})
    te.add_edit(data, {"kind": "autofill", "frame": 0, "items": [[50, 3, 6, 1], [50, 10, 12, 2]]})
    out = te.apply_edits(df, data["edits"], FPS)
    at = lambda f, t: set(out[(out["frame"] == f)]["track_id"]) & {t}  # noqa: E731
    new_ref = te.AUTO_ID_BASE + 1000 * ref["id"]
    assert at(7, new_ref) and not at(7, 2) and at(4, 2)
    assert (out.loc[out["track_id"] == new_ref, "kit"] == "official").all()
    assert ((out["track_id"] == 1) & out["frame"].between(3, 6)).sum() == 8   # his own + the filled box
    assert ((out["track_id"] == 50) & out["frame"].between(3, 6)).sum() == 0
    assert ((out["track_id"] == 50) & (out["frame"] == 15)).sum() == 1      # outside the items: untouched


def test_auto_edits_validate_and_retire_on_rerun():
    with pytest.raises(te.EditError):
        te.clean_edit({"kind": "autofill", "frame": 0, "items": []})
    with pytest.raises(te.EditError):
        te.clean_edit({"kind": "autofill", "frame": 0, "items": [[50, 6, 3, 1]]})    # f1 < f0
    with pytest.raises(te.EditError):
        te.clean_edit({"kind": "autoref", "frame": 0, "items": [[2, 1, 2, 3]]})       # 3 numbers each
    old = tracks()
    data = te.empty("m")
    te.add_edit(data, {"kind": "autofill", "frame": 0, "items": [[50, 3, 6, 1]]}, base=te.fingerprint(old))
    new = old.copy()
    new["track_id"] += 1
    data, rep = te.rebase(data, new, {"players": {}})
    assert data["edits"] == [] and rep["retired"] == 1
