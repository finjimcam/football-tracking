"""Hand fixes to the tracking (tools/track_edits.py, issue #42) on small
synthetic tracks: a swap, an assignment of an unidentified fragment, a hand
box with interpolation, split / delete / not-a-player, and the loader that
applies them for every downstream tool."""
import json

import numpy as np
import pandas as pd
import pytest

import track_edits as te
from _common import load_tracks

FPS = 25


def make_tracks(n=100):
    """Ids 1 and 2 (team_a) walk along y=10 and y=20; id 50 is an unknown
    fragment on frames 40-59 (tracklet 9); 1 is missing on frames 40-59."""
    rows = []
    for f in range(n):
        for tid, y, tl in ((1, 10.0, 1 if f < 50 else 3), (2, 20.0, 2)):
            if tid == 1 and 40 <= f < 60:
                continue
            x = 10 + f * 0.1
            rows.append((f, f / FPS, tid, "player", x * 10 - 5, y * 10 - 20, x * 10 + 5, y * 10, x * 10, y * 10,
                         0.9, x, y, tl, "team_a"))
        if 40 <= f < 60:
            x = 10 + f * 0.1
            rows.append((f, f / FPS, 50, "player", x * 10 - 5, 80, x * 10 + 5, 100, x * 10, 100, 0.6, x, 10.0, 9,
                         "unknown"))
        rows.append((f, f / FPS, -1, "ball", 0, 0, 2, 2, 1, 2, 0.5, 30.0, 30.0, np.nan, None))
    return pd.DataFrame(rows, columns=["frame", "t_s", "track_id", "class", "x1", "y1", "x2", "y2", "foot_x",
                                       "foot_y", "conf", "pitch_x", "pitch_y", "tracklet", "kit"])


def ids_at(df, f):
    g = df[(df["frame"] == f) & (df["class"] == "player")]
    return dict(zip(g["track_id"], g["pitch_y"]))


def test_swap_from_frame_on():
    df = te.apply_edits(make_tracks(), [{"kind": "swap", "frame": 70, "a": 1, "b": 2}], FPS)
    assert ids_at(df, 69) == {1: 10.0, 2: 20.0}
    assert ids_at(df, 70) == {2: 10.0, 1: 20.0}
    assert ids_at(df, 99) == {2: 10.0, 1: 20.0}
    # bounded swap
    df = te.apply_edits(make_tracks(), [{"kind": "swap", "frame": 70, "a": 1, "b": 2, "to": 80}], FPS)
    assert ids_at(df, 80) == {2: 10.0, 1: 20.0}
    assert ids_at(df, 81) == {1: 10.0, 2: 20.0}


def test_assign_fragment_trades_with_the_wrong_owner():
    tracks = make_tracks()
    # the unknown fragment 50 is really player 1 (who is missing there)
    df = te.apply_edits(tracks, [{"kind": "assign", "frame": 45, "tid": 50, "to": 1}], FPS)
    assert 50 not in set(df.loc[df["frame"].between(45, 59), "track_id"])
    assert ids_at(df, 50)[1] == 10.0
    # from the click on only: frames 40-44 stay the fragment
    assert 50 in ids_at(df, 44)
    # assigning a fragment to 2 while 2 is tracked: they trade on those frames
    df = te.apply_edits(tracks, [{"kind": "assign", "frame": 40, "tid": 50, "to": 2}], FPS)
    at = ids_at(df, 45)
    assert at[2] == 10.0 and at[50] == 20.0
    assert ids_at(df, 70)[2] == 20.0  # outside the fragment's frames nothing moves


def test_hand_box_interpolates_to_neighbours():
    box = {"kind": "box", "frame": 50, "tid": 1, "x1": 145, "y1": 80, "x2": 155, "y2": 100,
           "pitch_x": 15.0, "pitch_y": 10.0}
    df = te.apply_edits(make_tracks(), [box], FPS)
    one = df[(df["track_id"] == 1)].set_index("frame")
    assert set(range(40, 60)) <= set(one.index)  # the gap 40-59 is filled
    assert one.loc[50, "source"] == "hand"
    assert one.loc[45, "source"] == "hand_fill" and one.loc[55, "source"] == "hand_fill"
    assert one.loc[39, "source"] == "auto"
    # linear between frame 39 (x=13.9) and the hand box (x=15.0)
    assert one.loc[45, "pitch_x"] == pytest.approx(13.9 + (15.0 - 13.9) * 6 / 11)
    assert one.loc[50, "kit"] == "team_a"


def test_lone_hand_box_is_held_briefly():
    tracks = make_tracks(400)
    tracks = tracks[~((tracks["track_id"] == 1) & (tracks["frame"] >= 30))]  # 1 gone after f29
    f0 = 30 + int(te.INTERP_S * FPS) + 100  # too far from f29 to interpolate
    box = {"kind": "box", "frame": f0, "tid": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 20, "pitch_x": 1, "pitch_y": 2}
    df = te.apply_edits(tracks, [box], FPS)
    frames = sorted(df.loc[(df["track_id"] == 1) & (df["frame"] > 29), "frame"])
    hold = int(te.HOLD_S * FPS)
    assert frames == list(range(f0 - hold, f0 + hold + 1))


def test_split_delete_not_player():
    tracks = make_tracks()
    df = te.apply_edits(tracks, [{"id": 4, "kind": "split", "frame": 20, "tid": 2}], FPS)
    assert 2 in ids_at(df, 19) and 2 not in ids_at(df, 20)
    assert te.NEW_ID_BASE + 4 in ids_at(df, 99)
    df = te.apply_edits(tracks, [{"kind": "delete", "frame": 20, "tid": 2, "to": 22}], FPS)
    assert [2 in ids_at(df, f) for f in (19, 20, 22, 23)] == [True, False, False, True]
    # not a player: the whole tracklet the box is on (1's first tracklet ends at f49)
    df = te.apply_edits(tracks, [{"id": 7, "kind": "not_player", "frame": 10, "tid": 1}], FPS)
    assert 1 not in ids_at(df, 0) and 1 in ids_at(df, 60)
    gone = df[df["track_id"] == te.NEW_ID_BASE + 7]
    assert len(gone) == 40 and set(gone["kit"]) == {"official"}


def test_edits_apply_in_order_and_ball_untouched():
    tracks = make_tracks()
    edits = [{"kind": "swap", "frame": 70, "a": 1, "b": 2}, {"kind": "swap", "frame": 80, "a": 1, "b": 2},
             {"kind": "ok", "problem": "swap:70:1:2"}]
    df = te.apply_edits(tracks, edits, FPS)
    assert ids_at(df, 75) == {2: 10.0, 1: 20.0}
    assert ids_at(df, 85) == {1: 10.0, 2: 20.0}
    assert (df["class"] == "ball").sum() == (tracks["class"] == "ball").sum()
    assert len(df) == len(tracks)


def test_clean_edit_rejects_bad_input():
    for bad in ({"kind": "nope"}, {"kind": "swap", "frame": 1, "a": 1, "b": 1},
                {"kind": "box", "frame": 1, "tid": 1, "x1": 0, "y1": 0, "x2": 1, "y2": 1, "pitch_x": 0, "pitch_y": 0},
                {"kind": "assign", "frame": "x", "tid": 1, "to": 2}, {"kind": "ok"}):
        with pytest.raises(te.EditError):
            te.clean_edit(bad)
    e = te.clean_edit({"kind": "swap", "frame": "5", "a": 1, "b": 2, "junk": 1})
    assert e == {"kind": "swap", "frame": 5, "a": 1, "b": 2}


def test_loader_applies_saved_fixes_and_checks_fingerprint(tmp_path):
    mid = "m_1234abcd"
    mdir = tmp_path / ".tmp" / mid
    mdir.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (mdir / "meta.json").write_text(json.dumps({"fps": FPS}))
    tracks = make_tracks()
    tracks.to_parquet(mdir / "tracks.parquet", index=False)
    path = te.edits_path(mid, tmp_path / "config")
    data = te.empty(mid)
    te.add_edit(data, {"kind": "swap", "frame": 70, "a": 1, "b": 2}, base=te.fingerprint(tracks))
    te.save_edits(path, data)
    df = load_tracks(mid, tmp_path / ".tmp", tmp_path / "config")
    assert ids_at(df, 75) == {2: 10.0, 1: 20.0}
    raw = load_tracks(mid, tmp_path / ".tmp", tmp_path / "config", edits=False)
    assert ids_at(raw, 75) == {1: 10.0, 2: 20.0}
    cols = load_tracks(mid, tmp_path / ".tmp", tmp_path / "config", columns=["frame", "track_id"])
    assert list(cols.columns) == ["frame", "track_id"]
    # tracking re-run (different rows): the fixes are not applied
    tracks[tracks["frame"] < 90].to_parquet(mdir / "tracks.parquet", index=False)
    df = load_tracks(mid, tmp_path / ".tmp", tmp_path / "config")
    assert ids_at(df, 75) == {1: 10.0, 2: 20.0}
    # undo
    te.remove_edit(data, data["edits"][0]["id"])
    assert data["edits"] == [] and data["base"] is None
