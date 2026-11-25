import numpy as np
import pandas as pd
import pytest

import track_periods as P

FPS = 30.0
CFG = {
    "clock_sync": {"kickoff_frame": 20},
    "periods": {
        "second_half": {"start_frame": 87110, "end_frame": 179867, "kickoff_frame": 86760, "clock_start_s": 2700},
        "first_half": {"start_frame": 20, "end_frame": 86746},
    },
}


def test_periods_in_time_order_and_fallback_to_window():
    assert [p[0] for p in P.periods_of(CFG)] == ["first_half", "second_half"]
    assert P.periods_of({"analysis_window": {"start_frame": 5, "end_frame": 9}}) == [("match", 5, 9, {})]


@pytest.mark.parametrize("m, want", [(62, 62.0), ("62", 62.0), ("45+2", 47.0), ("90+3'", 93.0)])
def test_parse_minute(m, want):
    assert P.parse_minute(m) == want


def test_minute_to_frame_uses_the_period_kickoff():
    # 10th minute -> 9:30 after the first kick-off
    assert P.minute_to_frame(10, CFG, FPS) == 20 + int(9.5 * 60 * FPS)
    # 62nd minute -> 61:30 = 16.5 min after the second-half kick-off at 45:00
    assert P.minute_to_frame("62", CFG, FPS) == 86760 + int(16.5 * 60 * FPS)


def test_substitutions_sorted_and_explicit_frame_wins():
    ref = {"substitutions": [
        {"team": "team_a", "minute": "84", "off": {"number": 6}, "on": {"number": 24}},
        {"team": "team_b", "minute": "64", "frame": 120000, "off": {"number": 15}, "on": {"number": 9}},
    ]}
    subs = P.substitutions(ref, CFG, FPS)
    assert [s["frame"] for s in subs] == [120000, P.minute_to_frame(84, CFG, FPS)]


def test_npz_slice_and_merge_replace_frames():
    base = {"frames": np.array([1, 2, 3, 4]), "H": np.arange(4.0)[:, None] * np.ones((4, 9)), "K": np.eye(3)}
    part = {"frames": np.array([3, 4, 5]), "H": np.full((3, 9), 9.0)}
    merged = P.npz_merge(base, part)
    assert merged["frames"].tolist() == [1, 2, 3, 4, 5]
    assert merged["H"][:, 0].tolist() == [0.0, 1.0, 9.0, 9.0, 9.0]

    class Npz(dict):
        files = property(lambda self: list(self))

    sl = P.npz_slice(Npz(base), 2, 3)
    assert sl["frames"].tolist() == [2, 3] and sl["K"].shape == (3, 3)


def team(frames, spots, gk_x, gk_id=1):
    """Rows for players standing at fixed pitch spots each frame, plus a keeper."""
    rows = []
    for f in frames:
        for tid, (x, y) in spots.items():
            rows.append({"frame": f, "track_id": tid, "pitch_x": x, "pitch_y": y, "class": "player"})
        rows.append({"frame": f, "track_id": gk_id, "pitch_x": gk_x, "pitch_y": 34.0, "class": "player"})
    return pd.DataFrame(rows)


def test_shape_match_follows_roles_across_the_change_of_ends():
    # first half: keeper at x = 5, so attacking towards x = 105. Two full-backs, two wingers, a striker.
    h1 = {2: (30, 50), 3: (30, 18), 4: (60, 60), 5: (60, 8), 6: (70, 34)}
    # second half: ends changed, so every spot is mirrored through the centre spot; ids shuffled
    mirror = {12: (105 - 30, 68 - 50), 13: (105 - 30, 68 - 18), 14: (105 - 60, 68 - 60),
              15: (105 - 60, 68 - 8), 16: (105 - 70, 68 - 34)}
    t1 = team(range(10), h1, gk_x=5)
    t2 = team(range(10), mirror, gk_x=100)
    r = P.shape_offsets(t1, list(h1), 1)
    n = P.shape_offsets(t2, list(mirror), 1)
    assert P.match_by_shape(r, n) == {12: 2, 13: 3, 14: 4, 15: 5, 16: 6}


def test_shape_offsets_ignore_rows_after_the_first_substitution():
    h = {2: (30, 50), 3: (30, 18), 4: (60, 60), 5: (60, 8), 6: (70, 34)}
    t = team(range(10), h, gk_x=5)
    late = t["frame"] >= 5
    t.loc[late & (t["track_id"] == 6), "pitch_x"] = 10.0  # after the sub, 6's chain is someone else
    assert P.shape_offsets(t, list(h), 1, until=5)[6] == P.shape_offsets(team(range(5), h, gk_x=5), list(h), 1)[6]


def test_remap_period_maps_identities_and_shifts_fragments():
    tr = pd.DataFrame({"frame": [1, 1, 1], "track_id": [12, 10005, 101], "tracklet": [3, 5, 7]})
    gaps = pd.DataFrame({"track_id": [12], "lo": [1], "partner": [10005.0]})
    links = pd.DataFrame({"tracklet": [3, 5], "identity": [12, -1]})
    out, lk, gp = P.remap_period(tr, links, gaps, {12: 2, 101: 101}, offset=100)
    assert out["track_id"].tolist() == [2, 10105, 101]
    assert out["tracklet"].tolist() == [103, 105, 107]
    assert lk["identity"].tolist() == [2, -1] and lk["tracklet"].tolist() == [103, 105]
    assert gp["track_id"].tolist() == [2] and gp["partner"].tolist() == [10105]


def test_apply_substitutions_splits_the_identity_at_the_frame():
    tracks = pd.DataFrame({"frame": [10, 20, 30, 30], "track_id": [10, 10, 10, 105], "class": "player"})
    gaps = pd.DataFrame({"track_id": [10, 10], "lo": [12, 25]})
    players = {"a10": {"team": "team_a", "number": 10, "track_ids": [10], "role": "AM"},
               "b5": {"team": "team_b", "number": 5, "track_ids": [105]}}
    t2p = {"10": "a10", "105": "b5"}
    subs = [{"team": "team_a", "minute": "62", "frame": 20, "off": {"number": 10}, "on": {"number": 11, "name": "Hwang"}},
            {"team": "team_b", "frame": 25, "off": {"number": 99}, "on": {"number": 9}}]
    tracks, gaps, rep = P.apply_substitutions(tracks, gaps, players, t2p, subs)
    assert tracks["track_id"].tolist() == [10, 12, 12, 105]
    assert gaps["track_id"].tolist() == [10, 12]
    assert players["a11"]["track_ids"] == [12] and players["a11"]["substitute"]
    assert players["a10"]["off"]["frame"] == 20 and t2p["12"] == "a11"
    assert rep[0]["status"] == "applied" and rep[0]["rows_moved"] == 2
    assert rep[1]["status"] == "player going off not found"


def test_identity_map_prefers_analyst_names_then_shape():
    h1 = {2: (30, 50), 3: (30, 18), 4: (60, 60)}
    first_pm = {"players": {f"a{t}": {"team": "team_a", "number": t + 10, "track_ids": [t]} for t in (1, *h1)}}
    first_tracks = team(range(10), h1, gk_x=5)
    mirror = {2: (75, 18), 3: (75, 50), 4: (45, 8)}  # same roles, ends changed
    pm = {"players": {
        "a1": {"team": "team_a", "number": 11, "track_ids": [1]},
        "x2": {"team": "team_a", "number": 14, "track_ids": [2], "name_source": "calibration names"},
        "x3": {"team": "team_a", "number": None, "track_ids": [3]},
        "x4": {"team": "team_a", "number": None, "track_ids": [4]},
    }}
    m, how = P.identity_map(first_pm, first_tracks, pm, team(range(10), mirror, gk_x=100), {})
    # 2 was named #14 (first-half id 4); 3 and 4 are matched by shape to the remaining 2 and 3
    assert m[1] == 1 and m[2] == 4 and how[2] == "named in this half"
    assert {m[3], m[4]} == {2, 3}
