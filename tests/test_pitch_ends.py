"""pitch_ends.py: each team attacks one way all match (issue #63)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import pitch_ends as PE  # noqa: E402

L, W = 105.0, 68.0
PMAP = {"players": {"a1": {"team": "team_a", "role": "GK", "track_ids": [1, 11]},
                    "a9": {"team": "team_a", "role": "line3-slot1", "track_ids": [9, 19]},
                    "b1": {"team": "team_b", "role": "GK", "track_ids": [2, 12]}}}
CFG = {"periods": {"first_half": {"start_frame": 0, "end_frame": 99},
                   "second_half": {"start_frame": 200, "end_frame": 299}}}


def match():
    """team_a's keeper at x=5 in the first half, x=100 in the second (ends changed); the
    striker on the left wing (y=10) near the goal team_a attacks in both."""
    rows = []
    for f in range(0, 100):
        rows += [(f, 1, 5.0, 34.0), (f, 2, 100.0, 34.0), (f, 9, 90.0, 10.0)]
    for f in range(200, 300):
        rows += [(f, 11, 100.0, 34.0), (f, 12, 5.0, 34.0), (f, 19, 15.0, 58.0)]
    return pd.DataFrame(rows, columns=["frame", "track_id", "pitch_x", "pitch_y"])


def test_second_half_is_turned_round_so_each_team_attacks_one_way():
    t = match()
    flips = PE.flipped_ranges(t, PMAP, CFG, L)
    assert flips == [(200, 299)]
    o = PE.orient(t, flips, L, W)
    gk = o[o["track_id"].isin([1, 11])]["pitch_x"]
    assert np.allclose(gk, 5.0)  # keeper at the same end all match
    st = o[o["track_id"].isin([9, 19])]
    assert np.allclose(st["pitch_x"], 90.0) and np.allclose(st["pitch_y"], 10.0)  # same wing, same end
    assert PE.attacking_x(o, PMAP, L) == {"team_a": L, "team_b": 0.0}
    assert t["pitch_x"].iloc[-1] == 15.0  # input untouched


def test_no_periods_or_same_ends_change_nothing():
    t = match()
    assert PE.flipped_ranges(t, PMAP, {}, L) == []
    t2 = t[t["frame"] < 100]
    assert PE.flipped_ranges(t2, PMAP, CFG, L) == []
    assert PE.orient(t, [], L, W) is t
