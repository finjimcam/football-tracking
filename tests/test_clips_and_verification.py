import numpy as np
import pandas as pd

import verify_tracking as V
from cut_phase_clips import depth_lines, phases
from detect_frames import kit_features, model_input_size
from stitch_tracks import find_stitches
from track_identities import kit_colour


def test_phases_merge_short_interruptions_drop_blips_and_pad():
    state = ["team_a"] * 1000
    for a, b in ((100, 199), (220, 299), (600, 640)):
        state[a:b + 1] = ["team_b"] * (b - a + 1)
    tl = pd.DataFrame({"frame": np.arange(1000), "state": state})
    segs = phases(tl, "team_a", "out_of_possession", 25.0, merge_s=2.0, min_s=3.0,
                  pre_s=1.0, post_s=1.0, lo=0, hi=999)
    assert segs == [[75, 324, 100, 299]]  # 0.8 s gap merged; 1.6 s blip dropped; 1 s padding


def test_depth_lines_find_back_four_and_midfield():
    rows = pd.DataFrame({"pitch_x": [10, 11, 12, 10.5, 25, 26, 24, 40],
                         "pitch_y": [5, 25, 45, 60, 20, 34, 50, 34]})
    lines = depth_lines(rows, own_goal_x=0.0)
    assert sorted(lines[0].tolist()) == [0, 1, 2, 3] and sorted(lines[1].tolist()) == [4, 5, 6]


def test_depth_lines_refuse_incoherent_or_sparse_views():
    rows = pd.DataFrame({"pitch_x": [10, 30, 50, 70, 80, 85, 90, 95], "pitch_y": [34] * 8})
    assert depth_lines(rows, 0.0)[0] is None      # deepest four span 60 m: not a line
    assert depth_lines(rows.iloc[:5], 0.0) == []  # too few on screen to know who defends


def make_players(mixed, seed=0):
    rng = np.random.default_rng(seed)
    offsets = np.array([[-20, -15], [-20, 15], [0, 0], [15, -10], [15, 10]], float)
    rows = []
    for f in range(0, 500, 5):
        perm = rng.permutation(5) if mixed else np.arange(5)
        centre = np.array([50 + 10 * np.sin(f / 50), 34.0])
        for k in range(5):
            x, y = centre + offsets[perm[k]] + rng.normal(0, 1.5, 2)
            rows.append((f, 1 + k, x, y))
    return pd.DataFrame(rows, columns=["frame", "track_id", "pitch_x", "pitch_y"])


def test_role_separability_collapses_when_identities_mix():
    pm = {"players": {f"a{k}": {"team": "team_a", "track_ids": [1 + k], "role": "line1"} for k in range(5)}}
    s_ok = V.role_separability(make_players(False), pm)["team_a"]["separability"]
    s_mix = V.role_separability(make_players(True), pm)["team_a"]["separability"]
    assert s_ok > 3 and s_mix < 0.5


def test_detection_input_size_and_kit_features():
    assert model_input_size(1280, 720, 1280) == (736, 1280)
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:] = (40, 140, 40)                    # grass
    frame[20:50, 40:60] = (221, 171, 108)       # City sky-blue shirt (BGR of #6CABDD)
    feats = kit_features(frame, (35, 10, 65, 90))
    det = pd.DataFrame([dict(zip(["shirt_L", "shirt_a", "shirt_b", "shirt_n"], feats[:4]),
                             x1=35, y1=10, x2=65, y2=90)])
    assert feats[3] > 20 and kit_colour(det)[0] == "sky"


def test_legacy_stitch_joins_the_continuation_only():
    profiles = {
        1: dict(start_t=0, end_t=5, entry=np.array([0.0, 10.0]), exit=np.array([10.0, 10.0]), vel=np.array([2.0, 0.0])),
        2: dict(start_t=6, end_t=9, entry=np.array([12.2, 10.0]), exit=np.array([18.0, 10.0]), vel=np.zeros(2)),
        3: dict(start_t=6, end_t=9, entry=np.array([40.0, 40.0]), exit=np.array([40.0, 40.0]), vel=np.zeros(2)),
    }
    assert [(a, b) for a, b, _ in find_stitches(profiles, 2.0, 2.0, 3.0, 8.0)] == [(1, 2)]
