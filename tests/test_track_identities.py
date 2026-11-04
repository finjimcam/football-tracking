import argparse

import numpy as np
import pandas as pd

import track_identities as T


def args(**overrides):
    a = argparse.Namespace(
        gate=1.2, gate_step=0.3, gate_max=3.0, kit_penalty=2.5, max_miss=12, new_conf=0.35,
        vmax=8.5, reach_m=3.0, w_role=1.5, role_tau_s=5.0, role_sigma_m=8.0, c_link=0.5,
        reward=0.5, reward_per_s=1.0, edge_s=20.0, cost_late=8.0, max_gap_s=300.0, max_succ=12,
        overlap_frames=3, w_vel=0.0, contact_role=0.6)
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def test_suppress_duplicates_keeps_the_most_confident_box():
    p = pd.DataFrame({"frame": [1, 1, 1], "conf": [0.6, 0.2, 0.9],
                      "x1": [100, 100.5, 300], "y1": [50, 50, 50],
                      "x2": [112, 112.5, 312], "y2": [80, 80.4, 80]})
    out = T.suppress_duplicates(p)
    assert len(out) == 2 and set(out["conf"]) == {0.6, 0.9}


def test_kit_colour_rules():
    det = pd.DataFrame({
        "shirt_L": [200, 240, 50, 150, np.nan], "shirt_a": [118, 120, 124, 156, np.nan],
        "shirt_b": [118, 137, 129, 163, np.nan], "shirt_n": [50, 50, 50, 50, 0],
        "x1": [0] * 5, "x2": [12] * 5, "y1": [0] * 5, "y2": [30] * 5})
    assert list(T.kit_colour(det)) == ["sky", "white", "dark", "orange", "green"]


def test_build_tracklets_keeps_crossing_players_of_different_teams_apart():
    rows = []
    for f in range(200):  # A runs +x, B runs -x, they pass 0.6 m apart at f ~ 62
        t = f / 25
        rows += [(f, 30 + 4 * t, 30.0, "team_a"), (f, 50 - 4 * t, 30.6, "team_b")]
    frames = np.array([r[0] for r in rows])
    xs, ys = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    kit = np.array([r[3] for r in rows], dtype=object)
    ids, _ = T.build_tracklets(frames, xs, ys, np.full(len(rows), 0.9), kit, lambda f: np.eye(3),
                               (1000, 1000), lambda f: True, 25.0, args())
    a_ids, b_ids = set(ids[kit == "team_a"]), set(ids[kit == "team_b"])
    assert len(a_ids) == 1 and len(b_ids) == 1 and a_ids != b_ids


def test_split_by_kit_cuts_a_tracklet_that_changes_team():
    p = pd.DataFrame({"frame": np.arange(60), "tracklet": 0, "kit": ["team_a"] * 30 + ["team_b"] * 30})
    out, _, cuts = T.split_by_kit(p, next_id=1)
    assert cuts == 1 and out["tracklet"].nunique() == 2
    assert set(out.loc[out["frame"] < 25, "tracklet"]) == {0}
    assert set(out.loc[out["frame"] > 35, "tracklet"]) == {1}


def test_link_flow_chains_each_player_through_off_screen_gaps():
    rows, tid = [], 0
    for y, direction in ((10.0, 1), (50.0, -1)):  # two players, each seen in 3 pieces
        for s, e in ((0, 100), (150, 250), (300, 400)):
            rows.append(dict(tracklet=tid, start=s, end=e, n=e - s + 1,
                             entry_x=50 + direction * 0.08 * s, entry_y=y,
                             exit_x=50 + direction * 0.08 * e, exit_y=y,
                             vin_x=direction * 2.0, vin_y=0.0,
                             vout_x=direction * 2.0, vout_y=0.0, role_x=np.nan, role_y=np.nan))
            tid += 1
    assign, links = T.link_flow(pd.DataFrame(rows), 25.0, args(), (0, 400), n_paths=2)
    chain_a, chain_b = {assign[t] for t in (0, 1, 2)}, {assign[t] for t in (3, 4, 5)}
    assert len(chain_a) == 1 and len(chain_b) == 1 and chain_a != chain_b
    assert len(links) == 4


def test_mostly_unreadable_referee_is_not_a_team_player():
    # The referee's dark-green kit reads as grass; a few 'white' votes come from
    # standing next to Spurs players. He must not become a Spurs outfielder.
    r = next(pd.DataFrame([dict(offpitch=0.0, v_team_a=7, v_team_b=52, v_gk_b=0, v_official=7,
                                v_gk_a=320, v_unknown=13, n=399)]).itertuples())
    assert T.decide_class(r) not in ("team_a", "team_b")
