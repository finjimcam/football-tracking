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


# ---- issue #46: frame context around handovers --------------------------------
def test_duplicate_box_on_a_tracked_player_does_not_start_a_tracklet():
    rows = []
    for f in range(100):  # one player jogging; frames 40-47 carry a second box 0.3 m off him
        x = 30 + 3 * f / 25
        rows.append((f, x, 30.0))
        if 40 <= f < 48:
            rows.append((f, x + 0.3, 30.1))
    frames = np.array([r[0] for r in rows])
    xs, ys = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    kit = np.full(len(rows), "team_a", dtype=object)
    run = lambda a: T.build_tracklets(frames, xs, ys, np.full(len(rows), 0.9), kit, lambda f: np.eye(3),
                                      (1000, 1000), lambda f: True, 25.0, a)[0]
    assert len(set(run(args())) - {-1}) == 2               # old behaviour: the duplicate spawns a tracklet
    assert set(run(args(dup_m=0.8))) - {-1} == {0}           # now: one player, one tracklet


def _handover_world():
    """Royal at 26:58 in miniature. Player P jogs along y=30. Tracklet 0
    follows him to frame 49, coasts (no boxes) 50-53, and then grabs
    neighbour N's box 1.3 m away from frame 54 on. Tracklet 1 starts on P at
    frame 52 and follows him from there."""
    rows = []
    for f in range(0, 120):
        x = 30 + 3 * f / 25
        if f < 50:
            rows.append((f, 0, x, 30.0))
        if f >= 52:
            rows.append((f, 1, x, 30.0))
        if f >= 54:
            rows.append((f, 0, x + 1.0, 30.8))  # N, picked up by the coasting tracklet 0
    p = pd.DataFrame(rows, columns=["frame", "tracklet", "pitch_x", "pitch_y"])
    tl = pd.DataFrame({"tracklet": [0, 1], "start": [0, 52], "end": [119, 119], "cls": ["team_a", "team_a"]})
    return p, tl


def test_split_at_handovers_cuts_the_coasting_tracklet_where_the_new_one_takes_over():
    p, tl = _handover_world()
    out, next_id, n, tails = T.split_at_handovers(p, tl, 2, 25.0)
    assert n == 1 and tails == {2: 0}
    head = out[out.tracklet == 0]
    assert head.frame.max() == 49                     # tracklet 0 is P up to where it lost him
    assert set(out[out.frame >= 54][lambda d: d.pitch_y == 30.8].tracklet) == {2}  # its tail (N) is separate
    assert (out[out.tracklet == 1].pitch_y == 30.0).all()                           # tracklet 1 untouched


def test_split_at_handovers_leaves_a_different_player_alone():
    p, tl = _handover_world()
    p.loc[p.tracklet == 1, "pitch_y"] = 34.0          # the new tracklet starts 4 m away: someone else
    _, _, n, _ = T.split_at_handovers(p, tl, 2, 25.0)
    assert n == 0
    p, tl = _handover_world()
    extra = pd.DataFrame({"frame": [52], "tracklet": [0], "pitch_x": [30 + 3 * 52 / 25], "pitch_y": [31.5]})
    p = pd.concat([p, extra])                         # tracklet 0 has its own box 1.5 m away on frame 52
    _, _, n, _ = T.split_at_handovers(p, tl, 2, 25.0)
    assert n == 0                                     # both detected, not a duplicate: two players
