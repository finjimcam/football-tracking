import argparse

import numpy as np
import pandas as pd

import track_identities as T


def args(**overrides):
    a = argparse.Namespace(
        gate=1.2, gate_step=0.3, gate_max=3.0, kit_penalty=2.5, max_miss=12, new_conf=0.35,
        vmax=8.5, reach_m=3.0, w_role=1.5, role_tau_s=5.0, role_sigma_m=8.0, c_link=0.5,
        reward=0.5, reward_per_s=1.0, edge_s=20.0, cost_late=8.0, max_gap_s=300.0, max_succ=12,
        overlap_frames=3, w_vel=0.0, contact_role=0.6, w_shape=1.0, shape_sigma_m=6.0)
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


# ---- issue #46: linking through the analyst's names ---------------------------
def _two_players():
    """Two players, each seen in 3 pieces: A along y=10 (tracklets 0-2), B along
    y=50 (3-5). Plain motion links A-A-A and B-B-B."""
    rows, tid = [], 0
    for y, direction in ((10.0, 1), (50.0, -1)):
        for s, e in ((0, 100), (150, 250), (300, 400)):
            rows.append(dict(tracklet=tid, start=s, end=e, n=e - s + 1,
                             entry_x=50 + direction * 0.08 * s, entry_y=y,
                             exit_x=50 + direction * 0.08 * e, exit_y=y,
                             vin_x=direction * 2.0, vin_y=0.0, vout_x=direction * 2.0, vout_y=0.0,
                             role_x=0.0, role_y=y - 30.0))
            tid += 1
    return pd.DataFrame(rows)


def test_link_anchored_follows_the_names_even_against_the_motion():
    T_ = _two_players()
    # the analyst says the first A piece and the LAST B piece are the same player
    out = T.link_anchored(T_, 25.0, args(w_shape=0.0, max_gap_s=300.0), (0, 400), {0: 0, 5: 0, 3: 1}, 2)
    assert out is not None
    assign, links = out
    assert assign[0] == assign[5] == 0 and assign[3] == 1


def test_link_anchored_uses_the_team_shape_between_names():
    T_ = _two_players()
    assign, _ = T.link_anchored(T_, 25.0, args(), (0, 400), {0: 0, 3: 1}, 2)
    assert {assign[t] for t in (0, 1, 2)} == {0} and {assign[t] for t in (3, 4, 5)} == {1}


def test_split_conflicts_cuts_a_tracklet_named_as_two_players():
    from identity_anchors import split_conflicts
    p = pd.DataFrame({"frame": np.arange(100), "tracklet": np.zeros(100, int)})
    res = [({"team": "team_a", "number": 7, "frame": 10}, 0), ({"team": "team_a", "number": 9, "frame": 60}, 0)]
    p2, cuts, tails = split_conflicts(p, res, 5)
    assert cuts == 1 and tails == {5: 0}
    assert set(p2.loc[p2.frame <= 35, "tracklet"]) == {0} and set(p2.loc[p2.frame >= 36, "tracklet"]) == {5}


def test_team_pins_keep_the_newest_name():
    from identity_anchors import team_pins
    span = {1: (0, 100), 2: (50, 150)}
    res = [({"team": "team_a", "number": 7, "created": 1}, 1), ({"team": "team_a", "number": 7, "created": 2}, 2),
           ({"team": "team_b", "number": 7, "created": 3}, 1)]
    pins, dropped = team_pins(res, "team_a", [9, 7], span)
    assert pins == {2: 1} and dropped == 1


def _tl_row(t, s, e, ident, ex, ey, nx, ny, cls="team_a", team="team_a"):
    return dict(tracklet=t, start=s, end=e, n=e - s + 1, identity=ident, team=team if ident >= 0 else "",
                link="", cls=cls, entry_x=ex, entry_y=ey, exit_x=nx, exit_y=ny)


def test_fill_partial_gives_the_gap_part_of_an_overlapping_tracklet_to_the_player():
    # player 5 seen 0-49 and 150-199 along y=30; tracklet 9 (unlabelled) starts at 40 on him
    # (a duplicate) and runs on to 170, so it never fits his gap whole
    rows = []
    for f in range(200):
        x = 20 + 2 * f / 25
        if f < 50:
            rows.append((f, 0, x, 30.0))
        if f >= 150:
            rows.append((f, 1, x, 30.0))
        if 40 <= f <= 170:
            rows.append((f, 9, x + 0.4, 30.3))
    p = pd.DataFrame(rows, columns=["frame", "tracklet", "pitch_x", "pitch_y"])
    p = p.assign(kit="team_a", conf=0.9)
    tl = T.summarize(p, 25.0, 105.0, 68.0)
    tl["identity"] = tl["tracklet"].map({0: 5, 1: 5, 9: -1})
    tl["team"] = np.where(tl["identity"] >= 0, "team_a", "")
    tl["link"] = ""
    p2, tl2, n = T.fill_partial(p, tl, 25.0, 105.0, 68.0)
    assert n == 1
    piece = tl2[tl2["link"] == "gap_piece"].iloc[0]
    assert piece["identity"] == 5 and piece["start"] == 50 and piece["end"] == 149


def test_assign_by_elimination_gives_a_loose_tracklet_to_the_only_player_off_screen():
    a = args()
    tl = pd.DataFrame([
        _tl_row(0, 0, 400, 2, 10, 10, 20, 10),        # player 2: on screen throughout
        _tl_row(1, 0, 100, 3, 40, 40, 42, 40),        # player 3: seen, then gone 101-299
        _tl_row(2, 300, 400, 3, 46, 40, 50, 40),
        _tl_row(3, 150, 250, -1, 43, 40, 45, 40),     # loose piece in 3's gap
    ])
    roles = pd.DataFrame({"rx": [0.0, 5.0, 5.0, 5.0], "ry": [-10.0, 10.0, 10.0, 10.0]}, index=[0, 1, 2, 3])
    n = T.assign_by_elimination(tl, roles, 25.0, a, [2, 3])
    assert n == 1 and tl.loc[tl.tracklet == 3, "identity"].iat[0] == 3
def _kit_rows(rows):
    """rows: (shirt_L, shirt_a, shirt_b, shirt_n, shorts_L, shorts_a, shorts_n), 20x60 px boxes."""
    cols = ["shirt_L", "shirt_a", "shirt_b", "shirt_n", "shorts_L", "shorts_a", "shorts_n"]
    d = pd.DataFrame(rows, columns=cols, dtype=float)
    d["shorts_b"] = 130.0
    d["x1"], d["x2"], d["y1"], d["y2"] = 0.0, 20.0, 0.0, 60.0
    return d


def test_kit_colour_extra_classes_only_when_named():
    d = _kit_rows([
        (140, 113, 173, 200, 25, 120, 50),   # yellow keeper
        (160, 115, 140, 100, 25, 120, 50),   # mint referee shirt, black shorts
        (150, 115, 140, 0, 25, 120, 50),     # referee shirt masked as grass, black shorts
        (150, 115, 140, 0, 25, 134, 50),     # grass-masked shirt over red shorts: not a referee
        (113, 149, 154, 200, 82, 134, 50),   # red outfield shirt
        (181, 125, 134, 200, 182, 124, 50),  # white outfield shirt
    ])
    base = T.kit_colour(d)
    assert list(T.kit_colour(d, extra={"sky", "white"})) == list(base)  # unnamed extras change nothing
    assert base[0] != "yellow" and base[1] != "mint"
    got = list(T.kit_colour(d, extra={"yellow", "mint"}))
    assert got == ["yellow", "mint", "mint", "green", "orange", "white"]
