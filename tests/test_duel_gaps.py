"""Issue #47 item 3: duel / absorbed-gap filler on synthetic tracks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from duel_gaps import apply, classify_gaps, summarise  # noqa: E402

FPS = 25


def rows(tid, frames, xy, p_team_a):
    xy = np.asarray(xy, float)
    return pd.DataFrame({
        "frame": frames, "t_s": np.asarray(frames) / FPS, "track_id": tid, "class": "player",
        "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 30.0, "foot_x": 5.0, "foot_y": 30.0, "conf": 0.8,
        "pitch_x": xy[:, 0], "pitch_y": xy[:, 1], "tracklet": float(tid), "kit": None,
        "p_team_a": p_team_a, "interp": None})


def duel(k=2, P=102, n=150, gap=(50, 89), swap=False, k_far=False, kit_swap=False):
    """P runs along y=30 at 4 m/s. k comes up from below at 2 m/s, runs
    shoulder to shoulder with P (0.8 m below it) while hidden in frames
    gap[0]..gap[1], then carries on upwards. swap=True: at the split the ids
    are exchanged (k's id comes out on P's body and vice versa). kit_swap:
    k's detections after the split read as the other team. k_far: k
    re-emerges 6 m away from P."""
    f = np.arange(n)
    px = 10 + 4 * f / FPS
    p_xy = np.c_[px, np.full(n, 30.0)]
    ride0 = gap[0] - 2
    side = np.where(f < ride0, -0.8 - 0.08 * (ride0 - f), np.where(f <= gap[1], -0.8, -0.8 + 0.08 * (f - gap[1])))
    k_xy = np.c_[px - 0.3, 30.0 + side]
    pa_k = 0.95 if k <= 11 else 0.05
    pa_p = 0.95 if P <= 11 else 0.05
    keep = (f < gap[0]) | (f > gap[1])
    kk = k_xy.copy()
    if k_far:
        kk[f > gap[1]] += [0.0, -6.0]
    a = rows(k, f[keep], kk[keep], np.where(f[keep] > gap[1], 1 - pa_k if kit_swap else pa_k, pa_k))
    b = rows(P, f, p_xy, pa_p)
    if swap:
        post_a, post_b = a["frame"] > gap[1], b["frame"] > gap[1]
        a.loc[post_a, "track_id"], b.loc[post_b, "track_id"] = P, k
        a.loc[post_a, "p_team_a"], b.loc[post_b, "p_team_a"] = pa_p, pa_k
        a.loc[post_a, "tracklet"], b.loc[post_b, "tracklet"] = float(P), float(k)  # tracker-level swap
    return pd.concat([a, b], ignore_index=True)


def one(recs, k=2):
    duel_recs = [r for r in recs if r["identity"] == k and r["kind"] == "duel"]
    assert len(duel_recs) == 1, recs
    return duel_recs[0]


def test_cross_team_duel_is_filled_along_partner_path():
    t = duel()
    out, recs = apply(t, FPS, 0, 149)
    r = one(recs)
    assert r["verdict"] == "filled" and r["partner"] == 102 and r["frames"] == 40
    fill = out[out["absorbed_by"].notna()]
    assert len(fill) == 40 and set(fill["track_id"]) == {2}
    assert fill["interp"].all() and (fill["conf"] == 0).all() and not fill["absorbed_review"].any()
    # truth: k stays 0.8 m beside P the whole gap; the fill follows P, not a line
    f = fill["frame"].to_numpy()
    truth_x = 10 + 4 * f / FPS - 0.3
    assert np.allclose(fill["pitch_x"], truth_x, atol=1e-6)
    assert np.allclose(fill["pitch_y"], 29.2, atol=0.1)
    s = summarise(recs, FPS)
    assert s["duel_gaps"] == 1 and s["filled"] == 1 and s["filled_cross_team"] == 1


def test_fill_rides_partner_turns_unlike_straight_line():
    """Partner zig-zags in the gap: the fill must follow it (a straight line wouldn't)."""
    t = duel()
    m = (t["track_id"] == 102) & t["frame"].between(50, 89)
    t.loc[m, "pitch_y"] += 2.0 * np.sin((t.loc[m, "frame"] - 50) / 39 * np.pi)
    out, recs = apply(t, FPS, 0, 149)
    fill = out[out["absorbed_by"].notna()].set_index("frame")
    p = t[t["track_id"] == 102].set_index("frame")
    assert np.allclose(fill["pitch_y"] - p.loc[fill.index, "pitch_y"], -0.8, atol=0.1)


def test_swap_at_split_between_teammates_is_flagged_not_filled():
    t = duel(k=2, P=3, swap=True)
    out, recs = apply(t, FPS, 0, 149)
    r = one(recs)
    assert r["verdict"] == "ambiguous", r
    assert "swap" in r["why"] or "sides" in r["why"]
    assert out["absorbed_by"].notna().sum() == 0  # not written by default
    out2, _ = apply(t, FPS, 0, 149, keep_ambiguous=True)
    rv = out2[out2["absorbed_by"].notna()]
    assert len(rv) == 40 and rv["absorbed_review"].all()


def test_same_team_duel_without_swap_is_filled():
    t = duel(k=2, P=3, swap=False)
    _, recs = apply(t, FPS, 0, 149)
    assert one(recs)["verdict"] == "filled"


def test_kit_after_split_contradicting_team_is_flagged():
    t = duel(kit_swap=True)
    _, recs = apply(t, FPS, 0, 149)
    r = one(recs)
    assert r["verdict"] == "ambiguous" and "kit" in r["why"]


def test_reemerging_away_from_partner_is_not_absorbed():
    t = duel(k_far=True)
    _, recs = apply(t, FPS, 0, 149)
    r = one(recs)
    assert r["verdict"] == "no_partner" and "away" in r["why"]


def test_gap_far_from_anyone_is_not_a_duel():
    t = duel()
    t = t[~((t["track_id"] == 102) & (t["frame"] < 60))]  # nobody near k when it is lost
    recs = classify_gaps(t, FPS, 0, 149)
    assert [r["kind"] for r in recs if r["identity"] == 2] == ["other"]


def test_partner_must_be_tracked_through_the_gap():
    t = duel()
    t = t[~((t["track_id"] == 102) & t["frame"].between(55, 85))]
    _, recs = apply(t, FPS, 0, 149)
    assert one(recs)["verdict"] == "no_partner"


def test_partner_relinked_inside_the_gap_is_not_trusted():
    """The partner identity hops to another tracklet mid-gap (e.g. onto a
    referee): its path there is a guess, so it can't carry k."""
    t = duel()
    t.loc[(t["track_id"] == 102) & t["frame"].between(60, 80), "tracklet"] = 999.0
    _, recs = apply(t, FPS, 0, 149)
    r = one(recs)
    assert r["verdict"] == "no_partner" and "re-linked" in r["why"]


def test_too_long_gap_is_left_open():
    t = duel(n=400, gap=(50, 340))
    _, recs = apply(t, FPS, 0, 399, max_gap_s=10.0)
    r = one(recs)
    assert r["verdict"] == "no_partner" and r["why"] == "too long"


def test_rerun_is_idempotent():
    t = duel()
    out1, _ = apply(t, FPS, 0, 149)
    out2, recs2 = apply(out1, FPS, 0, 149)
    assert len(out1) == len(out2)
    assert one(recs2)["verdict"] == "filled"  # classified on observed rows, not on its own fill


class _Ctx:
    """Minimal stand-in for gap_status.GapContext (issue-47-status branch)."""
    fps = FPS

    def __init__(self, t, with_index=False):
        self.t = t
        if with_index:
            from duel_gaps import TrackIndex
            self.duel_index = TrackIndex(t)

    def track(self, tid):
        s = self.t[self.t["track_id"] == tid].sort_values("frame")
        return s["frame"].to_numpy(), s[["pitch_x", "pitch_y"]].to_numpy()


def _gap(t, k=2, P=102, f0=49, f1=90):
    from types import SimpleNamespace
    xy = t[t["track_id"] == k].set_index("frame")[["pitch_x", "pitch_y"]]
    return SimpleNamespace(track_id=k, start=f0, end=f1, lo=f0 + 1, hi=f1 - 1, n_missing=f1 - f0 - 1,
                           xy_start=xy.loc[f0].to_numpy(), xy_end=xy.loc[f1].to_numpy(), partner=P, duel=True)


def test_plugin_fill_matches_tool_fill():
    from duel_gaps import fill
    t = duel()
    out, _ = apply(t, FPS, 0, 149)
    ref = out[out["absorbed_by"].notna()].sort_values("frame")[["pitch_x", "pitch_y"]].to_numpy()
    for ctx in (_Ctx(t), _Ctx(t, with_index=True)):
        xy = fill(_gap(t), ctx)
        assert xy.shape == (40, 2) and np.allclose(xy, ref)


def test_plugin_fill_declines_swap_with_index_only():
    from duel_gaps import fill
    t = duel(k=2, P=3, swap=True)
    assert fill(_gap(t, P=3), _Ctx(t, with_index=True)) is None  # identity check at the split
    assert fill(_gap(t, P=3), _Ctx(t)) is not None                 # geometry alone can't tell


def test_chains_show_absorbed_rows_as_filled():
    """Lines UI / phase clips: a duel-filled row is followed like a detection
    but drawn as 'filled' (amber), not 'seen'."""
    from cut_phase_clips import fill_gaps, index_rows, track_run
    out, _ = apply(duel(), FPS, 0, 149)
    rows_idx = index_rows({int(f): g for f, g in out.groupby("frame")})
    pts = fill_gaps(track_run(rows_idx, 2, range(0, 150), FPS), FPS, None, 0, 149)
    assert pts[60][5] == "filled" and pts[10][5] == "seen" and pts[120][5] == "seen"
