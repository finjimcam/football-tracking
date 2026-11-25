"""Issue #47 item 4: smoothed fill of short identity gaps (tools/interp_gaps.py)."""
import numpy as np
import pandas as pd

from interp_gaps import fill_gap, fill_tracks, gap_table, holdout, linear_fill, rts_smooth, strip_fill

FPS = 25.0


def arc(n=200, speed=6.0, radius=10.0):
    """A player running a quarter circle at constant speed - a straight line
    cuts the corner, a velocity-aware fill shouldn't."""
    t = np.arange(n) / FPS
    th = speed * t / radius
    return np.c_[30 + radius * np.sin(th), 30 + radius * (1 - np.cos(th))]


def tracks_from(paths):
    """{track_id: (frames, (n,2) xy)} -> tracks.parquet-shaped frame."""
    rows = []
    for tid, (fr, xy) in paths.items():
        for f, (x, y) in zip(fr, xy):
            rows.append({"frame": int(f), "t_s": f / FPS, "track_id": tid, "class": "player",
                         "x1": 100.0, "y1": 100.0, "x2": 120.0, "y2": 160.0, "foot_x": 110.0, "foot_y": 160.0,
                         "conf": 0.9, "pitch_x": x, "pitch_y": y, "kit": "team_a"})
    return pd.DataFrame(rows)


def test_rts_beats_straight_line_on_a_curved_run():
    xy = arc()
    fr = np.arange(len(xy))
    keep = np.ones(len(fr), bool)
    keep[80:130] = False  # 2 s gap
    fk, xk = fr[keep], xy[keep]
    i = 79
    gap_f, path, how = fill_gap(fk, xk, i, FPS, method="rts")
    assert how == "smooth"
    lin = linear_fill(fk[i], xk[i], fk[i + 1], xk[i + 1], gap_f)
    err = np.hypot(*(path - xy[80:130]).T).mean()
    err_lin = np.hypot(*(lin - xy[80:130]).T).mean()
    assert err < 0.5 * err_lin
    # the fill joins the sightings exactly (no jump at either end)
    assert np.hypot(*(path[0] - xk[i])) < 0.5 and np.hypot(*(path[-1] - xk[i + 1])) < 0.5


def test_rts_reproduces_constant_velocity():
    fr = np.r_[0:20, 40:60]
    xy = np.c_[fr * 0.2, np.full(len(fr), 10.0)]
    pos, vel = rts_smooth(fr, xy, np.arange(20, 40), FPS)
    assert np.allclose(pos[:, 0], np.arange(20, 40) * 0.2, atol=1e-3)
    assert np.allclose(vel[:, 0], 0.2 * FPS, atol=1e-2)


def test_fill_tracks_fills_short_skips_long_duel_and_implausible():
    walk = lambda fr, x0: np.c_[x0 + fr * 0.1, np.full(len(fr), 20.0)]  # noqa: E731
    f1 = np.r_[0:50, 70:120, 300:350]          # 0.8 s gap (filled), 7 s gap (too long)
    f2 = np.r_[0:50, 60:100]                   # a 10-frame gap, but a teammate stands next to it (duel)
    f3 = np.r_[0:50, 60:100]                   # jump of 50 m in 0.4 s (implausible)
    xy3 = walk(f3, 60.0)
    xy3[50:, 1] += 50
    t = tracks_from({2: (f1, walk(f1, 0.0)), 3: (f2, walk(f2, 30.0)),
                     4: (f2, walk(f2, 30.5)), 5: (f3, xy3)})
    out, st = fill_tracks(t, FPS, max_gap_s=3.0)
    fill = out[out["interp"] == True]  # noqa: E712
    assert set(fill["track_id"]) == {2}
    assert sorted(fill["frame"]) == list(range(50, 70))
    assert (fill["status"] == "interpolated").all() and (fill["conf"] == 0).all()
    assert st["too_long"] == 1 and st["implausible"] == 1 and st["duel_skipped"] == 2  # ids 3 and 4
    # duels can be filled on request
    out2, st2 = fill_tracks(t, FPS, max_gap_s=3.0, skip_duels=False)
    assert st2["duel_skipped"] == 0 and set(out2.loc[out2["interp"] == True, "track_id"]) == {2, 3, 4}  # noqa: E712
    # re-running replaces the old fill instead of stacking a second copy
    again, _ = fill_tracks(out, FPS, max_gap_s=3.0)
    assert len(again) == len(out)
    assert len(strip_fill(out)) == len(t)


def test_fill_rows_get_projected_foot_point():
    fr = np.r_[0:30, 40:70]
    xy = np.c_[fr * 0.1, np.full(len(fr), 20.0)]
    H = np.array([[10.0, 0, 5], [0, 10.0, 7], [0, 0, 1]])  # pitch -> image
    out, _ = fill_tracks(tracks_from({7: (fr, xy)}), FPS, world_to_img=lambda f: H)
    r = out[out["interp"] == True].iloc[0]  # noqa: E712
    assert np.isclose(r["foot_x"], 10 * r["pitch_x"] + 5) and np.isclose(r["foot_y"], 10 * r["pitch_y"] + 7)
    assert np.isclose(r["x2"] - r["x1"], 20.0) and np.isclose(r["y2"], r["foot_y"])


def test_gap_table_buckets_and_holdout():
    fr = np.r_[0:100, 110:200, 300:1000]   # 10-frame gap (<1 s), 100-frame gap (3-10 s)
    xy = arc(1000, speed=3.0, radius=30.0)[fr]
    t = tracks_from({2: (fr, xy)})
    g = gap_table(t, FPS, 0, 999)
    assert g["missing_frames"] == {"<1s": 10, "1-3s": 0, "3-10s": 100, ">10s": 0} and g["gaps"] == 2
    filled, _ = fill_tracks(t, FPS, max_gap_s=3.0)
    assert gap_table(filled, FPS, 0, 999)["missing_frames"]["<1s"] == 0
    res = holdout(t, FPS, gap_lengths_s=(1,), n_per=20)
    assert res[1]["rts"].mean() < res[1]["linear"].mean()


def test_make_filler_plugs_into_gap_status_interface():
    from types import SimpleNamespace

    from interp_gaps import make_filler
    xy = arc()
    fr = np.r_[0:80, 130:200]

    class Ctx:
        fps = FPS

        def track(self, tid):
            return fr, xy[fr]

    gap = SimpleNamespace(track_id=2, start=79, end=130, lo=80, hi=129, n_missing=50)
    pts = make_filler()(gap, Ctx())
    assert pts.shape == (50, 2) and np.hypot(*(pts - xy[80:130]).T).mean() < 0.3
    assert make_filler(max_gap_s=1.0)(gap, Ctx()) is None          # too long -> unknown
    assert make_filler()(SimpleNamespace(track_id=2, start=None, end=10, lo=0, hi=9), Ctx()) is None
