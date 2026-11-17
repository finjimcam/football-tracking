"""Gap labelling (issue #47 item 2): one synthetic track per status case."""
import numpy as np
import pandas as pd
import pytest

import gap_status as gs

FPS = 25.0


def rows(tid, frames, xy):
    xy = np.asarray(xy, float)
    if xy.ndim == 1:
        xy = np.tile(xy, (len(frames), 1))
    return pd.DataFrame({"frame": np.asarray(frames, int), "t_s": np.asarray(frames) / FPS, "track_id": tid,
                         "class": "player", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
                         "foot_x": xy[:, 0] * 10, "foot_y": xy[:, 1] * 10, "conf": 0.9,
                         "pitch_x": xy[:, 0], "pitch_y": xy[:, 1]})


def walk(frames, x0, y0, vx=0.0, vy=0.0):
    f = np.asarray(frames, float)
    return np.column_stack([x0 + vx * (f - f[0]) / FPS, y0 + vy * (f - f[0]) / FPS])


def label(df, ids, window=(0, 199), **kw):
    return gs.label_tracks(df, FPS, track_ids=ids, window=window, **kw)


def test_short_gap_far_from_others_is_interpolated():
    seen = [f for f in range(200) if not 50 <= f < 80]  # 30-frame (1.2 s) gap
    df = pd.concat([rows(1, seen, walk(seen, 10, 10, vx=2.0)),
                    rows(2, range(200), (60.0, 40.0))])  # nobody near player 1
    out, gaps = label(df, [1])
    g = gaps[(gaps["track_id"] == 1) & gaps["start"].notna() & gaps["end"].notna()].iloc[0]
    assert g["kind"] == "interpolated" and g["reason"] == "short" and not g["duel"]
    fill = out[(out["track_id"] == 1) & (out["status"] == "interpolated")]
    assert sorted(fill["frame"]) == list(range(50, 80))
    # straight line in pitch space between the two sightings
    assert np.allclose(fill["pitch_x"], 10 + 2.0 * fill["frame"] / FPS, atol=0.11)
    assert (out.loc[out["status"] == "detected", "conf"] > 0).all()


def test_duel_gap_with_reemergence_is_absorbed():
    # player 1 runs into opponent 2 (both at x=20), vanishes for 1 s, reappears beside 2
    seen = [f for f in range(200) if not 100 <= f < 125]
    xy1 = np.where(np.asarray(seen)[:, None] < 100, (20.3, 30.0), (20.8, 30.5))
    df = pd.concat([rows(1, seen, xy1), rows(2, range(200), (20.0, 30.0))])
    out, gaps = label(df, [1])
    g = gaps[gaps["lo"] == 100].iloc[0]
    assert g["kind"] == "absorbed" and g["duel"] and g["reemerge"] and g["partner"] == 2
    assert (out.loc[out["frame"].between(100, 124) & (out["track_id"] == 1), "status"] == "absorbed").all()


def test_duel_without_reemergence_near_partner_is_interpolated():
    # last seen next to 2, but comes back 10 m away from 2 -> not absorbed (runnable, short)
    seen = [f for f in range(200) if not 100 <= f < 125]
    xy1 = np.where(np.asarray(seen)[:, None] < 100, (20.3, 30.0), (24.0, 38.0))
    df = pd.concat([rows(1, seen, xy1), rows(2, range(200), (20.0, 30.0))])
    _, gaps = label(df, [1])
    g = gaps[gaps["lo"] == 100].iloc[0]
    assert g["duel"] and not g["reemerge"] and g["kind"] == "interpolated"


def test_long_gap_is_unknown_and_has_no_rows():
    seen = [f for f in range(400) if not 50 <= f < 250]  # 8 s
    df = rows(1, seen, walk(seen, 10, 10, vx=1.0))
    out, gaps = label(df, [1], window=(0, 399))
    g = gaps[gaps["lo"] == 50].iloc[0]
    assert g["kind"] == "unknown" and g["reason"] == "too_long"
    assert not out["frame"].between(50, 249).any()


def test_unrunnable_gap_is_unknown():
    # 40 m in 1 s: an identity jump, not a run - never draw a phantom trail
    seen = [f for f in range(200) if not 50 <= f < 75]
    xy = np.where(np.asarray(seen)[:, None] < 50, (10.0, 10.0), (50.0, 10.0))
    _, gaps = label(rows(1, seen, xy), [1])
    assert gaps[gaps["lo"] == 50].iloc[0]["reason"] == "not_runnable"


def test_open_ends_are_unknown():
    seen = list(range(30, 170))
    _, gaps = label(rows(1, seen, (10.0, 10.0)), [1])
    assert list(gaps["reason"]) == ["open_end", "open_end"]
    assert gaps["n_missing"].sum() == 60


def test_relabel_is_idempotent_and_summary_counts_unknown():
    seen = [f for f in range(200) if not (50 <= f < 80 or f >= 190)]
    df = rows(1, seen, walk(seen, 10, 10, vx=1.0))
    out1, gaps1 = label(df, [1])
    out2, gaps2 = label(out1, [1])
    assert len(out1) == len(out2) and gaps1.equals(gaps2)
    s = gs.summarize(out2, gaps2, [1], (0, 199), FPS)
    assert s["frames_by_status"] == {"detected": 160, "interpolated": 30, "absorbed": 0, "unknown": 10}
    assert s["missing_by_gap_length"]["1-3s"]["frames"] == 30
    assert s["missing_by_gap_length"]["0-1s"]["frames"] == 10


def test_custom_filler_and_refusal_plug_in():
    seen = [f for f in range(200) if not 50 <= f < 80]
    df = rows(1, seen, (10.0, 10.0))
    calls = []

    def hold(gap, ctx):
        calls.append(gap)
        return np.tile(gap.xy_start, (gap.n_missing, 1))

    out, _ = label(df, [1], fillers={"interpolated": hold})
    assert len(calls) == 1 and (out["status"] == "interpolated").sum() == 30
    out, gaps = label(df, [1], fillers={"interpolated": lambda gap, ctx: None})
    assert (out["status"] == "detected").all()
    assert gaps[gaps["lo"] == 50].iloc[0]["kind"] == "unknown"


def test_weights_and_observed():
    df = pd.DataFrame({"status": ["detected", "interpolated", "absorbed", None], "class": "player"})
    assert list(gs.status_weights(df)) == [1.0, 0.5, 0.5, 1.0]  # None = old row = detected
    assert list(gs.status_weights(df, {"absorbed": 0.0})) == [1.0, 0.5, 0.0, 1.0]
    assert len(gs.observed(df)) == 2
    old = pd.DataFrame({"frame": [1, 2]})
    assert len(gs.observed(old)) == 2 and list(gs.status_weights(old)) == [1.0, 1.0]


def test_fill_rows_project_to_foot_pixels():
    seen = [f for f in range(200) if not 50 <= f < 60]
    df = rows(1, seen, (10.0, 10.0))
    out, _ = label(df, [1], to_image=lambda fr, xy: xy * 10)
    fill = out[out["status"] == "interpolated"]
    assert np.allclose(fill["foot_x"], 100.0) and fill["x1"].isna().all()


def test_absorbed_lookup():
    gaps = pd.DataFrame({"track_id": [1, 2], "lo": [10, 50], "hi": [20, 60], "kind": ["absorbed", "interpolated"]})
    is_abs = gs.absorbed_lookup(gaps)
    assert is_abs(1, 15) and not is_abs(1, 21) and not is_abs(2, 55)


@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
def test_weighted_heatmap_counts(w, tmp_path):
    import compute_positions as cp
    xy = np.array([[10.0, 10.0], [50.0, 30.0]])
    hist = cp.position_hist(xy, (0, 105, 0, 68), (105, 68), weights=np.array([1.0, w]))
    assert hist.sum() == pytest.approx(1.0 + w)
