"""Fill short "lost nearby" gaps in identity tracks with a Kalman (RTS) smoother
in pitch space (issue #47, item 4).

A gap is a stretch of frames in which an identity (track_id 1-11 / 101-111)
has no detection, between two frames where it has one. Gaps up to
--max-gap-s are filled; longer ones stay empty ("unknown"). The fill:

  1. takes the identity's own detections within --context-s on either side
     of the gap (the gap's neighbourhood, nothing else),
  2. runs a constant-velocity Kalman filter forward over every frame of that
     window and a Rauch-Tung-Striebel smoother backward, so the path leaves
     the last sighting with the player's velocity there and arrives at the
     next sighting with the velocity there (a straight line has a velocity
     jump at both ends and cuts corners on a curved run),
  3. checks the result is something a footballer can do: peak speed
     <= --vmax and peak acceleration <= --amax. If the smoothed curve
     breaks that, the straight line is used instead; if even the straight
     line needs more than --vmax on average, the gap is left empty (it is a
     tracking jump, not a player who was hidden for a moment).

Gaps that look like duels are skipped: if at the last sighting another
identity stood within --duel-m (1.5 m), the player was most likely absorbed
into that player's box. Those belong to the duel filler (status "absorbed"),
not to this one, so the two never write the same rows.

Filled rows get interp=True, status="interpolated", conf=0, kit="unknown",
a foot point projected back through that frame's homography (when given) and
a box of the neighbouring detections' size standing on that foot point.

Hook: tools/track_identities.py calls fill_tracks() before writing
tracks.parquet (--interp-max-s, 0 = off). This CLI re-applies the fill to an
existing tracks.parquet (previous fill rows are dropped first), and with
--holdout measures how well it works: well-observed stretches are blanked
artificially and refilled, the error against the hidden detections is
reported per gap length for the straight line (baseline), this smoother and
a cubic Hermite spline.

Usage:
    python tools/interp_gaps.py --match-id <id>                 # fill tracks.parquet in place
    python tools/interp_gaps.py --match-id <id> --holdout       # error table, writes nothing to tracks
"""
import argparse
import time

import numpy as np
import pandas as pd

from _common import match_dir, load_meta, load_calibration, write_json

IDENTITY_MAX = 1000  # identities are 1-11 / 101-111; unlinked fragments are 10000+

DEFAULTS = dict(max_gap_s=3.0, context_s=1.0, q=2.0, r=0.35, vmax=10.0, amax=8.0, duel_m=1.5)


# --------------------------------------------------------------------------
# Smoothers
# --------------------------------------------------------------------------
def rts_smooth(t_obs, xy_obs, t_query, fps, q=DEFAULTS["q"], r=DEFAULTS["r"]):
    """Constant-velocity Kalman filter + RTS smoother on a frame grid.

    t_obs: observation frames (sorted ints), xy_obs: (n, 2) metres.
    t_query: frames to return. q: white-noise acceleration spectral density
    (m^2/s^3), r: measurement noise std (m). x and y share the dynamics, so
    one scalar covariance recursion serves both axes."""
    t_obs = np.asarray(t_obs, dtype=np.int64)
    xy_obs = np.asarray(xy_obs, dtype=float)
    f0, f1 = int(min(t_obs[0], np.min(t_query))), int(max(t_obs[-1], np.max(t_query)))
    n = f1 - f0 + 1
    meas = np.full((n, 2), np.nan)
    meas[t_obs - f0] = xy_obs
    has = ~np.isnan(meas[:, 0])
    dt = 1.0 / fps
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    R = r * r
    # state per axis: [p, v]; stacked as (2 states, 2 axes)
    xs_f = np.zeros((n, 2, 2))
    Ps_f = np.zeros((n, 2, 2))
    xs_p = np.zeros((n, 2, 2))
    Ps_p = np.zeros((n, 2, 2))
    first = int(np.flatnonzero(has)[0])
    # initial velocity from the first two sightings in the window (else 0)
    obs_i = np.flatnonzero(has)
    v0 = np.zeros(2)
    if len(obs_i) >= 2 and obs_i[1] - obs_i[0] <= fps:
        v0 = (meas[obs_i[1]] - meas[obs_i[0]]) / ((obs_i[1] - obs_i[0]) * dt)
    x = np.array([meas[first], v0])
    P = np.diag([R, 4.0])
    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xs_p[k], Ps_p[k] = x, P
        if has[k] and k >= first:
            S = P[0, 0] + R
            K = P[:, 0] / S
            x = x + np.outer(K, meas[k] - x[0])
            P = P - np.outer(K, P[0])
        xs_f[k], Ps_f[k] = x, P
    xs = xs_f.copy()
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ F.T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
    idx = np.asarray(t_query, dtype=np.int64) - f0
    return xs[idx, 0, :], xs[idx, 1, :]


def linear_fill(fa, pa, fb, pb, t_query):
    """Straight line from the last sighting to the next (the baseline: the
    pitch-space fill cut_phase_clips.fill_gaps / the Lines tab use)."""
    u = (np.asarray(t_query, float) - fa) / (fb - fa)
    return np.asarray(pa)[None, :] + (np.asarray(pb) - np.asarray(pa))[None, :] * u[:, None]


def _end_velocity(t, xy, fps, at_end):
    """Least-squares velocity over the context sightings next to the gap."""
    if len(t) < 2:
        return np.zeros(2)
    tt = (np.asarray(t, float) - (t[-1] if at_end else t[0])) / fps
    A = np.c_[np.ones_like(tt), tt]
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return coef[1]


def hermite_fill(t_before, xy_before, t_after, xy_after, t_query, fps):
    """Cubic Hermite spline between the two sightings bounding the gap, with
    end velocities fitted on the context either side."""
    fa, fb = t_before[-1], t_after[0]
    pa, pb = xy_before[-1], xy_after[0]
    va = _end_velocity(t_before, xy_before, fps, at_end=True)
    vb = _end_velocity(t_after, xy_after, fps, at_end=False)
    T = (fb - fa) / fps
    s = (np.asarray(t_query, float) - fa) / (fb - fa)
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return (h00[:, None] * pa + h10[:, None] * T * va + h01[:, None] * pb + h11[:, None] * T * vb)


# --------------------------------------------------------------------------
# One gap
# --------------------------------------------------------------------------
def physically_ok(path, fps, vmax, amax):
    """path: (n, 2) positions on consecutive frames, sightings at both ends."""
    if len(path) < 3:
        return True
    v = np.diff(path, axis=0) * fps
    sp = np.hypot(v[:, 0], v[:, 1])
    # acceleration over a 0.2 s baseline - frame-to-frame differences of a
    # path would amplify rounding, and no player changes velocity faster
    k = max(1, int(round(0.2 * fps)))
    if len(v) > k:
        a = np.hypot(*(v[k:] - v[:-k]).T) * fps / k
        amx = float(a.max())
    else:
        amx = 0.0
    return float(sp.max()) <= vmax and amx <= amax


def fill_gap(fr, xy, i, fps, method="rts", context_s=DEFAULTS["context_s"], q=DEFAULTS["q"],
             r=DEFAULTS["r"], vmax=DEFAULTS["vmax"], amax=DEFAULTS["amax"]):
    """Fill the gap between sightings fr[i] and fr[i+1] of one identity.
    Returns (frames, (n, 2) positions, how) with how in "smooth" | "linear" |
    "implausible" (then positions is None)."""
    fa, fb = int(fr[i]), int(fr[i + 1])
    pa, pb = xy[i], xy[i + 1]
    gap_f = np.arange(fa + 1, fb)
    T = (fb - fa) / fps
    if np.hypot(*(pb - pa)) / T > vmax:
        return gap_f, None, "implausible"
    lin = linear_fill(fa, pa, fb, pb, gap_f)
    if method == "linear":
        return gap_f, lin, "linear"
    c = int(round(context_s * fps))
    lo = np.searchsorted(fr, fa - c, side="left")
    hi = np.searchsorted(fr, fb + c, side="right")
    if method == "hermite":
        path = hermite_fill(fr[lo:i + 1], xy[lo:i + 1], fr[i + 1:hi], xy[i + 1:hi], gap_f, fps)
    else:
        sm, _ = rts_smooth(fr[lo:hi], xy[lo:hi], np.r_[fa, gap_f, fb], fps, q=q, r=r)
        # The smoother doesn't pass exactly through the (noisy) sightings;
        # shift its curve so it joins them, else the fill would jump at both ends.
        u = ((gap_f - fa) / (fb - fa))[:, None]
        path = sm[1:-1] + (1 - u) * (pa - sm[0]) + u * (pb - sm[-1])
    full = np.vstack([pa, path, pb])
    if not np.isfinite(path).all() or not physically_ok(full, fps, vmax, amax):
        return gap_f, lin, "linear"
    return gap_f, path, "smooth"


def make_filler(max_gap_s=DEFAULTS["max_gap_s"], method="rts", **kw):
    """Adapter for gap_status.py (issue-47-status branch): returns
    fill(gap, ctx) -> (gap.n_missing, 2) pitch xy for frames gap.lo..gap.hi,
    or None to leave the gap unknown. Plug in as
    label_tracks(..., fillers={"interpolated": make_filler()}).
    Duck-typed (no import of gap_status): gap needs track_id/start/end/lo/hi,
    ctx needs track(tid) -> (frames, xy) and fps."""
    params = {k: kw.get(k, DEFAULTS[k]) for k in ("context_s", "q", "r", "vmax", "amax")}

    def fill(gap, ctx):
        if gap.start is None or gap.end is None:
            return None
        fps = ctx.fps
        if (gap.end - gap.start - 1) > round(max_gap_s * fps):
            return None
        fr, xy = ctx.track(gap.track_id)
        fr, xy = np.asarray(fr), np.asarray(xy, dtype=float)
        i = int(np.searchsorted(fr, gap.start))
        if i + 1 >= len(fr) or fr[i] != gap.start or fr[i + 1] != gap.end:
            return None
        _, path, _ = fill_gap(fr, xy, i, fps, method=method, **params)
        return path

    return fill


# --------------------------------------------------------------------------
# Whole tracks
# --------------------------------------------------------------------------
def _frame_index(players):
    """frame-sorted arrays for 'who else was where at frame f'."""
    s = players.sort_values("frame")
    return (s["frame"].to_numpy(), s["track_id"].to_numpy(), s["pitch_x"].to_numpy(), s["pitch_y"].to_numpy())


def near_other(index, f, tid, x, y, radius):
    """Distance from (x, y) to the nearest OTHER identity at frame f (inf if none)."""
    frames, tids, xs, ys = index
    lo, hi = np.searchsorted(frames, f), np.searchsorted(frames, f, side="right")
    m = tids[lo:hi] != tid
    if not m.any():
        return np.inf
    return float(np.hypot(xs[lo:hi][m] - x, ys[lo:hi][m] - y).min())


def strip_fill(tracks):
    """Drop rows a previous run of this filler added (player rows with interp=True)."""
    if "interp" not in tracks.columns:
        return tracks
    fill = (tracks["class"] == "player") & tracks["interp"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return tracks[~fill]


def fill_tracks(tracks, fps, world_to_img=None, max_gap_s=DEFAULTS["max_gap_s"], method="rts",
                skip_duels=True, **kw):
    """tracks.parquet frame -> (frame with fill rows appended, stats dict).

    world_to_img(frame) -> 3x3 pitch->image homography or None; used to give
    fill rows a foot point and box for the video renders."""
    t0 = time.time()
    duel_m = kw.pop("duel_m", DEFAULTS["duel_m"])
    tracks = strip_fill(tracks)
    pl = tracks[(tracks["class"] == "player") & (tracks["track_id"] > 0)
                & (tracks["track_id"] < IDENTITY_MAX) & tracks["pitch_x"].notna()]
    index = _frame_index(pl)
    max_f = int(round(max_gap_s * fps))
    new, stats = [], {"gaps": 0, "filled": 0, "smooth": 0, "linear_fallback": 0, "too_long": 0,
                      "implausible": 0, "duel_skipped": 0, "frames_filled": 0}
    for tid, g in pl.groupby("track_id"):
        g = g.sort_values("frame").drop_duplicates("frame")
        fr = g["frame"].to_numpy()
        xy = g[["pitch_x", "pitch_y"]].to_numpy()
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy() if {"x1", "y2"} <= set(g.columns) else None
        t_s = g["t_s"].to_numpy() if "t_s" in g.columns else fr / fps
        for i in np.flatnonzero(np.diff(fr) > 1):
            n_missing = int(fr[i + 1] - fr[i] - 1)
            stats["gaps"] += 1
            if n_missing > max_f:
                stats["too_long"] += 1
                continue
            if skip_duels and near_other(index, fr[i], tid, *xy[i], duel_m) <= duel_m:
                stats["duel_skipped"] += 1
                continue
            gap_f, path, how = fill_gap(fr, xy, i, fps, method=method, **kw)
            if path is None:
                stats["implausible"] += 1
                continue
            stats["filled"] += 1
            stats["smooth" if how == "smooth" else "linear_fallback"] += 1
            stats["frames_filled"] += len(gap_f)
            u = (gap_f - fr[i]) / (fr[i + 1] - fr[i])
            ts = t_s[i] + (t_s[i + 1] - t_s[i]) * u
            if boxes is not None:
                wh = (boxes[i, 2:] - boxes[i, :2]) * (1 - u[:, None]) + (boxes[i + 1, 2:] - boxes[i + 1, :2]) * u[:, None]
            else:
                wh = np.full((len(gap_f), 2), np.nan)
            for k, f in enumerate(gap_f):
                fx = fy = np.nan
                H = world_to_img(int(f)) if world_to_img else None
                if H is not None:
                    pt = H @ np.array([path[k, 0], path[k, 1], 1.0])
                    if abs(pt[2]) > 1e-9:
                        fx, fy = pt[0] / pt[2], pt[1] / pt[2]
                new.append((int(f), float(ts[k]), int(tid), float(path[k, 0]), float(path[k, 1]),
                            fx, fy, fx - wh[k, 0] / 2, fy - wh[k, 1], fx + wh[k, 0] / 2, fy))
    fill = pd.DataFrame(new, columns=["frame", "t_s", "track_id", "pitch_x", "pitch_y", "foot_x", "foot_y",
                                      "x1", "y1", "x2", "y2"])
    fill["class"], fill["conf"], fill["kit"] = "player", 0.0, "unknown"
    fill["interp"], fill["status"] = True, "interpolated"
    out = tracks.copy()
    if "interp" not in out.columns:
        out["interp"] = None
    out["interp"] = out["interp"].astype(object)
    if len(fill):
        out = pd.concat([out, fill.reindex(columns=[c for c in out.columns] + (["status"] if "status" not in out.columns else []))],
                        ignore_index=True)
    sort = ["frame", "class", "track_id"]
    out = out.sort_values(sort).reset_index(drop=True)
    stats["runtime_s"] = round(time.time() - t0, 1)
    stats["max_gap_s"] = max_gap_s
    return out, stats


def gap_table(tracks, fps, f_lo=None, f_hi=None, duel_m=DEFAULTS["duel_m"]):
    """Missing identity-frames bucketed by gap length, plus duel-gap count,
    counting interpolated rows as present. Window edges count as gaps."""
    present = tracks[(tracks["class"] == "player") & (tracks["track_id"] > 0)
                     & (tracks["track_id"] < IDENTITY_MAX) & tracks["pitch_x"].notna()]
    f_lo = int(present["frame"].min()) if f_lo is None else f_lo
    f_hi = int(present["frame"].max()) if f_hi is None else f_hi
    real = strip_fill(present)
    index = _frame_index(real)
    buckets = {"<1s": 0, "1-3s": 0, "3-10s": 0, ">10s": 0}
    n_gaps, duel = 0, 0
    for tid, g in present.groupby("track_id"):
        fr = np.unique(g["frame"].to_numpy())
        edges = np.r_[f_lo - 1, fr, f_hi + 1]
        lens = np.diff(edges) - 1
        gp = g.drop_duplicates("frame").set_index("frame")
        for j in np.flatnonzero(lens > 0):
            n = int(lens[j])
            s = n / fps
            buckets["<1s" if s < 1 else "1-3s" if s < 3 else "3-10s" if s < 10 else ">10s"] += n
            n_gaps += 1
            last = int(edges[j])
            if last in gp.index:
                r = gp.loc[last]
                if near_other(index, last, tid, r["pitch_x"], r["pitch_y"], duel_m) <= duel_m:
                    duel += 1
    return {"missing_frames": buckets, "missing_total": int(sum(buckets.values())), "gaps": n_gaps,
            "duel_gaps": duel}


# --------------------------------------------------------------------------
# Hold-out evaluation
# --------------------------------------------------------------------------
def holdout(tracks, fps, gap_lengths_s=(0.5, 1, 2, 3, 5, 8, 10), n_per=600, seed=0, methods=("linear", "rts", "hermite"),
            params=None, frame_filter=None):
    """Blank fully observed stretches of each length and refill them.
    Returns {gap_s: {method: errors (m, per blanked frame, mean over the gap)}}.

    A sample is any stretch of gap_s seconds an identity was seen on EVERY
    frame (plus the frame either side). The filler then sees the identity's
    real sightings around it - including any real gaps nearby - just as a
    real gap would. Truth = the hidden detections (themselves ~0.3 m noisy),
    so even a perfect filler scores above zero."""
    params = dict(DEFAULTS, **(params or {}))
    kw = {k: params[k] for k in ("context_s", "q", "r", "vmax", "amax")}
    rng = np.random.default_rng(seed)
    pl = strip_fill(tracks)
    pl = pl[(pl["class"] == "player") & (pl["track_id"] > 0) & (pl["track_id"] < IDENTITY_MAX)
            & pl["pitch_x"].notna()]
    per_id = {}
    for tid, g in pl.groupby("track_id"):
        g = g.sort_values("frame").drop_duplicates("frame")
        per_id[tid] = (g["frame"].to_numpy(), g[["pitch_x", "pitch_y"]].to_numpy())
    out = {}
    for gs in gap_lengths_s:
        G = int(round(gs * fps))
        cands = []
        for tid, (fr, xy) in per_id.items():
            # start index s: frames fr[s-1 .. s+G] consecutive -> fr[s+G]-fr[s-1] == G+1
            if len(fr) < G + 2:
                continue
            ok = np.flatnonzero(fr[G + 1:] - fr[:-(G + 1)] == G + 1)  # ok[j]: fr[j..j+G+1] consecutive
            if frame_filter is not None:
                ok = ok[frame_filter(fr[ok])]
            cands.extend((tid, j) for j in ok[:: max(1, G // 2)])
        if not cands:
            continue
        pick = rng.choice(len(cands), size=min(n_per, len(cands)), replace=False)
        res = {m: [] for m in methods}
        res["fallback"] = {m: 0 for m in methods}
        for c in pick:
            tid, j = cands[c]
            fr, xy = per_id[tid]
            keep = np.ones(len(fr), bool)
            keep[j + 1:j + G + 1] = False
            truth = xy[j + 1:j + G + 1]
            fk, xk = fr[keep], xy[keep]
            for m in methods:
                gap_f, path, how = fill_gap(fk, xk, j, fps, method=m, **kw)
                if path is None:  # implausible jump - can't happen on a fully observed stretch often
                    path = linear_fill(fk[j], xk[j], fk[j + 1], xk[j + 1], gap_f)
                if m != "linear" and how != "smooth":
                    res["fallback"][m] += 1
                res[m].append(float(np.hypot(*(path - truth).T).mean()))
                res.setdefault(m + "_max", []).append(float(np.hypot(*(path - truth).T).max()))
        out[gs] = {k: (np.array(v) if isinstance(v, list) else v) for k, v in res.items()}
    return out


def summarize_holdout(res, methods=("linear", "rts", "hermite")):
    rows = []
    for gs, r in res.items():
        row = {"gap_s": gs, "n": len(r[methods[0]])}
        for m in methods:
            e = r[m]
            row[f"{m}_mean"] = round(float(e.mean()), 2)
            row[f"{m}_p95"] = round(float(np.percentile(e, 95)), 2)
            if m != "linear":
                row[f"{m}_fallback"] = int(r["fallback"][m])
        rows.append(row)
    return pd.DataFrame(rows)


def world_to_img_from_calib(calib):
    """frame -> pitch->image homography, from a dynamic calibration."""
    if calib is None or "frame_H" not in calib:
        return None
    w2i = {int(f): np.linalg.inv(H) for f, H in zip(calib["frame_index"], calib["frame_H"])}
    return w2i.get


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--max-gap-s", type=float, default=DEFAULTS["max_gap_s"],
                    help="Longest gap filled; longer gaps stay unknown")
    ap.add_argument("--method", default="rts", choices=["rts", "hermite", "linear"])
    ap.add_argument("--context-s", type=float, default=DEFAULTS["context_s"],
                    help="Sightings used either side of a gap")
    ap.add_argument("--q", type=float, default=DEFAULTS["q"], help="Kalman acceleration noise, m^2/s^3")
    ap.add_argument("--r", type=float, default=DEFAULTS["r"], help="Detection noise std, m")
    ap.add_argument("--vmax", type=float, default=DEFAULTS["vmax"], help="Peak speed cap, m/s")
    ap.add_argument("--amax", type=float, default=DEFAULTS["amax"], help="Peak acceleration cap, m/s^2")
    ap.add_argument("--duel-m", type=float, default=DEFAULTS["duel_m"],
                    help="Skip gaps whose last sighting was this close to another identity (duel filler's job)")
    ap.add_argument("--fill-duels", action="store_true", help="Fill duel-like gaps too")
    ap.add_argument("--holdout", action="store_true", help="Hold-out error table instead of filling")
    ap.add_argument("--holdout-n", type=int, default=600, help="Blanked stretches per gap length")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()

    meta = load_meta(a.match_id, a.tmp_dir)
    fps = meta["fps"]
    mdir = match_dir(a.match_id, a.tmp_dir)
    tracks = pd.read_parquet(mdir / "tracks.parquet")
    params = dict(context_s=a.context_s, q=a.q, r=a.r, vmax=a.vmax, amax=a.amax)
    if a.holdout:
        res = holdout(tracks, fps, n_per=a.holdout_n, params=params)
        table = summarize_holdout(res)
        print(table.to_string(index=False))
        write_json(mdir / "interp_holdout.json", {"params": params, "table": table.to_dict(orient="records")})
        return
    calib = load_calibration(a.match_id, a.config_dir, a.calibration)
    out, stats = fill_tracks(tracks, fps, world_to_img_from_calib(calib), max_gap_s=a.max_gap_s,
                             method=a.method, skip_duels=not a.fill_duels, duel_m=a.duel_m, **params)
    out.to_parquet(mdir / "tracks.parquet", index=False)
    write_json(mdir / "interp_gaps.json", {"params": dict(params, max_gap_s=a.max_gap_s, method=a.method,
                                                         duel_m=a.duel_m, skip_duels=not a.fill_duels),
                                            "stats": stats})
    print(f"[{a.match_id}] {stats['gaps']} identity gaps: filled {stats['filled']} "
          f"({stats['frames_filled']} frames; {stats['linear_fallback']} fell back to a straight line), "
          f"{stats['too_long']} longer than {a.max_gap_s} s, {stats['duel_skipped']} duel-like (skipped), "
          f"{stats['implausible']} physically implausible - {stats['runtime_s']} s")


if __name__ == "__main__":
    main()
