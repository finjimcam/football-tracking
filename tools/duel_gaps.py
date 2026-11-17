"""Fill duel (absorbed) gaps in identity tracks -> tracks.parquet rows with
interp=True, absorbed_by=<partner track_id>; review list in duel_gaps.json.

Issue #47, item 3. In a duel two players' boxes merge, the detector keeps one
box, and the hidden player's identity goes missing while he is visibly moving
WITH the other one. A straight line across that gap ignores the duel; the
surviving track already knows where the pair went. So:

  A gap in identity k's track (last seen at frame f0, seen again at f1) is
  ABSORBED when some other tracked person P was within --entry-m of k at f0,
  P is tracked through the gap (>= --min-cover of the gap frames, one
  tracklet - not a chain re-linked mid-gap), and k re-emerges within
  --exit-m of P at f1.
  Fill: pos_k(f) = pos_P(f) + offset(f), the offset interpolated linearly
  from the entry offset (k - P at f0) to the exit offset (k - P at f1).

This is the merge/split framing: every fill is recorded as a group
"k+P together from f0 to f1" (duel_gaps.json `groups`, and the absorbed_by
column on the filled rows). At the split we check the identity before
trusting the fill:
  - kit: k's detections in the first second after f1 must read as k's team
    (p_team_a), and an identified P's as P's team. Opponents can't swap
    teams, so a cross-team duel with consistent kits is safe.
  - motion: velocities over --motion-s before f0 and after f1; if
    "k in ~ P out and P in ~ k out" fits better than the straight pairing by
    more than --motion-margin m/s, the two may have swapped at the split.
  - same-team duels must also keep their side: an offset that flips sign
    (k left of P at entry, right at exit) while both offsets are clear of
    noise could be a swap the kit can't see.
A fill that fails a check is AMBIGUOUS: it is listed for review and not
written (unless --keep-ambiguous), so it never feeds stats as if it were known.

Marking: written rows carry interp=True (the column track_ball.py already uses
for interpolated ball rows), conf 0, tracklet NaN, p_team_a NaN, and
absorbed_by = partner track_id. When the status column (issue #47 item 2)
lands: interp & absorbed_by.notna() -> 'absorbed'; ambiguous fills ->
'unknown'. Re-runs first drop rows written by a previous run (idempotent).

Run after track_identities.py (before or after track_ball.py, which keeps
player rows as they are), then re-run the downstream stages.

Usage:
    python tools/duel_gaps.py --match-id <id>
    python tools/duel_gaps.py --match-id <id> --dry-run   # classify + report only
"""
import argparse
import time

import numpy as np
import pandas as pd

from _common import load_calibration, load_meta, load_player_map, match_dir, write_json

IDENTITIES = list(range(1, 12)) + list(range(101, 112))


def team_of(tid):
    tid = int(tid)
    return "team_a" if 1 <= tid <= 11 else "team_b" if 101 <= tid <= 111 else None


def _is_filled(df):
    """Rows written by this tool (or any other player interpolation)."""
    if "absorbed_by" in df.columns:
        return df["absorbed_by"].notna().to_numpy()
    return np.zeros(len(df), bool)


class TrackIndex:
    """Fast per-track and per-frame lookups over the player rows."""

    def __init__(self, players):
        p = players.sort_values(["track_id", "frame"])
        self.tid = p["track_id"].to_numpy()
        self.frame = p["frame"].to_numpy()
        self.xy = p[["pitch_x", "pitch_y"]].to_numpy(float)
        self.pa = p["p_team_a"].to_numpy(float) if "p_team_a" in p else np.full(len(p), np.nan)
        self.kit = p["kit"].to_numpy(object) if "kit" in p else np.full(len(p), None, object)
        self.box = p[["x1", "y1", "x2", "y2"]].to_numpy(float) if "x1" in p else None
        self.tracklet = p["tracklet"].to_numpy(float) if "tracklet" in p else np.full(len(p), np.nan)
        bounds = np.flatnonzero(np.diff(np.r_[-1, self.tid, -2]))
        self.span = {int(self.tid[s]): (s, e) for s, e in zip(bounds[:-1], bounds[1:])}
        order = np.argsort(self.frame, kind="stable")
        fr = self.frame[order]
        fb = np.flatnonzero(np.diff(np.r_[fr[0] - 1, fr, fr[-1] + 1])) if len(fr) else np.array([0])
        self.by_frame = {int(fr[s]): order[s:e] for s, e in zip(fb[:-1], fb[1:])}

    def rows(self, tid, f0=None, f1=None):
        s, e = self.span.get(int(tid), (0, 0))
        if f0 is None:
            return np.arange(s, e)
        fr = self.frame[s:e]
        return np.arange(s + np.searchsorted(fr, f0), s + np.searchsorted(fr, f1, side="right"))

    def at(self, tid, f):
        r = self.rows(tid, f, f)
        return r[0] if len(r) else None


def find_gaps(idx, ids, lo, hi):
    """[(k, f0, f1)] interior gaps of each identity (f0 last seen, f1 seen again, f1 - f0 > 1)."""
    out = []
    for k in ids:
        r = idx.rows(k)
        fr = idx.frame[r]
        fr = fr[(fr >= lo) & (fr <= hi)]
        for i in np.flatnonzero(np.diff(fr) > 1):
            out.append((int(k), int(fr[i]), int(fr[i + 1])))
    return out


def neighbours(idx, k, f, radius, ignore=()):
    """[(tid, dist)] other tracks within radius of k at frame f, nearest first."""
    rk = idx.at(k, f)
    rows = idx.by_frame.get(int(f))
    if rk is None or rows is None:
        return []
    d = np.hypot(*(idx.xy[rows] - idx.xy[rk]).T)
    out = [(int(idx.tid[r]), float(dd)) for r, dd in zip(rows, d)
           if dd <= radius and idx.tid[r] != k and int(idx.tid[r]) not in ignore]
    return sorted(out, key=lambda x: x[1])


def _velocity(idx, tid, f0, f1):
    r = idx.rows(tid, f0, f1)
    if len(r) < 3:
        return None
    dt = idx.frame[r[-1]] - idx.frame[r[0]]
    return (idx.xy[r[-1]] - idx.xy[r[0]]) / dt if dt > 0 else None


def _kit_after(idx, tid, other, f0, f1, clear_m=1.2):
    """Mean p_team_a of tid's readable detections in [f0, f1] taken while it
    stands clear of `other` (overlapping boxes mix the two kits); None if < 3."""
    r = idx.rows(tid, f0, f1)
    ro = idx.rows(other, f0 - 3, f1 + 3)
    if len(r) and len(ro):
        ox = np.interp(idx.frame[r], idx.frame[ro], idx.xy[ro, 0])
        oy = np.interp(idx.frame[r], idx.frame[ro], idx.xy[ro, 1])
        r = r[np.hypot(idx.xy[r, 0] - ox, idx.xy[r, 1] - oy) > clear_m]
    v = idx.pa[r]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) >= 3 else None


def _team_of_track(idx, tid, f0, f1):
    """Team of a partner: its identity range, else the majority kit label in the window."""
    t = team_of(tid)
    if t:
        return t
    kits = [k for k in idx.kit[idx.rows(tid, f0, f1)] if k in ("team_a", "team_b", "gk_a", "gk_b")]
    if not kits:
        return None
    top = max(set(kits), key=kits.count)
    return {"gk_a": "team_a", "gk_b": "team_b"}.get(top, top)


def partner_path(p_frames, p_xy, f0, f1, k0, k1):
    """k's pitch xy for frames f0+1..f1-1: the partner's path (its own holes
    interpolated) plus k's offset from it, blended linearly from the entry
    offset (k0 - P at f0) to the exit offset (k1 - P at f1)."""
    fr = np.arange(f0 + 1, f1)
    p = np.c_[np.interp(fr, p_frames, p_xy[:, 0]), np.interp(fr, p_frames, p_xy[:, 1])]
    p0 = np.array([np.interp(f0, p_frames, p_xy[:, j]) for j in (0, 1)])
    p1 = np.array([np.interp(f1, p_frames, p_xy[:, j]) for j in (0, 1)])
    u = ((fr - f0) / (f1 - f0))[:, None]
    return p + (np.asarray(k0) - p0) + ((np.asarray(k1) - p1) - (np.asarray(k0) - p0)) * u


def classify_gap(idx, k, f0, f1, fps, entry_m=1.5, exit_m=2.5, min_cover=0.8, max_gap_s=10.0, end_tol=3,
                 motion_s=0.5, motion_margin=1.5, flip_m=0.7, kit_s=2.0, ignore=()):
    """One identity gap -> dict with kind 'duel' (entry near another track) or
    'other'; duel gaps get verdict 'filled' | 'ambiguous' | 'no_partner'
    (nobody tracked through the gap who k re-emerges next to) and, when a
    partner is found, the fill path."""
    n = f1 - f0 - 1
    rec = {"identity": k, "start": f0 + 1, "end": f1 - 1, "frames": n, "gap_s": round(n / fps, 2)}
    near = neighbours(idx, k, f0, entry_m, ignore)
    if not near:
        rec["kind"] = "other"
        return rec
    rec["kind"] = "duel"
    rec["entry_partners"] = [t for t, _ in near]
    if n / fps > max_gap_s:
        rec["verdict"], rec["why"] = "no_partner", "too long"
        return rec
    rk0, rk1 = idx.at(k, f0), idx.at(k, f1)
    best, why_not = None, "nobody near k at the start is tracked through the gap"
    for P, d_in in near:
        r = idx.rows(P, f0, f1 + end_tol)
        if len(r) == 0 or idx.frame[r[-1]] < f1 - end_tol:
            continue  # P must be seen at (or within end_tol frames of) the split
        tls = idx.tracklet[r[idx.frame[r] <= f1]]
        if len(np.unique(tls[np.isfinite(tls)])) > 1:
            why_not = "partner's own track is re-linked inside the gap"
            continue  # its path there is itself a stitched guess (e.g. it jumped onto a referee)
        inside = int(((idx.frame[r] > f0) & (idx.frame[r] < f1)).sum())
        if n > 0 and inside / n < min_cover:
            continue
        p1 = np.array([np.interp(f1, idx.frame[r], idx.xy[r, j]) for j in (0, 1)])
        d_out = float(np.hypot(*(idx.xy[rk1] - p1)))
        if d_out > exit_m:
            why_not = "k re-emerges away from the partner"
            continue
        if best is None or d_in + d_out < best[1] + best[2]:
            best = (P, d_in, d_out, r, p1)
    if best is None:
        rec["verdict"], rec["why"] = "no_partner", why_not
        return rec
    P, d_in, d_out, r, p1 = best
    rec.update(partner=P, entry_m=round(d_in, 2), exit_m=round(d_out, 2))
    off0, off1 = idx.xy[rk0] - idx.xy[r[0]], idx.xy[rk1] - p1

    # identity checks at the split
    why = []
    tk, tp = team_of(k), _team_of_track(idx, P, f0 - int(2 * fps), f1 + int(2 * fps))
    rec["partner_team"] = tp
    ks = int(kit_s * fps)
    keepers = (1, 101)  # keeper kits aren't in the outfield colour model
    pa_k = _kit_after(idx, k, P, f1, f1 + ks) if k not in keepers else None
    if pa_k is not None and tk and ((pa_k > 0.5) != (tk == "team_a")):
        why.append("k's kit after the split reads as the other team")
    if team_of(P) and P not in keepers:
        pa_p = _kit_after(idx, P, k, f1, f1 + ks)
        if pa_p is not None and ((pa_p > 0.5) != (team_of(P) == "team_a")):
            why.append("partner's kit after the split reads as the other team")
    same_team = tp is None or tp == tk
    m = max(int(motion_s * fps), 2)
    vk_in, vp_in = _velocity(idx, k, f0 - m, f0), _velocity(idx, P, f0 - m, f0)
    vk_out, vp_out = _velocity(idx, k, f1, f1 + m), _velocity(idx, P, f1, f1 + m)
    if same_team and all(v is not None for v in (vk_in, vp_in, vk_out, vp_out)):
        vk_in, vp_in, vk_out, vp_out = (v * fps for v in (vk_in, vp_in, vk_out, vp_out))
        straight = np.linalg.norm(vk_in - vk_out) + np.linalg.norm(vp_in - vp_out)
        crossed = np.linalg.norm(vk_in - vp_out) + np.linalg.norm(vp_in - vk_out)
        rec["motion_straight"], rec["motion_crossed"] = round(float(straight), 2), round(float(crossed), 2)
        if crossed + motion_margin < straight:
            why.append("exit motion fits a swap better")
    if same_team and min(np.linalg.norm(off0), np.linalg.norm(off1)) > flip_m and off0 @ off1 < 0:
        why.append("players changed sides during the duel")
    if same_team:
        rec["same_team"] = True
    rec["verdict"] = "ambiguous" if why else "filled"
    if why:
        rec["why"] = "; ".join(why)

    fr = np.arange(f0 + 1, f1)
    xy = partner_path(idx.frame[r], idx.xy[r], f0, f1, idx.xy[rk0], idx.xy[rk1])
    rec["_fill"] = (fr, xy[:, 0], xy[:, 1],
                    np.interp(fr, idx.frame[r], idx.box[r, 3] - idx.box[r, 1]) if idx.box is not None else None,
                    np.interp(fr, idx.frame[r], idx.box[r, 2] - idx.box[r, 0]) if idx.box is not None else None)
    return rec


def fill(gap, ctx, min_cover=0.8, exit_m=2.5, end_tol=3, **kw):
    """Plug-in filler for tools/gap_status.py (issue #47 item 2): Gap + GapContext
    -> (gap.n_missing, 2) pitch xy for frames gap.lo..gap.hi, or None (leave
    the gap 'unknown'). Rows it fills map to status 'absorbed'.

    If the context carries a TrackIndex as `ctx.duel_index` (built from the
    same rows, with kit / p_team_a / tracklet columns), the full
    classify_gap() runs, identity checks included, and only a 'filled' verdict
    is returned. Otherwise only geometry is available: partner tracked over
    >= min_cover of the gap and seen within end_tol frames of the split,
    k re-emerging within exit_m of it."""
    if gap.start is None or gap.end is None or not getattr(gap, "duel", False) or gap.partner < 0:
        return None
    idx = getattr(ctx, "duel_index", None)
    if idx is not None:
        rec = classify_gap(idx, gap.track_id, gap.start, gap.end, ctx.fps, min_cover=min_cover, exit_m=exit_m,
                           end_tol=end_tol, **kw)
        if rec.get("verdict") != "filled":
            return None
        _, x, y, _, _ = rec["_fill"]
        return np.c_[x, y]
    fr, xy = ctx.track(gap.partner)
    m = (fr >= gap.start) & (fr <= gap.end + end_tol)
    fr, xy = fr[m], xy[m]
    if not len(fr) or fr[0] != gap.start or fr[-1] < gap.end - end_tol:
        return None
    if ((fr > gap.start) & (fr < gap.end)).sum() < min_cover * gap.n_missing:
        return None
    p1 = np.array([np.interp(gap.end, fr, xy[:, j]) for j in (0, 1)])
    if np.hypot(*(np.asarray(gap.xy_end) - p1)) > exit_m:
        return None
    return partner_path(fr, xy, gap.start, gap.end, gap.xy_start, gap.xy_end)


def classify_gaps(players, fps, lo, hi, ids=IDENTITIES, ignore=(), **kw):
    """All interior gaps of the identities in `ids` -> list of classify_gap records."""
    obs = players[~_is_filled(players)]
    idx = TrackIndex(obs)
    return [classify_gap(idx, k, f0, f1, fps, ignore=ignore, **kw) for k, f0, f1 in find_gaps(idx, ids, lo, hi)]


def fill_rows(recs, fps, H_at=None, keep_ambiguous=False, team_kit=True):
    """Fill records -> DataFrame of tracks.parquet rows (interp=True, absorbed_by=partner)."""
    out = []
    for rec in recs:
        if rec.get("verdict") not in ("filled", "ambiguous") or (rec["verdict"] == "ambiguous"
                                                                 and not keep_ambiguous):
            continue
        fr, x, y, h, w = rec["_fill"]
        fx = np.full(len(fr), np.nan)
        fy = np.full(len(fr), np.nan)
        if H_at is not None:
            for i, f in enumerate(fr):
                H = H_at(int(f))
                if H is None:
                    continue
                q = np.linalg.solve(H, np.array([x[i], y[i], 1.0]))
                if abs(q[2]) > 1e-9:
                    fx[i], fy[i] = q[0] / q[2], q[1] / q[2]
        h = h if h is not None else np.full(len(fr), np.nan)
        w = w if w is not None else np.full(len(fr), np.nan)
        out.append(pd.DataFrame({
            "frame": fr, "t_s": fr / fps, "track_id": rec["identity"], "class": "player",
            "x1": fx - w / 2, "y1": fy - h, "x2": fx + w / 2, "y2": fy, "foot_x": fx, "foot_y": fy,
            "conf": 0.0, "pitch_x": x, "pitch_y": y, "tracklet": np.nan,
            "kit": team_of(rec["identity"]) if team_kit else None, "p_team_a": np.nan,
            "interp": True, "absorbed_by": float(rec["partner"]),
            "absorbed_review": rec["verdict"] == "ambiguous"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarise(recs, fps):
    duel = [r for r in recs if r["kind"] == "duel"]
    by = {v: [r for r in duel if r.get("verdict") == v] for v in ("filled", "ambiguous", "no_partner")}
    return {
        "gaps": len(recs), "duel_gaps": len(duel),
        "filled": len(by["filled"]), "ambiguous": len(by["ambiguous"]), "no_partner": len(by["no_partner"]),
        "filled_frames": int(sum(r["frames"] for r in by["filled"])),
        "ambiguous_frames": int(sum(r["frames"] for r in by["ambiguous"])),
        "filled_cross_team": sum(1 for r in by["filled"] if not r.get("same_team")),
        "ambiguous_reasons": pd.Series([w for r in by["ambiguous"] for w in r["why"].split("; ")])
        .value_counts().to_dict() if by["ambiguous"] else {},
    }


def apply(tracks, fps, lo, hi, H_at=None, ignore=(), keep_ambiguous=False, **kw):
    """tracks.parquet frame -> (new tracks frame, records). Drops rows a previous run wrote."""
    tracks = tracks[~_is_filled(tracks)]
    pl = tracks[(tracks["class"] == "player") & tracks["pitch_x"].notna()]
    recs = classify_gaps(pl, fps, lo, hi, ignore=ignore, **kw)
    add = fill_rows(recs, fps, H_at, keep_ambiguous)
    if "interp" in tracks.columns:
        tracks = tracks.assign(interp=tracks["interp"].astype(object))
    if len(add):
        tracks = pd.concat([tracks, add.reindex(columns=list(tracks.columns) + [c for c in add.columns
                                                                                 if c not in tracks.columns])],
                           ignore_index=True)
    if "absorbed_by" not in tracks.columns:
        tracks["absorbed_by"] = np.nan
    tracks["absorbed_review"] = (tracks["absorbed_review"].fillna(False).astype(bool)
                                 if "absorbed_review" in tracks.columns else False)
    tracks = tracks.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)
    return tracks, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--entry-m", type=float, default=1.5, help="Max distance to the partner when k is lost")
    ap.add_argument("--exit-m", type=float, default=2.5, help="Max distance to the partner when k re-emerges")
    ap.add_argument("--min-cover", type=float, default=0.8, help="Share of the gap the partner must be tracked")
    ap.add_argument("--max-gap-s", type=float, default=10.0, help="Longest duel gap to fill")
    ap.add_argument("--motion-margin", type=float, default=1.5, help="m/s; swap check at the split")
    ap.add_argument("--keep-ambiguous", action="store_true",
                    help="Also write ambiguous fills (absorbed_review=True) instead of only listing them")
    ap.add_argument("--dry-run", action="store_true", help="Classify and write duel_gaps.json only")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()
    t0 = time.time()
    meta = load_meta(a.match_id, a.tmp_dir)
    fps = meta["fps"]
    mdir = match_dir(a.match_id, a.tmp_dir)
    pm = load_player_map(a.match_id, a.tmp_dir) or {}
    win = pm.get("window", {})
    ignore = {int(i["track_id"]) for i in pm.get("ignored_tracks", [])}
    tracks = pd.read_parquet(mdir / "tracks.parquet")
    lo, hi = win.get("start_frame", int(tracks["frame"].min())), win.get("end_frame", int(tracks["frame"].max()))
    calib = load_calibration(a.match_id, a.config_dir)
    H_at = None
    if calib is not None and "frame_H" in calib:
        H_at = dict(zip(calib["frame_index"].tolist(), calib["frame_H"])).get
    elif calib is not None and "homography" in calib:
        H_at = lambda f: calib["homography"]  # noqa: E731
    out, recs = apply(tracks, fps, lo, hi, H_at, ignore, a.keep_ambiguous, entry_m=a.entry_m, exit_m=a.exit_m,
                      min_cover=a.min_cover, max_gap_s=a.max_gap_s, motion_margin=a.motion_margin)
    summ = summarise(recs, fps)
    duel = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if not k.startswith("_")}
            for r in recs if r["kind"] == "duel"]
    groups = [{"members": [r["identity"], r["partner"]], "start": r["start"], "end": r["end"],
               "verdict": r["verdict"]} for r in duel if "partner" in r]
    write_json(mdir / "duel_gaps.json", {"summary": summ, "params": {k: v for k, v in vars(a).items()
                                                                      if k not in ("match_id",)},
                                         "groups": groups, "gaps": duel})
    if not a.dry_run:
        out.to_parquet(mdir / "tracks.parquet", index=False)
    print(f"[{a.match_id}] {summ['gaps']} identity gaps, {summ['duel_gaps']} start in a duel "
          f"(< {a.entry_m} m from another track): {summ['filled']} filled ({summ['filled_frames']} frames), "
          f"{summ['ambiguous']} ambiguous -> review ({summ['ambiguous_frames']} frames), "
          f"{summ['no_partner']} without a partner through the gap")
    if summ["ambiguous_reasons"]:
        print("  ambiguous because: " + ", ".join(f"{k} {v}" for k, v in summ["ambiguous_reasons"].items()))
    print(f"  {'(dry run) ' if a.dry_run else ''}wrote {mdir / 'duel_gaps.json'}"
          + ("" if a.dry_run else f" and {mdir / 'tracks.parquet'}") + f" in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
