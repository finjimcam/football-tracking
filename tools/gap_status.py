"""Gap labelling for player tracks (issue #47, item 2): every player position
carries a `status`, and position-weighted outputs weight by it.

Status schema (column `status` in tracks.parquet, player rows):
    detected      a real detection (every row track_identities.py writes)
    interpolated  a short gap filled from the player's own track
    absorbed      a duel gap: the player vanished next to another player and
                  re-emerged next to that same player, so for the gap they
                  were hidden "inside" that player's box
    unknown       not filled: gap too long, physically implausible to bridge,
                  or open-ended (before the first / after the last sighting).
                  Unknown frames have NO row - absence of a row means unknown,
                  as in the Lines picker ("missing frame = lost").
Rows without a status (old tracks.parquet, ball rows) read as `detected`.

Downstream weights (DEFAULT_WEIGHTS, overridable per tool with
--w-interpolated / --w-absorbed): detected 1, interpolated 0.5, absorbed 0.5,
unknown 0 (skipped). Heatmaps and average positions weight rows by these;
physical stats count filled steps towards distance/time but never towards
speed peaks; possession, contact checks and verification read detected rows
only (observed()).

Pipeline hook: track_identities.py calls label_tracks() on the identity rows
before writing tracks.parquet, and writes the one-row-per-gap table
gaps.parquet. This tool re-labels an existing tracks.parquet in place
(idempotent: earlier fill rows are dropped first), so classifier or filler
experiments don't need the ~3-10 min identity re-link.

Plug-in points for other fill strategies (issue #47 items 3-4):
    classify(gap, ctx, **params) -> kind          (default classify_gap)
    fillers[kind](gap, ctx) -> (n, 2) pitch xy for frames gap.start+1 ..
                               gap.end-1, or None to leave the gap unknown
Pass them to label_tracks(..., classify=..., fillers={...}). A filler sees
the gap (endpoints, partner track id) and a GapContext with every detected
player position, so a duel filler can follow ctx.track(gap.partner) and a
smoother can read ctx.track(gap.track_id) on both sides.

Usage:
    python tools/gap_status.py --match-id <match_id>
    python tools/gap_status.py --match-id <match_id> --max-fill-s 4 --duel-m 1.5
"""
import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STATUSES = ("detected", "interpolated", "absorbed", "unknown")
FILLED = ("interpolated", "absorbed")
DEFAULT_WEIGHTS = {"detected": 1.0, "interpolated": 0.5, "absorbed": 0.5, "unknown": 0.0}
GAP_BUCKETS = ((0, 1), (1, 3), (3, 10), (10, np.inf))  # seconds, for reporting
DEFAULTS = {"max_fill_s": 4.0, "max_absorbed_s": None, "duel_m": 1.5, "reemerge_m": 2.5, "vmax": 8.5,
            "reach_m": 3.0}
# The pipeline (issue #47 combined): smoothed fill for short gaps, duel filler
# for absorbed gaps up to 10 s. See pipeline_fillers().
PIPELINE_DEFAULTS = {"max_fill_s": 3.0, "max_absorbed_s": 10.0}


# --------------------------------------------------------------------------
# Reading the column (used by every downstream tool)
# --------------------------------------------------------------------------
def status_of(df):
    """The status of every row; missing column / NaN = detected."""
    if "status" not in df.columns:
        return pd.Series("detected", index=df.index, dtype=object)
    return df["status"].astype(object).where(df["status"].notna(), "detected")


def observed(df):
    """Only real detections - for tools that must not see filled positions
    (possession, contact checks, chain following, verification)."""
    if "status" not in df.columns:
        return df
    return df[status_of(df).to_numpy() == "detected"]


def read_detections(path, columns, **kw):
    """`columns` of tracks.parquet at `path`, detected rows only (fill rows
    dropped) - for readers that want the tracker's real boxes."""
    import pyarrow.parquet as pq
    has = "status" in pq.read_schema(path).names
    df = pd.read_parquet(path, columns=list(columns) + (["status"] if has else []), **kw)
    return observed(df)[list(columns)]


def status_weights(df, weights=None):
    """Per-row weight from the status (unknown / unlisted statuses -> 0)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return status_of(df).map(lambda s: w.get(s, 0.0)).to_numpy(dtype=float)


def add_weight_args(parser):
    parser.add_argument("--w-interpolated", type=float, default=DEFAULT_WEIGHTS["interpolated"],
                        help="Weight of interpolated positions (gap_status.py); 0 = detected only")
    parser.add_argument("--w-absorbed", type=float, default=DEFAULT_WEIGHTS["absorbed"],
                        help="Weight of duel-absorbed positions (gap_status.py)")


def weights_from_args(a):
    return {**DEFAULT_WEIGHTS, "interpolated": a.w_interpolated, "absorbed": a.w_absorbed}


# --------------------------------------------------------------------------
# Gaps
# --------------------------------------------------------------------------
@dataclass
class Gap:
    """Missing frames start+1 .. end-1 of one track. start/end are the last /
    next detected frames; None for an open end (before the first or after the
    last sighting inside the window, with lo/hi then the window edge)."""
    track_id: int
    start: int | None
    end: int | None
    lo: int  # first missing frame
    hi: int  # last missing frame
    xy_start: np.ndarray | None = None
    xy_end: np.ndarray | None = None
    partner: int = -1          # nearest other track at the last sighting (if within duel_m)
    partner_dist_m: float = np.nan
    duel: bool = False         # last seen within duel_m of another player
    reemerge: bool = False     # ... and seen again within reemerge_m of that same player
    kind: str = "unknown"
    reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_missing(self):
        return self.hi - self.lo + 1

    @property
    def frames(self):
        return np.arange(self.lo, self.hi + 1)


class GapContext:
    """Detected player positions, indexed by frame and by track."""

    def __init__(self, frames, track_ids, xy, fps):
        order = np.lexsort((track_ids, frames))
        self.frames = np.asarray(frames)[order]
        self.tids = np.asarray(track_ids)[order]
        self.xy = np.asarray(xy, dtype=float)[order]
        self.fps = float(fps)
        self._by_track = None

    @classmethod
    def from_tracks(cls, players, fps, exclude=()):
        p = observed(players)
        if len(exclude):
            p = p[~p["track_id"].isin(list(exclude))]
        return cls(p["frame"].to_numpy(), p["track_id"].to_numpy(), p[["pitch_x", "pitch_y"]].to_numpy(), fps)

    def at(self, frame):
        """(track_ids, xy) of everyone detected at `frame`."""
        a, b = np.searchsorted(self.frames, [frame, frame + 1])
        return self.tids[a:b], self.xy[a:b]

    def nearest(self, frame, xy, exclude):
        tids, pts = self.at(frame)
        m = tids != exclude
        if not m.any():
            return -1, np.inf
        d = np.hypot(*(pts[m] - xy).T)
        k = int(np.argmin(d))
        return int(tids[m][k]), float(d[k])

    def position(self, track_id, frame, tol=0):
        """Detected position of track_id at frame (nearest within +-tol frames), or None."""
        for df in sorted(range(-tol, tol + 1), key=abs):
            tids, pts = self.at(frame + df)
            k = np.flatnonzero(tids == track_id)
            if len(k):
                return pts[k[0]]
        return None

    def track(self, track_id):
        """(frames, xy) of one track's detections, sorted by frame."""
        if self._by_track is None:
            order = np.lexsort((self.frames, self.tids))
            t = self.tids[order]
            cuts = np.flatnonzero(np.diff(t)) + 1
            self._by_track = {int(t[s]): order[s:e] for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(t)])} \
                if len(t) else {}
        idx = self._by_track.get(int(track_id), np.array([], int))
        return self.frames[idx], self.xy[idx]


def find_gaps(track_id, frames, xy, lo, hi):
    """All Gaps of one track inside [lo, hi] (frames sorted, unique)."""
    gaps = []
    if not len(frames):
        return [Gap(track_id, None, None, lo, hi)]
    if frames[0] > lo:
        gaps.append(Gap(track_id, None, int(frames[0]), lo, int(frames[0]) - 1, None, xy[0]))
    for k in np.flatnonzero(np.diff(frames) > 1):
        a, b = int(frames[k]), int(frames[k + 1])
        gaps.append(Gap(track_id, a, b, a + 1, b - 1, xy[k], xy[k + 1]))
    if frames[-1] < hi:
        gaps.append(Gap(track_id, int(frames[-1]), None, int(frames[-1]) + 1, hi, xy[-1], None))
    return gaps


def classify_gap(gap, ctx, max_fill_s=4.0, max_absorbed_s=None, duel_m=1.5, reemerge_m=2.5, vmax=8.5,
                 reach_m=3.0):
    """Sets gap.partner/duel/reemerge/reason and returns the kind:
    absorbed     two-sided, <= max_absorbed_s (default max_fill_s), last seen within duel_m of another
                 player AND seen again within reemerge_m of that same player
    interpolated two-sided, <= max_fill_s, runnable (distance <= reach_m +
                 vmax * gap time), not a duel re-emergence
    unknown      everything else (open-ended, too long, not runnable)"""
    fps = ctx.fps
    if gap.start is not None:
        gap.partner, gap.partner_dist_m = ctx.nearest(gap.start, gap.xy_start, gap.track_id)
        gap.duel = gap.partner_dist_m <= duel_m
        if not gap.duel:
            gap.partner = -1
    if gap.start is None or gap.end is None:
        gap.reason = "open_end"
        return "unknown"
    gap_s = (gap.end - gap.start) / fps
    if gap.duel:
        q = ctx.position(gap.partner, gap.end, tol=2)
        gap.reemerge = q is not None and float(np.hypot(*(q - gap.xy_end))) <= reemerge_m
    cap = max_absorbed_s if gap.reemerge and max_absorbed_s is not None else max_fill_s
    if gap_s > cap:
        gap.reason = "too_long"
        return "unknown"
    dist = float(np.hypot(*(gap.xy_end - gap.xy_start)))
    if dist > reach_m + vmax * gap_s:
        gap.reason = "not_runnable"
        return "unknown"
    gap.reason = "duel" if gap.reemerge else "short"
    return "absorbed" if gap.reemerge else "interpolated"


def linear_fill(gap, ctx):
    """The existing fill (cut_phase_clips.fill_gaps): straight line in pitch space."""
    u = (gap.frames - gap.start) / (gap.end - gap.start)
    return gap.xy_start + (gap.xy_end - gap.xy_start) * u[:, None]


DEFAULT_FILLERS = {"interpolated": linear_fill, "absorbed": linear_fill}


def pipeline_fillers(max_fill_s=PIPELINE_DEFAULTS["max_fill_s"], ignore=(), linear=False):
    """The fillers the pipeline uses (issue #47 combined):
    interpolated  interp_gaps.py RTS smoother (straight line where the smooth
                  path is not physical); refuses gaps > max_fill_s
    absorbed      duel_gaps.py: the partner's path plus a blended offset, with
                  its kit / swap checks at the split; ambiguous -> unknown
    Returns dict(fillers=..., context_hook=...) for label_tracks(**...).
    linear=True gives the old straight-line fill for both kinds."""
    if linear:
        return {"fillers": dict(DEFAULT_FILLERS), "context_hook": None}
    import duel_gaps
    import interp_gaps

    def hook(ctx, players):
        ctx.duel_index = duel_gaps.TrackIndex(players[players["pitch_x"].notna()])

    def absorbed(gap, ctx):
        return duel_gaps.fill(gap, ctx, ignore=tuple(ignore))

    return {"fillers": {"interpolated": interp_gaps.make_filler(max_gap_s=max_fill_s), "absorbed": absorbed},
            "context_hook": hook}


def label_tracks(tracks, fps, track_ids=None, window=None, to_image=None, exclude_partners=(),
                 classify=classify_gap, fillers=None, context_hook=None, **params):
    """Label the gaps of `track_ids` (default: every player track) inside
    `window` (lo, hi; default: the tracks' frame range) and append fill rows.

    Returns (tracks with a `status` column + fill rows, gaps DataFrame).
    Earlier fill rows are dropped first, so re-labelling is idempotent.
    to_image(frames, xy) -> (n, 2) foot pixels for fill rows (downstream
    tools re-project foot pixels through the calibration); without it the
    fill rows' foot_x/foot_y are NaN. exclude_partners: track ids that can't
    be a duel partner (officials, people off the pitch). context_hook(ctx,
    players) runs once after the GapContext is built (detected player rows,
    all columns), so a filler can attach its own index to ctx."""
    fillers = {**DEFAULT_FILLERS, **(fillers or {})}
    params = {**DEFAULTS, **params}
    base = tracks[~status_of(tracks).isin(FILLED).to_numpy()].copy()
    is_player = (base["class"] == "player").to_numpy()
    base["status"] = np.where(is_player, "detected", None)
    players = base[is_player]
    if track_ids is None:
        track_ids = sorted(players["track_id"].unique())
    if window is None:
        window = (int(base["frame"].min()), int(base["frame"].max()))
    lo, hi = window
    ctx = GapContext.from_tracks(players, fps, exclude=exclude_partners)
    if context_hook is not None:
        context_hook(ctx, players)

    gap_rows, fill_parts = [], []
    excluded = set(exclude_partners)
    for tid in track_ids:
        frames, xy = ctx.track(tid)
        if tid in excluded:  # not in ctx: read it directly
            g = players[players["track_id"] == tid].sort_values("frame")
            frames, xy = g["frame"].to_numpy(), g[["pitch_x", "pitch_y"]].to_numpy()
        keep = (frames >= lo) & (frames <= hi)
        frames, xy = frames[keep], xy[keep]
        for gap in find_gaps(int(tid), frames, xy, lo, hi):
            gap.kind = classify(gap, ctx, **params)
            pts = None
            if gap.kind in fillers:
                pts = fillers[gap.kind](gap, ctx)
                if pts is None:
                    gap.kind, gap.reason = "unknown", gap.reason + "+fill_refused"
            if pts is not None:
                fill_parts.append((gap.track_id, gap.frames, np.asarray(pts, float), gap.kind))
            gap_s = None if gap.start is None or gap.end is None else (gap.end - gap.start) / fps
            dist = (None if gap.xy_start is None or gap.xy_end is None
                    else float(np.hypot(*(gap.xy_end - gap.xy_start))))
            gap_rows.append({
                "track_id": gap.track_id, "start": gap.start, "end": gap.end, "lo": gap.lo, "hi": gap.hi,
                "n_missing": gap.n_missing, "missing_s": gap.n_missing / fps, "gap_s": gap_s,
                "kind": gap.kind, "reason": gap.reason, "duel": gap.duel, "reemerge": gap.reemerge,
                "partner": gap.partner, "partner_dist_m": gap.partner_dist_m, "dist_m": dist,
                "speed_mps": None if not gap_s or dist is None else dist / gap_s})

    if fill_parts:
        fr = np.concatenate([p[1] for p in fill_parts])
        xy = np.concatenate([p[2] for p in fill_parts])
        fill = pd.DataFrame({
            "frame": fr,
            "t_s": np.round(fr / fps, 3),
            "track_id": np.concatenate([np.full(len(p[1]), p[0]) for p in fill_parts]),
            "class": "player", "conf": 0.0,
            "pitch_x": xy[:, 0], "pitch_y": xy[:, 1],
            "status": np.concatenate([np.full(len(p[1]), p[3], dtype=object) for p in fill_parts]),
        })
        foot = to_image(fr, xy) if to_image is not None else np.full((len(fr), 2), np.nan)
        fill["foot_x"], fill["foot_y"] = foot[:, 0], foot[:, 1]
        fill = fill.reindex(columns=base.columns)
        for col in ("frame", "track_id"):
            fill[col] = fill[col].astype(base[col].dtype)
        out = pd.concat([base, fill], ignore_index=True)
    else:
        out = base
    out = out.sort_values(["frame", "class", "track_id"], kind="stable").reset_index(drop=True)
    gaps = pd.DataFrame(gap_rows, columns=[
        "track_id", "start", "end", "lo", "hi", "n_missing", "missing_s", "gap_s", "kind", "reason",
        "duel", "reemerge", "partner", "partner_dist_m", "dist_m", "speed_mps"])
    return out, gaps


def absorbed_lookup(gaps):
    """is_absorbed(track_id, frame) from a gaps table - lets the Lines picker
    mark chain fills that fall inside a duel gap ('a' instead of 'f')."""
    spans = {}
    if gaps is not None and len(gaps):
        for r in gaps[gaps["kind"] == "absorbed"].itertuples():
            spans.setdefault(int(r.track_id), []).append((int(r.lo), int(r.hi)))
    spans = {t: np.array(sorted(v)) for t, v in spans.items()}

    def is_absorbed(track_id, frame):
        s = spans.get(int(track_id))
        if s is None:
            return False
        k = np.searchsorted(s[:, 0], frame, side="right") - 1
        return bool(k >= 0 and s[k, 1] >= frame)
    return is_absorbed


def homography_to_image(calib):
    """to_image() for label_tracks from a loaded dynamic/static calibration:
    pitch xy -> foot pixels through the inverse of each frame's homography."""
    if calib is None:
        return None

    def to_image(frames, xy):
        if "frame_H" in calib:
            pos = np.clip(np.searchsorted(calib["frame_index"], frames), 0, len(calib["frame_index"]) - 1)
            Hinv = np.linalg.inv(calib["frame_H"][pos])
        else:
            Hinv = np.broadcast_to(np.linalg.inv(calib["homography"]), (len(frames), 3, 3))
        h = np.einsum("nij,nj->ni", Hinv, np.column_stack([xy, np.ones(len(xy))]))
        with np.errstate(divide="ignore", invalid="ignore"):
            return h[:, :2] / h[:, 2:3]
    return to_image


def summarize(tracks, gaps, track_ids, window, fps):
    """Frames per status class (unknown = window frames with no row) and
    missing-frame totals by gap length."""
    lo, hi = window
    n = hi - lo + 1
    p = tracks[(tracks["class"] == "player") & tracks["track_id"].isin(list(track_ids))
               & tracks["frame"].between(lo, hi)]
    counts = status_of(p).value_counts().to_dict()
    counts = {s: int(counts.get(s, 0)) for s in STATUSES[:3]}
    counts["unknown"] = int(n * len(track_ids) - sum(counts.values()))
    buckets = {}
    for b0, b1 in GAP_BUCKETS:
        g = gaps[(gaps["missing_s"] >= b0) & (gaps["missing_s"] < b1)]
        key = f"{b0}-{b1}s" if np.isfinite(b1) else f">{b0}s"
        buckets[key] = {"gaps": int(len(g)), "frames": int(g["n_missing"].sum()),
                        "by_kind": {k: int(v) for k, v in g.groupby("kind")["n_missing"].sum().items()}}
    two = gaps[gaps["start"].notna()]
    return {"frames_by_status": counts,
            "share_by_status": {k: round(v / max(n * len(track_ids), 1), 4) for k, v in counts.items()},
            "gaps": int(len(gaps)), "gaps_by_kind": {k: int(v) for k, v in gaps["kind"].value_counts().items()},
            "duel_gaps": int(two["duel"].sum()), "missing_by_gap_length": buckets}


def main():
    from _common import load_calibration, load_meta, load_player_map, match_dir, write_json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--max-fill-s", type=float, default=PIPELINE_DEFAULTS["max_fill_s"],
                    help="Longest non-duel gap filled (smoothed fill); longer = unknown")
    ap.add_argument("--max-absorbed-s", type=float, default=PIPELINE_DEFAULTS["max_absorbed_s"],
                    help="Longest duel (absorbed) gap filled along the partner's path; longer = unknown")
    ap.add_argument("--linear-fill", action="store_true",
                    help="Old straight-line fill for both kinds (no smoother, no duel filler checks)")
    ap.add_argument("--duel-m", type=float, default=DEFAULTS["duel_m"],
                    help="Last sighting this close to another player = duel")
    ap.add_argument("--reemerge-m", type=float, default=DEFAULTS["reemerge_m"],
                    help="Duel counts as absorbed if seen again this close to the same player")
    ap.add_argument("--vmax", type=float, default=DEFAULTS["vmax"], help="Max speed to bridge a gap, m/s")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()

    t0 = time.time()
    meta = load_meta(a.match_id, a.tmp_dir)
    mdir = match_dir(a.match_id, a.tmp_dir)
    pm = load_player_map(a.match_id, a.tmp_dir)
    if pm is None:
        raise SystemExit("player_map.json missing - run tools/track_identities.py first")
    calib = load_calibration(a.match_id, a.config_dir, a.calibration)
    tracks = pd.read_parquet(mdir / "tracks.parquet")
    ids = sorted(int(t) for t in pm["track_to_player"])
    window = (pm["window"]["start_frame"], pm["window"]["end_frame"])
    ignored = [int(i["track_id"]) for i in pm.get("ignored_tracks", [])]
    out, gaps = label_tracks(tracks, meta["fps"], track_ids=ids, window=window,
                             to_image=homography_to_image(calib), exclude_partners=ignored,
                             **pipeline_fillers(a.max_fill_s, ignore=ignored, linear=a.linear_fill),
                             max_fill_s=a.max_fill_s, max_absorbed_s=a.max_absorbed_s, duel_m=a.duel_m,
                             reemerge_m=a.reemerge_m, vmax=a.vmax)
    out.to_parquet(mdir / "tracks.parquet", index=False)
    gaps.to_parquet(mdir / "gaps.parquet", index=False)
    s = summarize(out, gaps, ids, window, meta["fps"])
    s["params"] = {"max_fill_s": a.max_fill_s, "max_absorbed_s": a.max_absorbed_s, "linear_fill": a.linear_fill,
                   "duel_m": a.duel_m, "reemerge_m": a.reemerge_m, "vmax": a.vmax}
    s["runtime_s"] = round(time.time() - t0, 2)
    write_json(mdir / "gap_status.json", s)
    tot = sum(s["frames_by_status"].values())
    print(f"[{a.match_id}] {len(ids)} identities, {s['gaps']} gaps ({s['duel_gaps']} duel gaps) in {s['runtime_s']} s")
    print("  player-frames by status: " + ", ".join(
        f"{k} {v} ({v / max(tot, 1):.1%})" for k, v in s["frames_by_status"].items()))
    for k, b in s["missing_by_gap_length"].items():
        print(f"  missing in gaps {k:>7}: {b['frames']:>7} frames in {b['gaps']:>5} gaps  {b['by_kind']}")
    print(f"Wrote tracks.parquet (status column + fill rows), gaps.parquet, gap_status.json to {mdir}")


if __name__ == "__main__":
    main()
