"""Merge fragmented player track IDs in tracks.parquet (artifact-only, cheap).

Trackers mint a new ID whenever a player is lost longer than the track buffer
(occlusion, detection flicker, fast pan), so one player ends up as several
track_ids — inflating track counts, splitting per-player stats, and dotting
the average-position map with phantom players. This tool stitches those
fragments back together with space-time matching, no video access:

  For every track A that ends and track B that starts within --max-gap-s,
  extrapolate A's exit position along its exit velocity to B's start time.
  If the predicted point is within `base + growth * gap_seconds` of B's entry
  position, (A, B) is a stitch candidate. Greedy best-score matching (each
  track end / start used at most once) then merges chains via union-find;
  a merged chain keeps the earliest fragment's track_id.

Positions are pitch meters when a calibration exists (robust under a panning
camera — pixel positions jump when the camera moves, pitch positions don't),
otherwise pixels with thresholds scaled by the median player box height.

The first run backs up the tracker's original output to tracks_raw.parquet
and rewrites tracks.parquet in place, so every downstream tool just works.
Re-runs always start from tracks_raw.parquet (idempotent). Ball rows pass
through untouched.

Run order: run_tracking.py → THIS → assign_teams.py (team_map.json is keyed
by track_id, so stitching after team assignment leaves the map stale).

Usage:
    python tools/stitch_tracks.py --match-id <match_id>
    python tools/stitch_tracks.py --match-id <match_id> --max-gap-s 3 --base-m 3
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from _common import apply_calibration, load_calibration, load_meta, match_dir

EDGE_WINDOW = 3   # rows to average for a track's entry/exit position
VEL_WINDOW = 5    # rows to fit the exit velocity over


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def track_profile(grp):
    """Entry/exit state of one track: positions, exit velocity, frame span."""
    xy = grp[["pos_x", "pos_y"]].to_numpy()
    t = grp["t_s"].to_numpy()
    entry = xy[:EDGE_WINDOW].mean(axis=0)
    exit_ = xy[-EDGE_WINDOW:].mean(axis=0)
    if len(grp) >= 2:
        w = min(VEL_WINDOW, len(grp))
        dt = t[-1] - t[-w]
        vel = (xy[-1] - xy[-w]) / dt if dt > 0 else np.zeros(2)
    else:
        vel = np.zeros(2)
    return {
        "start_frame": int(grp["frame"].iloc[0]),
        "end_frame": int(grp["frame"].iloc[-1]),
        "start_t": float(t[0]), "end_t": float(t[-1]),
        "entry": entry, "exit": exit_, "vel": vel,
        "n": len(grp),
    }


def find_stitches(profiles, max_gap_s, base, growth, vmax):
    """Greedy best-score matching of track ends to later track starts."""
    tids = sorted(profiles)
    candidates = []
    for a in tids:
        pa = profiles[a]
        for b in tids:
            if b == a:
                continue
            pb = profiles[b]
            gap = pb["start_t"] - pa["end_t"]
            if not (0 < gap <= max_gap_s):
                continue
            vel = pa["vel"]
            speed = float(np.hypot(*vel))
            if speed > vmax:
                vel = vel * (vmax / speed)
            predicted = pa["exit"] + vel * gap
            score = float(np.hypot(*(predicted - pb["entry"])))
            if score <= base + growth * gap:
                candidates.append((score, a, b))

    used_end, used_start, pairs = set(), set(), []
    for score, a, b in sorted(candidates):
        if a in used_end or b in used_start:
            continue
        used_end.add(a)
        used_start.add(b)
        pairs.append((a, b, score))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--max-gap-s", type=float, default=2.0,
                        help="Longest disappearance (seconds) that can be bridged")
    parser.add_argument("--base-m", type=float, default=2.0,
                        help="Calibrated: base match radius in meters")
    parser.add_argument("--growth-m", type=float, default=3.0,
                        help="Calibrated: radius growth per second of gap (position uncertainty)")
    parser.add_argument("--vmax-m", type=float, default=8.0,
                        help="Calibrated: cap on extrapolation speed, m/s")
    parser.add_argument("--min-track-frames", type=int, default=2,
                        help="Drop player tracks shorter than this many rows (detector blips)")
    parser.add_argument("--calibration", default=None,
                        help="Calibration file (default: config/<match_id>_calibration.json)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    load_meta(args.match_id, args.tmp_dir)  # fail early with the standard hint
    mdir = match_dir(args.match_id, args.tmp_dir)
    raw_path = mdir / "tracks_raw.parquet"
    out_path = mdir / "tracks.parquet"
    src = raw_path if raw_path.exists() else out_path
    if not src.exists():
        raise SystemExit(f"{src} not found — run tools/run_tracking.py first")
    tracks = pd.read_parquet(src)
    if not raw_path.exists():
        tracks.to_parquet(raw_path, index=False)
        print(f"Backed up tracker output to {raw_path}")

    calib = load_calibration(args.match_id, args.config_dir, args.calibration)
    tracks, unit = apply_calibration(tracks, calib, out_cols=("pos_x", "pos_y"))

    players = tracks[(tracks["class"] == "player") & (tracks["track_id"] >= 0)]
    if unit == "m":
        base, growth, vmax = args.base_m, args.growth_m, args.vmax_m
    else:
        h = float((players["y2"] - players["y1"]).median())
        base, growth, vmax = 1.5 * h, 2.5 * h, 6.0 * h
        print(f"No calibration — pixel thresholds from median box height {h:.0f}px: "
              f"base {base:.0f}px, growth {growth:.0f}px/s, vmax {vmax:.0f}px/s")

    sizes = players.groupby("track_id").size()
    blips = sizes[sizes < args.min_track_frames].index
    profiles = {int(tid): track_profile(grp.sort_values("frame"))
                for tid, grp in players[~players["track_id"].isin(blips)].groupby("track_id")}

    pairs = find_stitches(profiles, args.max_gap_s, base, growth, vmax)
    uf = UnionFind()
    for a, b, _ in pairs:
        uf.union(a, b)

    groups = defaultdict(list)
    for tid in profiles:
        groups[uf.find(tid)].append(tid)
    remap = {}
    for members in groups.values():
        canonical = min(members, key=lambda t: profiles[t]["start_frame"])
        for t in members:
            remap[t] = canonical

    out = tracks.drop(columns=["pos_x", "pos_y"]).copy()
    is_player = out["class"] == "player"
    out = out[~(is_player & out["track_id"].isin(blips))]
    is_player = out["class"] == "player"
    out.loc[is_player, "track_id"] = out.loc[is_player, "track_id"].map(remap).astype(int)
    out = out.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)
    out.to_parquet(out_path, index=False)

    before = len(profiles) + len(blips)
    after = out.loc[out["class"] == "player", "track_id"].nunique()
    span = out[out["class"] == "player"].groupby("track_id")["frame"].agg(["min", "max"])
    med_life = float((span["max"] - span["min"] + 1).median())
    print(f"Stitched {len(pairs)} joins ({unit} space): {before} tracks -> {after} "
          f"({len(blips)} sub-{args.min_track_frames}-frame blips dropped)")
    print(f"Median track lifetime now {med_life:.0f} frames")
    print(f"Wrote {out_path}")
    print("Re-run tools/assign_teams.py now — team_map.json is keyed by track_id and is stale.")


if __name__ == "__main__":
    main()
