"""Per-player full-match paths, heatmaps and running stats from tracks.parquet (issue #16).

The Players tab in Stopframe (app/players.py) calls these functions. You can
also run the file on its own to print a stats table. It reads pipeline
artifacts only:
  tracks.parquet    pitch positions per frame (track_identities.py output, so
                    any Lines-tab identity anchors are already applied)
  player_map.json   track id -> player (number, name, how it was named)

Per player:
  - positions are smoothed with a ~0.5 s moving average inside each
    continuous run. A run breaks where tracking is lost for more than
    --max-gap-s seconds, and no distance is counted across that gap.
    Steps faster than --max-speed m/s are dropped as tracking glitches.
    These rules match compute_physical.py.
  - stats: distance, tracked minutes and coverage of the window, average and
    top speed (99th percentile), distance in speed bands, sprints (>= 7 m/s
    held for >= 1 s), and average position.
  - heatmap: seconds spent in each cell of the pitch grid.
  - path: the smoothed positions, sampled at --hz, as separate runs.

Usage:
    python tools/player_paths.py --match-id <id> [--from-min 0 --to-min 45]
"""
import argparse

import numpy as np
import pandas as pd

from _common import (load_calibration, load_meta, load_player_map, match_config, match_dir,
                     pitch_dims)

TEAMS = ("team_a", "team_b")
# m/s; the last band has no upper limit
BANDS = (("walk", 0.0, 2.0), ("jog", 2.0, 4.0), ("run", 4.0, 5.5), ("high", 5.5, 7.0), ("sprint", 7.0, None))
SPRINT_MS, SPRINT_MIN_S = 7.0, 1.0


def player_index(pm):
    """[{id, team, number, name, role, manual, anchors, tracklets, fragile_links, track_ids}] from player_map.json."""
    out = []
    for pid, p in (pm or {}).get("players", {}).items():
        if p.get("team") not in TEAMS or not p.get("track_ids"):
            continue
        out.append({"id": pid, "team": p["team"], "number": p.get("number"), "name": p.get("name"),
                    "role": p.get("role"), "name_source": p.get("name_source"),
                    "manual": str(p.get("name_source", "")).startswith("manual"),
                    "anchors": int(p.get("anchors") or 0), "tracklets": int(p.get("tracklets") or 0),
                    "fragile_links": int(p.get("fragile_links") or 0), "track_ids": [int(t) for t in p["track_ids"]]})
    out.sort(key=lambda p: (p["team"], p["role"] != "GK", p["number"] if p["number"] is not None else 999))
    return out


def player_rows(tracks, track_ids, start=None, end=None):
    """One row per frame (frame, x, y) for a player's track ids, inside [start, end]."""
    t = tracks[tracks["track_id"].isin(track_ids)]
    if start is not None:
        t = t[t["frame"] >= start]
    if end is not None:
        t = t[t["frame"] <= end]
    t = t.dropna(subset=["pitch_x", "pitch_y"])
    if t.empty:
        return pd.DataFrame({"frame": pd.Series(dtype=int), "x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
    g = t.groupby("frame", sort=True)[["pitch_x", "pitch_y"]].mean().reset_index()
    return g.rename(columns={"pitch_x": "x", "pitch_y": "y"})


def runs(rows, fps, max_gap_s=1.0, smooth_s=0.5):
    """Split rows into continuous runs and smooth each. Returns rows plus 'run' and smoothed 'sx', 'sy'."""
    if rows.empty:
        return rows.assign(run=pd.Series(dtype=int), sx=pd.Series(dtype=float), sy=pd.Series(dtype=float))
    f = rows["frame"].to_numpy()
    brk = np.concatenate([[True], np.diff(f) > max_gap_s * fps])
    out = rows.assign(run=np.cumsum(brk) - 1)
    win = max(1, int(round(smooth_s * fps)))
    grp = out.groupby("run", sort=False)
    out["sx"] = grp["x"].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    out["sy"] = grp["y"].transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
    return out


def steps(r, fps, max_speed=12.0):
    """Per-step (dist, dt, speed) inside runs, glitch steps removed."""
    if len(r) < 2:
        z = np.zeros(0)
        return z, z, z
    same = r["run"].to_numpy()[1:] == r["run"].to_numpy()[:-1]
    dx, dy = np.diff(r["sx"].to_numpy()), np.diff(r["sy"].to_numpy())
    dt = np.diff(r["frame"].to_numpy()) / fps
    d = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, d / dt, 0.0)
    ok = same & (dt > 0) & (v <= max_speed)
    return d[ok], dt[ok], v[ok]


def sprint_count(r, fps, max_speed=12.0):
    """Number of spells at >= SPRINT_MS lasting >= SPRINT_MIN_S, within runs."""
    if len(r) < 2:
        return 0
    same = r["run"].to_numpy()[1:] == r["run"].to_numpy()[:-1]
    dt = np.diff(r["frame"].to_numpy()) / fps
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.hypot(np.diff(r["sx"].to_numpy()), np.diff(r["sy"].to_numpy())) / dt
    fast = same & (v >= SPRINT_MS) & (v <= max_speed)
    n, held = 0, 0.0
    for is_fast, step in zip(fast, dt):
        if is_fast:
            held += step
            continue
        n += held >= SPRINT_MIN_S
        held = 0.0
    return int(n + (held >= SPRINT_MIN_S))


def stats(r, fps, window_frames, max_speed=12.0):
    d, dt, v = steps(r, fps, max_speed)
    tracked_s = len(r) / fps
    dist = float(d.sum())
    bands = {}
    for name, lo, hi in BANDS:
        m = (v >= lo) & (v < hi) if hi is not None else (v >= lo)
        bands[name] = round(float(d[m].sum()), 1)
    return {
        "distance_m": round(dist, 1),
        "tracked_min": round(tracked_s / 60, 1),
        "coverage": round(len(r) / window_frames, 3) if window_frames else 0.0,
        "avg_speed": round(dist / dt.sum(), 2) if dt.sum() > 0 else 0.0,
        "top_speed": round(float(np.percentile(v, 99)), 2) if len(v) else 0.0,
        "per_90_m": round(dist / tracked_s * 90 * 60) if tracked_s > 60 else None,
        "bands_m": bands,
        "sprints": sprint_count(r, fps, max_speed),
        "avg_pos": [round(float(r["sx"].mean()), 1), round(float(r["sy"].mean()), 1)] if len(r) else None,
    }


def heatmap(r, fps, L, W, cell=2.5):
    """Seconds per cell: rows are y bands (0 at y=0), columns x bands."""
    nx, ny = int(np.ceil(L / cell)), int(np.ceil(W / cell))
    if r.empty:
        return {"cell_m": cell, "nx": nx, "ny": ny, "secs": np.zeros((ny, nx)).tolist(), "max": 0.0}
    x = r["sx"].clip(0, L - 1e-6).to_numpy()
    y = r["sy"].clip(0, W - 1e-6).to_numpy()
    h, _, _ = np.histogram2d(y, x, bins=[ny, nx], range=[[0, ny * cell], [0, nx * cell]])
    h = h / fps
    return {"cell_m": cell, "nx": nx, "ny": ny, "secs": np.round(h, 2).tolist(), "max": round(float(h.max()), 2)}


def path(r, fps, hz=2.0):
    """[[[frame, x, y], ...] per run], sampled every fps/hz frames."""
    step = max(1, int(round(fps / hz)))
    out = []
    for _, g in r.groupby("run", sort=True):
        keep = ((g["frame"] - g["frame"].iloc[0]) % step == 0).to_numpy().copy()
        keep[-1] = True  # end each run where the player was last seen
        g = g[keep]
        if len(g) >= 2:
            out.append([[int(f), round(float(x), 2), round(float(y), 2)]
                        for f, x, y in g[["frame", "sx", "sy"]].itertuples(index=False)])
    return out


def attacking_x(tracks, players, L):
    """{team: x of the goal it attacks}, from where its keeper stands (else the team's mean x)."""
    out = {}
    for team in TEAMS:
        mine = [p for p in players if p["team"] == team]
        gk = [t for p in mine if p["role"] == "GK" for t in p["track_ids"]]
        ids = gk or [t for p in mine for t in p["track_ids"]]
        xs = tracks.loc[tracks["track_id"].isin(ids), "pitch_x"].dropna()
        own = xs.median() if len(xs) else (0.0 if team == "team_a" else L)
        out[team] = float(L) if own < L / 2 else 0.0
    if out["team_a"] == out["team_b"]:
        out["team_b"] = float(L) - out["team_a"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--from-min", type=float, default=None, help="match clock minute (needs clock_sync)")
    ap.add_argument("--to-min", type=float, default=None)
    ap.add_argument("--max-gap-s", type=float, default=1.0)
    ap.add_argument("--max-speed", type=float, default=12.0)
    ap.add_argument("--tmp-dir", default=".tmp")
    ap.add_argument("--config-dir", default="config")
    a = ap.parse_args()

    meta = load_meta(a.match_id, a.tmp_dir)
    cfg = match_config(a.match_id, a.tmp_dir, a.config_dir)
    fps = float(meta.get("fps") or 25.0)
    win = cfg.get("analysis_window") or {}
    start, end = win.get("start_frame", 0), win.get("end_frame", meta.get("frames", 0))
    ko = (cfg.get("clock_sync") or {}).get("kickoff_frame")
    if ko is not None and a.from_min is not None:
        start = max(start, int(ko + a.from_min * 60 * fps))
    if ko is not None and a.to_min is not None:
        end = min(end, int(ko + a.to_min * 60 * fps))
    tracks = pd.read_parquet(match_dir(a.match_id, a.tmp_dir) / "tracks.parquet",
                             columns=["frame", "track_id", "class", "pitch_x", "pitch_y"])
    tracks = tracks[tracks["class"] == "player"]
    players = player_index(load_player_map(a.match_id, a.tmp_dir))
    if not players:
        raise SystemExit("player_map.json has no players: run track_identities.py first")
    print(f"{'player':<24} {'dist m':>7} {'min':>5} {'cov':>5} {'top':>5} {'spr':>4}  bands walk/jog/run/high/sprint m")
    for p in players:
        r = runs(player_rows(tracks, p["track_ids"], start, end), fps, a.max_gap_s)
        s = stats(r, fps, end - start + 1, a.max_speed)
        b = s["bands_m"]
        label = f"{p['team'][-1].upper()} {p['number']} {p['name'] or ''}"
        print(f"{label:<24} {s['distance_m']:>7.0f} {s['tracked_min']:>5.1f} {s['coverage']:>5.0%} "
              f"{s['top_speed']:>5.1f} {s['sprints']:>4}  " + "/".join(f"{b[k]:.0f}" for k, *_ in BANDS))
    calib = load_calibration(a.match_id, a.config_dir)
    print("pitch", pitch_dims(calib))


if __name__ == "__main__":
    main()
