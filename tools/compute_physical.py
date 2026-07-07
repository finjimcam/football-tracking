"""Distance and speed stats from tracks.parquet →
.tmp/<match_id>/physical.json. Reads only pipeline artifacts, never the video.

Per player track: smooth foot positions with a ~0.5 s moving average before
differencing (tracker jitter would otherwise inflate distance), then sum step
distances. Glitch filtering: steps across gaps > 1 s are dropped (the tracker
lost the player), and calibrated speeds > 12 m/s are dropped as tracking
glitches (uncalibrated: top 0.5% of speeds dropped instead, since px/s has no
absolute ceiling).

Team totals are the primary numbers. If player_map.json exists (label_roster.py
+ assign_identities.py), per-player figures are aggregated across each
player's track fragments and reported by shirt number ("players" section,
confidence "identified" — each carries its tracking coverage, since distance
during untracked gaps is still missing). Without it, per-track figures are
emitted flagged best-effort — ID fragmentation splits one player across
track_ids, so per-track distance underestimates per-player distance.

Usage:
    python tools/compute_physical.py --match-id <match_id>
"""
import argparse

import numpy as np

from _common import (apply_calibration, load_calibration, load_meta, load_player_map,
                     load_team_map, load_tracks, match_dir, resolve_track_teams, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--smooth-s", type=float, default=0.5, help="Moving-average window in seconds")
    parser.add_argument("--max-speed", type=float, default=12.0, help="Calibrated glitch cutoff, m/s")
    parser.add_argument("--max-gap-s", type=float, default=1.0, help="Drop steps across gaps longer than this")
    parser.add_argument("--calibration", default=None, help="Calibration file (default: config/<match_id>_calibration.json)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    team_map = load_team_map(args.match_id, args.tmp_dir)
    calib = load_calibration(args.match_id, args.config_dir, args.calibration)
    tracks, unit = apply_calibration(load_tracks(args.match_id, args.tmp_dir), calib)
    calibrated = calib is not None

    window = max(1, round(args.smooth_s * meta["fps"]))
    players = tracks[(tracks["class"] == "player") & (tracks["track_id"] >= 0)].copy()
    # Teams in roster space, curation applied: identified tracks take their
    # player's team, ignored tracks (refs/spectators) are dropped entirely.
    pm = load_player_map(args.match_id, args.tmp_dir)
    track_teams = resolve_track_teams(players["track_id"].unique(), team_map, pm)
    players["team"] = players["track_id"].map(track_teams)
    n_ignored = sum(1 for v in track_teams.values() if v == "ignored")
    if n_ignored:
        print(f"Excluding {n_ignored} ignored track(s) (referees/spectators) from all stats")
    players = players[players["team"] != "ignored"]

    per_track = []
    all_speeds = []
    for tid, grp in players.groupby("track_id"):
        grp = grp.sort_values("frame")
        if len(grp) < window + 1:
            continue
        x = grp["pitch_x"].rolling(window, min_periods=1, center=True).mean().to_numpy()
        y = grp["pitch_y"].rolling(window, min_periods=1, center=True).mean().to_numpy()
        t = grp["t_s"].to_numpy()
        dt = np.diff(t)
        step = np.hypot(np.diff(x), np.diff(y))
        keep = dt <= args.max_gap_s
        speed = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
        if calibrated:
            keep &= speed <= args.max_speed
        per_track.append({
            "track_id": int(tid),
            "team": grp["team"].iloc[0],
            "_step": step, "_speed": speed, "_keep": keep, "_dt": dt,
        })
        all_speeds.append(speed[keep])

    if not per_track:
        raise SystemExit("No player track long enough to compute physical stats")

    if not calibrated and all_speeds:
        # No absolute m/s ceiling in pixel units — drop the top 0.5% as glitches.
        cutoff = np.percentile(np.concatenate(all_speeds), 99.5)
        for tr in per_track:
            tr["_keep"] &= tr["_speed"] <= cutoff

    track_rows = []
    for tr in per_track:
        keep = tr["_keep"]
        track_rows.append({
            "track_id": tr["track_id"],
            "team": tr["team"],
            "distance": round(float(tr["_step"][keep].sum()), 1),
            "duration_s": round(float(tr["_dt"][keep].sum()), 1),
            "max_speed": round(float(tr["_speed"][keep].max()), 2) if keep.any() else 0.0,
        })
    track_rows.sort(key=lambda r: r["distance"], reverse=True)

    players_out = []
    if pm:
        t2p = pm["track_to_player"]
        for r in track_rows:
            r["player"] = t2p.get(r["track_id"])
        for key, info in pm["players"].items():
            rows = [r for r in track_rows if r["player"] == key]
            if not rows:
                continue
            dist = sum(r["distance"] for r in rows)
            dur = sum(r["duration_s"] for r in rows)
            players_out.append({
                "player": key, "team": info["team"], "number": info["number"],
                "distance": round(dist, 1),
                "duration_s": round(dur, 1),
                "avg_speed": round(dist / dur, 2) if dur else 0.0,
                "max_speed": max(r["max_speed"] for r in rows),
                "n_tracks": len(rows),
                "coverage": info["coverage"],
            })
        players_out.sort(key=lambda r: r["distance"], reverse=True)

    teams = {}
    for team in ("team_a", "team_b"):
        rows = [r for r in track_rows if r["team"] == team]
        if not rows:
            continue
        dist = sum(r["distance"] for r in rows)
        dur = sum(r["duration_s"] for r in rows)
        teams[team] = {
            "distance_total": round(dist, 1),
            "avg_speed": round(dist / dur, 2) if dur else 0.0,
            "max_speed": max(r["max_speed"] for r in rows),
            "n_tracks": len(rows),
        }

    result = {
        "calibrated": calibrated,
        "unit": unit,
        "speed_unit": f"{unit}/s",
        "smooth_window_frames": window,
        "teams": teams,
        "players": players_out,
        "tracks": track_rows,
        "per_player_confidence": "identified" if players_out else "best_effort",
        "per_player_note": ("roster identities from player_map.json; distance during "
                            "untracked gaps is missing (see each player's coverage)"
                            if players_out else
                            "track IDs fragment; one player may appear as several tracks"),
    }
    out = write_json(match_dir(args.match_id, args.tmp_dir) / "physical.json", result)

    for team, t in teams.items():
        print(f"{team}: total distance {t['distance_total']} {unit} over {t['n_tracks']} tracks, "
              f"avg speed {t['avg_speed']} {unit}/s, max {t['max_speed']} {unit}/s")
    if not calibrated:
        print("NOTE: uncalibrated — all figures are in pixels, flagged accordingly")
    if players_out:
        print(f"\n{'player':>8} {'distance':>10} {'max speed':>10} {'coverage':>8}")
        for p in players_out:
            print(f"{p['player']:>8} {p['distance']:>8.0f} {unit} {p['max_speed']:>6.2f} {unit}/s {p['coverage']:>8.0%}")
        n_unassigned = sum(1 for r in track_rows if not r.get("player"))
        print(f"({n_unassigned} unidentified track(s) excluded from per-player numbers)")
    else:
        print("Per-player numbers are best-effort (ID fragmentation) — run "
              "label_roster.py + assign_identities.py for identified per-player stats.")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
