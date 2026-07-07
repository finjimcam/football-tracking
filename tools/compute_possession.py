"""Compute team possession from tracks.parquet + team_map.json →
.tmp/<match_id>/possession.json. Reads only pipeline artifacts, never the video.

Per frame where the ball is detected (highest-conf ball box if several):
possession candidate = team of the nearest player foot-point within 2.5 m
(calibrated) or ~60 px (uncalibrated). Referees/keepers (`unknown`) can't
hold possession. Hysteresis: possession only switches after ~0.4 s
(fps-scaled) of consecutive frames for the new team.

Team possession % = attributed frames / ball-detected frames.

Usage:
    python tools/compute_possession.py --match-id <match_id>
    python tools/compute_possession.py --match-id <match_id> --calibration config/other_calibration.json
"""
import argparse
import math

import numpy as np

from _common import (apply_calibration, load_calibration, load_meta, load_player_map,
                     load_team_map, load_tracks, match_dir, pitch_dims,
                     resolve_track_teams, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--radius-m", type=float, default=2.5, help="Possession radius when calibrated")
    parser.add_argument("--radius-px", type=float, default=60.0, help="Possession radius when uncalibrated")
    parser.add_argument("--switch-s", type=float, default=0.4, help="Hysteresis: seconds before possession switches")
    parser.add_argument("--ball-margin-m", type=float, default=2.0,
                        help="Calibrated: drop ball detections projecting farther than this outside the pitch")
    parser.add_argument("--calibration", default=None, help="Calibration file (default: config/<match_id>_calibration.json)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    team_map = load_team_map(args.match_id, args.tmp_dir)
    calib = load_calibration(args.match_id, args.config_dir, args.calibration)
    tracks, unit = apply_calibration(load_tracks(args.match_id, args.tmp_dir), calib)

    radius = args.radius_m if calib is not None else args.radius_px
    switch_frames = max(1, math.ceil(args.switch_s * meta["fps"]))

    players = tracks[tracks["class"] == "player"].copy()
    # Roster-space teams with curation applied: ignored tracks (refs/
    # spectators) map to "ignored" and fall out of the isin filter below.
    pm = load_player_map(args.match_id, args.tmp_dir)
    track_teams = resolve_track_teams(players["track_id"].unique(), team_map, pm)
    players["team"] = players["track_id"].map(track_teams)
    n_ignored = sum(1 for v in track_teams.values() if v == "ignored")
    if n_ignored:
        print(f"Excluding {n_ignored} ignored track(s) from possession")
    players = players[players["team"].isin(["team_a", "team_b"])]
    balls = tracks[tracks["class"] == "ball"]
    if calib is not None:
        # A "ball" projecting off the pitch is a false positive (spare ball /
        # bag behind the line) — on testvideo most detections were exactly
        # that, and pre-curation they attributed possession to the spectators
        # standing next to them.
        length, width = pitch_dims(calib)
        m = args.ball_margin_m
        off = (~balls["pitch_x"].between(-m, length + m)
               | ~balls["pitch_y"].between(-m, width + m))
        if int(off.sum()):
            print(f"Dropping {int(off.sum())}/{len(balls)} ball detection(s) that project "
                  "off the pitch (false positives)")
        balls = balls[~off]
    balls = balls.loc[balls.groupby("frame")["conf"].idxmax()]  # top-1 ball per frame

    player_frames = {f: g for f, g in players.groupby("frame")}

    attributed = {"team_a": 0, "team_b": 0}
    current, pending_team, pending_count, switches = None, None, 0, 0
    for _, ball in balls.sort_values("frame").iterrows():
        grp = player_frames.get(ball["frame"])
        candidate = None
        if grp is not None:
            d = np.hypot(grp["pitch_x"] - ball["pitch_x"], grp["pitch_y"] - ball["pitch_y"])
            if d.min() <= radius:
                candidate = grp.loc[d.idxmin(), "team"]

        if candidate is not None:
            if current is None:
                current = candidate
            elif candidate != current:
                pending_count = pending_count + 1 if pending_team == candidate else 1
                pending_team = candidate
                if pending_count >= switch_frames:
                    current, pending_team, pending_count = candidate, None, 0
                    switches += 1
            else:
                pending_team, pending_count = None, 0
        if current is not None and candidate is not None:
            attributed[current] += 1

    ball_frames = len(balls)
    total_attr = attributed["team_a"] + attributed["team_b"]
    ball_rate = ball_frames / meta["frame_count"] if meta["frame_count"] else 0.0
    attr_rate = total_attr / ball_frames if ball_frames else 0.0
    # Low confidence when the ball is rarely detected OR rarely near anyone —
    # a percentage built from a handful of attributed frames is noise.
    confidence = ("low" if ball_rate < 0.30 or attr_rate < 0.20
                  else ("medium" if calib is None else "high"))

    result = {
        "team_a_pct": round(100 * attributed["team_a"] / total_attr, 1) if total_attr else None,
        "team_b_pct": round(100 * attributed["team_b"] / total_attr, 1) if total_attr else None,
        "attributed_frames": total_attr,
        "ball_frames": ball_frames,
        "total_frames": meta["frame_count"],
        "ball_detection_rate": round(ball_rate, 4),
        "possession_switches": switches,
        "radius": radius,
        "unit": unit,
        "calibrated": calib is not None,
        "confidence": confidence,
    }
    out = write_json(match_dir(args.match_id, args.tmp_dir) / "possession.json", result)

    if total_attr:
        print(f"Possession: team_a {result['team_a_pct']}% — team_b {result['team_b_pct']}% "
              f"({total_attr}/{ball_frames} ball frames attributed, {switches} switches)")
    else:
        print("Possession: could not attribute any frames (ball never near a team player)")
    print(f"Ball detection rate: {ball_rate:.1%} — confidence: {confidence}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
