"""Compute team possession from tracks.parquet + team_map.json →
.tmp/<match_id>/possession.json (+ possession_timeline.parquet). Reads only
pipeline artifacts, never the video.

Per frame where the ball is located (highest-conf ball row if several):
possession candidate = team of the nearest player foot-point within 2.5 m
(calibrated) or ~60 px (uncalibrated). Referees/keepers' `unknown` tracks
can't hold possession. Hysteresis: possession only switches after ~0.4 s
(fps-scaled) of consecutive frames for the new team.

Two figures:
  - team_a_pct / team_b_pct: attributed frames / ball frames (the original
    definition: frames where the ball is seen next to a player);
  - team_a_time_pct / team_b_time_pct: share of match time each team holds
    the ball, with possession carried through ball/detection dropouts for up
    to --max-hold-s (a pass or a 2-second detection gap doesn't make the
    ball nobody's). This is the closer analogue of broadcast possession.

possession_timeline.parquet - one row per frame of the tracked window:
  frame, state (team_a|team_b|none after hysteresis + hold), candidate
  (raw nearest team or none), holder (track_id or -1), ball (bool),
  ball_x, ball_y. tools/cut_phase_clips.py and tools/verify_tracking.py
  read this.

--include-unidentified lets tracks without an identity take part when their
detections carry a confident team kit (tracks.parquet `kit` column from
tools/track_identities.py): the nearest player to the ball is often in a
short, unlinked fragment.

Usage:
    python tools/compute_possession.py --match-id <match_id>
    python tools/compute_possession.py --match-id <match_id> --calibration config/other_calibration.json
"""
import argparse
import math

import numpy as np
import pandas as pd

from _common import (apply_calibration, load_calibration, load_meta, load_player_map,
                     load_team_map, load_tracks, match_dir, pitch_dims,
                     resolve_track_teams, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--radius-m", type=float, default=2.5, help="Possession radius when calibrated")
    parser.add_argument("--radius-px", type=float, default=60.0, help="Possession radius when uncalibrated")
    parser.add_argument("--switch-s", type=float, default=0.4, help="Hysteresis: seconds before possession switches")
    parser.add_argument("--max-hold-s", type=float, default=8.0,
                        help="Carry possession through gaps without a ball-near-player observation for this long")
    parser.add_argument("--ball-margin-m", type=float, default=2.0,
                        help="Calibrated: drop ball detections projecting farther than this outside the pitch")
    parser.add_argument("--include-unidentified", action="store_true",
                        help="Let unlinked tracks with a confident team kit hold possession")
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
    hold_frames = int(round(args.max_hold_s * meta["fps"]))

    players = tracks[tracks["class"] == "player"].copy()
    # Roster-space teams with curation applied: ignored tracks (refs/
    # spectators) map to "ignored" and fall out of the isin filter below.
    pm = load_player_map(args.match_id, args.tmp_dir)
    track_teams = resolve_track_teams(players["track_id"].unique(), team_map, pm)
    players["team"] = players["track_id"].map(track_teams)
    if args.include_unidentified and "kit" in players.columns:
        loose = players["team"].eq("unknown") & players["kit"].isin(["team_a", "team_b"])
        players.loc[loose, "team"] = players.loc[loose, "kit"]
        print(f"Unidentified rows with a confident team kit allowed to hold possession: {int(loose.sum())}")
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

    f_lo = int(players["frame"].min()) if len(players) else 0
    f_hi = int(players["frame"].max()) if len(players) else meta["frame_count"] - 1
    frames = np.arange(f_lo, f_hi + 1)
    n = len(frames)
    cand = np.full(n, "", dtype=object)
    holder = np.full(n, -1, dtype=np.int64)
    has_ball = np.zeros(n, bool)
    bx, by = np.full(n, np.nan), np.full(n, np.nan)

    pf = players.sort_values("frame")
    p_frames = pf["frame"].to_numpy()
    p_x, p_y = pf["pitch_x"].to_numpy(), pf["pitch_y"].to_numpy()
    p_team, p_tid = pf["team"].to_numpy(), pf["track_id"].to_numpy()
    for ball in balls.itertuples():
        k = int(ball.frame) - f_lo
        if not 0 <= k < n:
            continue
        has_ball[k] = True
        bx[k], by[k] = ball.pitch_x, ball.pitch_y
        lo, hi = np.searchsorted(p_frames, ball.frame), np.searchsorted(p_frames, ball.frame, side="right")
        if hi == lo:
            continue
        d = np.hypot(p_x[lo:hi] - ball.pitch_x, p_y[lo:hi] - ball.pitch_y)
        j = int(np.argmin(d))
        if d[j] <= radius:
            cand[k], holder[k] = p_team[lo + j], int(p_tid[lo + j])

    # Hysteresis + hold: `state` is who has the ball after both rules.
    state = np.full(n, "none", dtype=object)
    attributed = {"team_a": 0, "team_b": 0}
    current, pending_team, pending_count, switches, since_obs = None, None, 0, 0, 0
    for k in range(n):
        c = cand[k] or None
        if c is not None:
            since_obs = 0
            if current is None:
                current = c
            elif c != current:
                pending_count = pending_count + 1 if pending_team == c else 1
                pending_team = c
                if pending_count >= switch_frames:
                    current, pending_team, pending_count = c, None, 0
                    switches += 1
            else:
                pending_team, pending_count = None, 0
            attributed[current] += 1
        else:
            since_obs += 1
            if since_obs > hold_frames:
                current, pending_team, pending_count = None, None, 0
        state[k] = current or "none"

    ball_frames = int(has_ball.sum())
    total_attr = attributed["team_a"] + attributed["team_b"]
    ball_rate = ball_frames / n if n else 0.0
    attr_rate = total_attr / ball_frames if ball_frames else 0.0
    held = {t: int((state == t).sum()) for t in ("team_a", "team_b")}
    known = held["team_a"] + held["team_b"]
    # Low confidence when the ball is rarely detected OR rarely near anyone —
    # a percentage built from a handful of attributed frames is noise.
    confidence = ("low" if ball_rate < 0.30 or attr_rate < 0.20
                  else ("medium" if calib is None else "high"))

    result = {
        "team_a_pct": round(100 * attributed["team_a"] / total_attr, 1) if total_attr else None,
        "team_b_pct": round(100 * attributed["team_b"] / total_attr, 1) if total_attr else None,
        "team_a_time_pct": round(100 * held["team_a"] / known, 1) if known else None,
        "team_b_time_pct": round(100 * held["team_b"] / known, 1) if known else None,
        "frames_with_known_possession": known,
        "attributed_frames": total_attr,
        "ball_frames": ball_frames,
        "total_frames": n,
        "window": [f_lo, f_hi],
        "ball_detection_rate": round(ball_rate, 4),
        "possession_switches": switches,
        "radius": radius,
        "max_hold_s": args.max_hold_s,
        "unit": unit,
        "calibrated": calib is not None,
        "confidence": confidence,
        "timeline": str(match_dir(args.match_id, args.tmp_dir) / "possession_timeline.parquet"),
    }
    out = write_json(match_dir(args.match_id, args.tmp_dir) / "possession.json", result)
    pd.DataFrame({"frame": frames, "state": state, "candidate": np.where(cand == "", "none", cand),
                  "holder": holder, "ball": has_ball, "ball_x": bx, "ball_y": by}).to_parquet(
        result["timeline"], index=False)

    if total_attr:
        print(f"Possession (ball-near-player frames): team_a {result['team_a_pct']}% — team_b "
              f"{result['team_b_pct']}% ({total_attr}/{ball_frames} ball frames attributed, {switches} switches)")
        print(f"Possession (time held, {args.max_hold_s:g}s hold): team_a {result['team_a_time_pct']}% — "
              f"team_b {result['team_b_time_pct']}% (known for {known / n:.0%} of the window)")
    else:
        print("Possession: could not attribute any frames (ball never near a team player)")
    print(f"Ball located in {ball_rate:.1%} of frames — confidence: {confidence}")
    print(f"Wrote {out} and {result['timeline']}")


if __name__ == "__main__":
    main()
