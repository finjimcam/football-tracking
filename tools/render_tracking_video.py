"""Render an annotated tracking video -> outputs/videos/<match_id>_tracking.mp4.

Overlays on every frame:
  - identified players: team-colored ellipse at the feet + shirt number tag
    (from player_map.json, i.e. label_roster.py + assign_identities.py);
  - unidentified player tracks: thin gray box + #track_id (hide with
    --hide-unidentified) — these are exactly the tracks assign_identities.py
    couldn't place, so gray boxes on a real player tell you where to add a
    roster anchor;
  - ball detections: yellow dot with a pointer;
  - HUD with frame/time and identified-box count.

Works with whatever artifacts exist: with no player_map.json it falls back to
team_map.json colors + track ids (useful for debugging raw tracking); with
neither, everything is gray. It's a debugging surface first (watch for
identity jumps, then re-anchor) and a tactical-video base second.

Optional --trail-s draws each player's recent foot path. Note that on a
panning camera pixel-space trails mix camera motion with player motion —
they're honest per-frame, but a long trail will look bent when the camera
moves. Use compute_positions.py pitch-space figures for true movement shape.

Uses the mp4v codec (universally available in OpenCV builds on Windows). The
whole video is decoded and re-encoded: roughly real-time on 720p, so a full
match takes about match length — use --start-s/--end-s to cut a clip.

Usage:
    python tools/render_tracking_video.py --match-id <match_id>
    python tools/render_tracking_video.py --match-id <match_id> \
        --start-s 60 --end-s 120 --trail-s 2
"""
import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from _common import OUTPUTS_DIR, Progress, ignored_track_ids, load_meta, load_player_map, load_tracks

UNKNOWN_COLOR = (160, 160, 160)
IGNORED_COLOR = (90, 90, 90)
BALL_COLOR = (0, 215, 255)


def parse_bgr(s):
    b, g, r = (int(v) for v in s.split(","))
    return (b, g, r)


def draw_player(canvas, row, color, tag, selected_thickness=2):
    x1, y1, x2, y2 = (int(row[c]) for c in ("x1", "y1", "x2", "y2"))
    fx, fy = int(row["foot_x"]), int(row["foot_y"])
    w = max(x2 - x1, 8)
    cv2.ellipse(canvas, (fx, fy), (int(0.55 * w), int(0.18 * w)),
                0, -35, 235, color, selected_thickness, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    tx, ty = fx - tw // 2, fy + int(0.18 * w) + th + 6
    cv2.rectangle(canvas, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), color, -1)
    cv2.putText(canvas, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_unidentified(canvas, row, tag, color=UNKNOWN_COLOR):
    x1, y1, x2, y2 = (int(row[c]) for c in ("x1", "y1", "x2", "y2"))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
    cv2.putText(canvas, tag, (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_ball(canvas, row):
    fx, fy = int((row["x1"] + row["x2"]) / 2), int(row["foot_y"])
    cv2.circle(canvas, (fx, fy), 6, BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (fx, fy), 6, (0, 0, 0), 1, cv2.LINE_AA)
    pts = np.array([[fx - 7, fy - 22], [fx + 7, fy - 22], [fx, fy - 10]])
    cv2.drawContours(canvas, [pts], 0, BALL_COLOR, -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None, help="Video path (default: from meta.json)")
    parser.add_argument("--output", default=None,
                        help="Default: outputs/videos/<match_id>_tracking.mp4")
    parser.add_argument("--start-s", type=float, default=None, help="Clip start (seconds)")
    parser.add_argument("--end-s", type=float, default=None, help="Clip end (seconds)")
    parser.add_argument("--scale", type=float, default=1.0, help="Output resolution scale")
    parser.add_argument("--trail-s", type=float, default=0.0,
                        help="Seconds of foot-path trail per identified player (0 = off)")
    parser.add_argument("--hide-unidentified", action="store_true",
                        help="Don't draw gray boxes for tracks without an identity")
    parser.add_argument("--show-ignored", action="store_true",
                        help="Draw ignored tracks (refs/spectators) dimmed instead of hiding them")
    parser.add_argument("--team-a-color", default="60,60,230", help="BGR, e.g. 60,60,230")
    parser.add_argument("--team-b-color", default="230,130,50", help="BGR")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    video = args.video or meta["video"]
    fps = meta["fps"]
    tracks = load_tracks(args.match_id, args.tmp_dir)

    team_color = {"team_a": parse_bgr(args.team_a_color),
                  "team_b": parse_bgr(args.team_b_color)}

    pm = load_player_map(args.match_id, args.tmp_dir)
    track_player, player_info = {}, {}
    ignored = ignored_track_ids(pm)
    if pm:
        track_player = pm["track_to_player"]
        player_info = pm["players"]
        names = pm.get("team_names", {})
        print(f"Using player_map.json: {len(player_info)} players "
              f"({names.get('team_a', 'team_a')} vs {names.get('team_b', 'team_b')}), "
              f"{pm['identified_detection_fraction']:.0%} of detections identified")
        if ignored:
            print(f"{len(ignored)} ignored track(s) "
                  + ("drawn dimmed (--show-ignored)" if args.show_ignored else
                     "hidden (--show-ignored to draw them dimmed)"))
    else:
        print("No player_map.json — falling back to team colors/track ids "
              "(run label_roster.py + assign_identities.py for numbered players)")

    team_map = {}
    tm_path = Path(args.tmp_dir) / args.match_id / "team_map.json"
    if tm_path.exists():
        import json
        with open(tm_path, encoding="utf-8") as f:
            team_map = {int(k): v for k, v in json.load(f).items()}

    start_f = int(round((args.start_s or 0) * fps))
    end_f = int(round(args.end_s * fps)) if args.end_s is not None else meta["frame_count"] - 1
    end_f = min(end_f, meta["frame_count"] - 1)
    if start_f > end_f:
        raise SystemExit(f"--start-s/--end-s select an empty range (frames {start_f}..{end_f})")

    sel = tracks[(tracks["frame"] >= start_f) & (tracks["frame"] <= end_f)]
    by_frame = {int(f): grp for f, grp in sel.groupby("frame")}

    out_path = Path(args.output) if args.output else (
        OUTPUTS_DIR / "videos" / f"{args.match_id}_tracking.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open {video}")
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    size = (int(meta["width"] * args.scale), int(meta["height"] * args.scale))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise SystemExit(f"cv2.VideoWriter failed to open {out_path} (mp4v)")

    trail_n = int(round(args.trail_s * fps))
    trails = defaultdict(lambda: deque(maxlen=max(trail_n, 1)))  # player key -> foot points

    n_written = 0
    prog = Progress(end_f - start_f + 1, "rendering")
    for frame_idx in range(start_f, end_f + 1):
        prog.update(frame_idx - start_f)
        ok, canvas = cap.read()
        if not ok:
            print(f"WARNING: video ended early at frame {frame_idx}")
            break
        rows = by_frame.get(frame_idx)
        n_ident = n_unident = 0
        if rows is not None:
            players = rows[(rows["class"] == "player") & (rows["track_id"] >= 0)]
            for _, row in players.iterrows():
                tid = int(row["track_id"])
                if tid in ignored:
                    if args.show_ignored:
                        draw_unidentified(canvas, row, f"ign#{tid}", IGNORED_COLOR)
                    continue
                key = track_player.get(tid)
                if key is not None:
                    info = player_info[key]
                    color = team_color[info["team"]]
                    if trail_n:
                        trails[key].append((int(row["foot_x"]), int(row["foot_y"])))
                        pts = np.array(trails[key], dtype=np.int32)
                        if len(pts) >= 2:
                            cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
                    draw_player(canvas, row, color, str(info["number"]))
                    n_ident += 1
                else:
                    n_unident += 1
                    if not args.hide_unidentified:
                        team = team_map.get(tid, "unknown")
                        color = team_color.get(team, UNKNOWN_COLOR)
                        if pm is None and team in team_color:
                            draw_player(canvas, row, color, f"#{tid}")
                        else:
                            draw_unidentified(canvas, row, f"#{tid}")
            for _, row in rows[rows["class"] == "ball"].iterrows():
                draw_ball(canvas, row)

        hud = f"frame {frame_idx}  t={frame_idx / fps:6.2f}s  identified {n_ident}  other {n_unident}"
        cv2.rectangle(canvas, (0, canvas.shape[0] - 26), (430, canvas.shape[0]), (0, 0, 0), -1)
        cv2.putText(canvas, hud, (8, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if args.scale != 1.0:
            canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
        writer.write(canvas)
        n_written += 1
        if n_written % 500 == 0:
            print(f"  {n_written}/{end_f - start_f + 1} frames...")

    cap.release()
    writer.release()
    prog.done()
    print(f"Wrote {out_path} ({n_written} frames @ {fps:g} fps, {size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
