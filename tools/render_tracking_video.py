"""Render an annotated tracking video -> outputs/videos/<match_id>_tracking.mp4.

Overlays on every frame:
  - identified players: team-colored ellipse at the feet + a tag (player
    name when player_map.json has one - tools/track_identities.py writes
    role-inferred names - else the shirt number);
  - unidentified player tracks: thin gray box + #track_id (hide with
    --hide-unidentified) — these are exactly the tracks the identity stage
    couldn't place;
  - ball: yellow dot with a pointer (interpolated positions drawn hollow);
  - HUD with frame, video time, match clock and identified-box count.
Optional:
  - --minimap: pitch map (top-right) with every identified player's pitch
    position; players currently off-screen stay as hollow markers at their
    last known spot (fading), and the camera's field of view is shaded. The
    fastest way to watch whether identities stay with the same body.
  - --possession: banner with who holds the ball (possession_timeline.parquet).

Works with whatever artifacts exist: with no player_map.json it falls back to
team_map.json colors + track ids; with neither, everything is gray.

Optional --trail-s draws each player's recent foot path. On a panning camera
pixel-space trails mix camera motion with player motion — use the minimap or
compute_positions.py figures for true movement shape.

Encodes H.264 via PyAV when available (tools/_common.VideoWriter, ~5 MB/min
at 720p), else mp4v. Defaults to the identity window recorded in
player_map.json (e.g. the first half) when --start-s/--end-s aren't given.

Usage:
    python tools/render_tracking_video.py --match-id <match_id>
    python tools/render_tracking_video.py --match-id <match_id> --minimap --possession
    python tools/render_tracking_video.py --match-id <match_id> --start-s 60 --end-s 120 --trail-s 2
"""
import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _common import (OUTPUTS_DIR, Progress, VideoWriter, clock_label, ignored_track_ids, load_calibration,
                     load_meta, load_player_map, load_tracks, match_config, pitch_dims)

UNKNOWN_COLOR = (160, 160, 160)
IGNORED_COLOR = (90, 90, 90)
BALL_COLOR = (0, 215, 255)


def parse_bgr(s):
    b, g, r = (int(v) for v in s.split(","))
    return (b, g, r)


def draw_player(canvas, x1, x2, fx, fy, color, tag, thickness=2):
    w = max(int(x2 - x1), 8)
    cv2.ellipse(canvas, (int(fx), int(fy)), (int(0.55 * w), int(0.18 * w)),
                0, -35, 235, color, thickness, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    tx, ty = int(fx) - tw // 2, int(fy) + int(0.18 * w) + th + 5
    cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 3), color, -1)
    dark = sum(color) > 450  # light fill (e.g. white kit) -> dark text
    cv2.putText(canvas, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (20, 20, 20) if dark else (255, 255, 255), 1, cv2.LINE_AA)


def draw_unidentified(canvas, x1, y1, x2, y2, tag, color=UNKNOWN_COLOR):
    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    cv2.putText(canvas, tag, (int(x1), max(12, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_ball(canvas, fx, fy, interp=False):
    fx, fy = int(fx), int(fy)
    if interp:
        cv2.circle(canvas, (fx, fy), 6, BALL_COLOR, 1, cv2.LINE_AA)
        return
    cv2.circle(canvas, (fx, fy), 6, BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (fx, fy), 6, (0, 0, 0), 1, cv2.LINE_AA)
    pts = np.array([[fx - 7, fy - 22], [fx + 7, fy - 22], [fx, fy - 10]])
    cv2.drawContours(canvas, [pts], 0, BALL_COLOR, -1, cv2.LINE_AA)


class Minimap:
    """Small TV-view pitch map: identified players (filled = on screen now,
    hollow = last known position), ball, shaded camera view."""

    def __init__(self, L, W, scale=3.0, pad=6):
        self.L, self.W, self.s, self.pad = L, W, scale, pad
        self.w, self.h = int(L * scale) + 2 * pad, int(W * scale) + 2 * pad
        base = np.full((self.h, self.w, 3), (40, 110, 40), np.uint8)
        c = (230, 230, 230)
        P = self.p
        cv2.rectangle(base, P(0, 0), P(L, W), c, 1)
        cv2.line(base, P(L / 2, 0), P(L / 2, W), c, 1)
        cv2.circle(base, P(L / 2, W / 2), int(9.15 * scale), c, 1)
        for x0, s in ((0, 1), (L, -1)):
            cv2.rectangle(base, P(x0, W / 2 - 20.16), P(x0 + s * 16.5, W / 2 + 20.16), c, 1)
            cv2.rectangle(base, P(x0, W / 2 - 9.16), P(x0 + s * 5.5, W / 2 + 9.16), c, 1)
        self.base = base
        self.last = {}  # track_id -> (x, y, frame, color, tag)

    def p(self, x, y):
        return (int(self.pad + x * self.s), int(self.pad + y * self.s))

    def render(self, frame, visible, view_poly, ball_xy, fps):
        img = self.base.copy()
        if view_poly is not None and len(view_poly) >= 3:
            overlay = img.copy()
            pts = np.array([self.p(x, y) for x, y in view_poly], np.int32)
            cv2.fillPoly(overlay, [pts], (90, 160, 90))
            img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        for tid, (x, y, color, tag) in visible.items():
            self.last[tid] = (x, y, frame, color, tag)
        for tid, (x, y, f, color, tag) in self.last.items():
            age = (frame - f) / fps
            if age > 90:
                continue
            c = self.p(x, y)
            if tid in visible:
                cv2.circle(img, c, 4, color, -1, cv2.LINE_AA)
            else:
                faded = tuple(int(v * max(0.35, 1 - age / 90)) for v in color)
                cv2.circle(img, c, 4, faded, 1, cv2.LINE_AA)
            cv2.putText(img, tag, (c[0] + 4, c[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.27, color, 1, cv2.LINE_AA)
        if ball_xy is not None:
            cv2.circle(img, self.p(*ball_xy), 3, BALL_COLOR, -1, cv2.LINE_AA)
        return img


def view_polygon(H, w, h, L, W):
    """Image border projected onto the pitch (only points in front of the
    camera and near the pitch) -> convex polygon in pitch metres."""
    edge = np.concatenate([
        np.column_stack([np.linspace(0, w, 16), np.zeros(16)]),
        np.column_stack([np.full(16, w), np.linspace(0, h, 16)]),
        np.column_stack([np.linspace(w, 0, 16), np.full(16, h)]),
        np.column_stack([np.zeros(16), np.linspace(h, 0, 16)])])
    q = np.column_stack([edge, np.ones(len(edge))]) @ H.T
    ok = q[:, 2] > 1e-9
    xy = q[ok, :2] / q[ok, 2:3]
    xy = xy[(xy[:, 0] > -30) & (xy[:, 0] < L + 30) & (xy[:, 1] > -30) & (xy[:, 1] < W + 30)]
    if len(xy) < 3:
        return None
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
    return np.clip(hull, [-2, -2], [L + 2, W + 2])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None, help="Video path (default: from meta.json)")
    parser.add_argument("--output", default=None,
                        help="Default: outputs/videos/<match_id>_tracking.mp4")
    parser.add_argument("--start-s", type=float, default=None, help="Clip start (seconds of video)")
    parser.add_argument("--end-s", type=float, default=None, help="Clip end (seconds of video)")
    parser.add_argument("--scale", type=float, default=1.0, help="Output resolution scale")
    parser.add_argument("--trail-s", type=float, default=0.0,
                        help="Seconds of foot-path trail per identified player (0 = off)")
    parser.add_argument("--hide-unidentified", action="store_true",
                        help="Don't draw gray boxes for tracks without an identity")
    parser.add_argument("--show-ignored", action="store_true",
                        help="Draw ignored tracks (refs/spectators) dimmed instead of hiding them")
    parser.add_argument("--minimap", action="store_true", help="Pitch map with all identities (needs calibration)")
    parser.add_argument("--possession", action="store_true", help="Possession banner (possession_timeline.parquet)")
    parser.add_argument("--numbers", action="store_true", help="Tag players by shirt number even if names exist")
    parser.add_argument("--team-a-color", default="60,60,230", help="BGR, e.g. 60,60,230")
    parser.add_argument("--team-b-color", default="230,130,50", help="BGR")
    parser.add_argument("--crf", type=int, default=26)
    parser.add_argument("--calibration", default=None, help="Default: config/<match_id>_calibration.json")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    video = args.video or meta["video"]
    fps = meta["fps"]
    cfg = match_config(args.match_id, args.tmp_dir, args.config_dir)
    tracks = load_tracks(args.match_id, args.tmp_dir)

    team_color = {"team_a": parse_bgr(args.team_a_color), "team_b": parse_bgr(args.team_b_color)}

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
    else:
        print("No player_map.json — falling back to team colors/track ids")

    team_map = {}
    tm_path = Path(args.tmp_dir) / args.match_id / "team_map.json"
    if tm_path.exists():
        import json
        with open(tm_path, encoding="utf-8") as f:
            team_map = {int(k): v for k, v in json.load(f).items()}

    def tag_of(info):
        if info.get("name") and not args.numbers:
            return info["name"].split(" ")[-1]
        return str(info["number"])

    window = (pm or {}).get("window", {})
    start_f = (int(round(args.start_s * fps)) if args.start_s is not None
               else int(window.get("start_frame", 0)))
    end_f = (int(round(args.end_s * fps)) if args.end_s is not None
             else int(window.get("end_frame", meta["frame_count"] - 1)))
    end_f = min(end_f, meta["frame_count"] - 1)
    if start_f > end_f:
        raise SystemExit(f"--start-s/--end-s select an empty range (frames {start_f}..{end_f})")

    sel = tracks[(tracks["frame"] >= start_f) & (tracks["frame"] <= end_f)]
    cols = ["track_id", "x1", "y1", "x2", "y2", "foot_x", "foot_y"]
    has_pitch = "pitch_x" in sel.columns
    by_frame = {}
    for f, grp in sel.groupby("frame"):
        pl = grp[grp["class"] == "player"]
        bl = grp[grp["class"] == "ball"]
        by_frame[int(f)] = (pl[cols + (["pitch_x", "pitch_y"] if has_pitch else [])].to_numpy(),
                            bl[["foot_x", "foot_y"] + (["pitch_x", "pitch_y"] if has_pitch else [])
                               + (["interp"] if "interp" in bl.columns else [])].to_numpy())

    mini, H_of, L, W = None, {}, 105.0, 68.0
    if args.minimap:
        calib = load_calibration(args.match_id, args.config_dir, args.calibration)
        if calib is None or not has_pitch:
            print("WARNING: --minimap needs a calibration and pitch_x/pitch_y in tracks - skipped")
        else:
            L, W = pitch_dims(calib)
            if "frame_H" in calib:
                H_of = dict(zip(calib["frame_index"].tolist(), calib["frame_H"]))
            mini = Minimap(L, W)
    poss = {}
    if args.possession:
        tl_path = Path(args.tmp_dir) / args.match_id / "possession_timeline.parquet"
        if tl_path.exists():
            t = pd.read_parquet(tl_path, columns=["frame", "state"])
            poss = dict(zip(t["frame"].tolist(), t["state"].tolist()))
        else:
            print("WARNING: --possession needs possession_timeline.parquet (compute_possession.py) - skipped")
    team_names = {"team_a": cfg.get("team_a_name", "team_a"), "team_b": cfg.get("team_b_name", "team_b")}

    out_path = Path(args.output) if args.output else (OUTPUTS_DIR / "videos" / f"{args.match_id}_tracking.mp4")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open {video}")
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    size = (int(meta["width"] * args.scale), int(meta["height"] * args.scale))
    writer = VideoWriter(out_path, fps, size[0], size[1], crf=args.crf)

    trail_n = int(round(args.trail_s * fps))
    trails = defaultdict(lambda: deque(maxlen=max(trail_n, 1)))

    n_written = 0
    prog = Progress(end_f - start_f + 1, "rendering")
    for frame_idx in range(start_f, end_f + 1):
        prog.update(frame_idx - start_f)
        ok, canvas = cap.read()
        if not ok:
            print(f"WARNING: video ended early at frame {frame_idx}")
            break
        pl, bl = by_frame.get(frame_idx, (np.empty((0, 9)), np.empty((0, 4))))
        n_ident = n_unident = 0
        visible = {}
        for row in pl:
            tid = int(row[0])
            x1, y1, x2, y2, fx, fy = row[1:7]
            if tid in ignored:
                if args.show_ignored:
                    draw_unidentified(canvas, x1, y1, x2, y2, f"ign#{tid}", IGNORED_COLOR)
                continue
            key = track_player.get(tid)
            if key is not None:
                info = player_info[key]
                color = team_color[info["team"]]
                if trail_n:
                    trails[key].append((int(fx), int(fy)))
                    pts = np.array(trails[key], dtype=np.int32)
                    if len(pts) >= 2:
                        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
                draw_player(canvas, x1, x2, fx, fy, color, tag_of(info))
                if mini is not None:
                    visible[tid] = (float(row[7]), float(row[8]), color, str(info["number"]))
                n_ident += 1
            else:
                n_unident += 1
                if not args.hide_unidentified:
                    team = team_map.get(tid, "unknown")
                    if pm is None and team in team_color:
                        draw_player(canvas, x1, x2, fx, fy, team_color[team], f"#{tid}")
                    else:
                        draw_unidentified(canvas, x1, y1, x2, y2, f"#{tid}")
        ball_xy = None
        for b in bl:
            # interp is True / NaN (real detection) - NaN is truthy, so compare explicitly
            interp = bool(b[-1] == True) if "interp" in sel.columns and len(b) >= 5 else False  # noqa: E712
            draw_ball(canvas, b[0], b[1], interp)
            if has_pitch and len(b) >= 4 and np.isfinite(b[2]):
                ball_xy = (float(b[2]), float(b[3]))

        clk = clock_label(frame_idx, cfg, fps)
        named = (not args.numbers) and any(v.get("name") for v in player_info.values())
        hud = (f"frame {frame_idx}  video {int(frame_idx / fps // 60):02d}:{frame_idx / fps % 60:05.2f}"
               + (f"  match {clk}" if clk else "") + f"  identified {n_ident}  other {n_unident}"
               + ("  | names role-inferred" if named else ""))
        cv2.rectangle(canvas, (0, canvas.shape[0] - 26), (760, canvas.shape[0]), (0, 0, 0), -1)
        cv2.putText(canvas, hud, (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if poss:
            st = poss.get(frame_idx, "none")
            txt = f"{team_names[st]} in possession" if st in team_names else "possession: unknown"
            col = team_color.get(st, (60, 60, 60))
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(canvas, (canvas.shape[1] // 2 - tw // 2 - 8, 6),
                          (canvas.shape[1] // 2 + tw // 2 + 8, 14 + th), col, -1)
            cv2.putText(canvas, txt, (canvas.shape[1] // 2 - tw // 2, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (20, 20, 20) if sum(col) > 450 else (255, 255, 255), 1, cv2.LINE_AA)
        if mini is not None:
            H = H_of.get(frame_idx)
            poly = view_polygon(H, meta["width"], meta["height"], L, W) if H is not None else None
            m = mini.render(frame_idx, visible, poly, ball_xy, fps)
            x0 = canvas.shape[1] - m.shape[1] - 8
            canvas[8:8 + m.shape[0], x0:x0 + m.shape[1]] = m

        if args.scale != 1.0:
            canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
        writer.write(canvas)
        n_written += 1
        if n_written % 2500 == 0:
            print(f"  {n_written}/{end_f - start_f + 1} frames...", flush=True)

    cap.release()
    writer.close()
    prog.done()
    print(f"Wrote {out_path} ({n_written} frames @ {fps:g} fps, {size[0]}x{size[1]}, {writer.codec})")


if __name__ == "__main__":
    main()
