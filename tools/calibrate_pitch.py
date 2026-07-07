"""Interactive pitch calibration → config/<match_id>_calibration.json.

Shows a frame of the video in an OpenCV window. It steps through a fixed
menu of standard pitch landmarks one at a time, starting from #1; the current
landmark's name is shown in a banner at the top of the window before you do
anything, so you always know what you're about to click. For each landmark:
  - CLICK it in the image to record it and advance to the next landmark.
  - SPACE skips the current landmark (e.g. it's off-screen) and advances.
  - BACKSPACE undoes the previous step (whether clicked or skipped).
  - ENTER finishes calibration early once 4+ points are recorded (spread
    them out — all four pitch corners is the best minimum set) and computes
    the homography (image pixels → pitch meters) with cv2.findHomography.
  - Q aborts without saving.
If you reach the end of the menu with fewer than 4 points recorded, you'll be
asked to keep going (BACKSPACE/click/space still work) until you hit 4.

Only part of the pitch needs to be visible: any 4+ landmarks from the menu
that you can see (and that aren't nearly collinear) are enough, whatever the
camera angle.

Two modes:
- Static camera (`--frame N`, the default): one frame, one homography,
  `"mode": "static"`. Runs once per camera setup, not per match — reuse an
  existing calibration file (compute tools accept --calibration) if the
  camera hasn't moved.
- Moving camera (`--frames N1,N2,...`): repeat the click flow on several
  keyframes spread across the camera's range of motion, `"mode": "keyframes"`.
  Afterwards run tools/estimate_camera_motion.py to fill in every frame in
  between; pick keyframes where the camera view differs most (e.g. panned
  fully left / center / fully right). 2-5 keyframes is typically enough for
  a panning camera.

Pitch coordinate system: origin at the top-left corner as seen on a pitch
map, x along the length (0..105 m), y along the width (0..68 m). Override
dimensions with --pitch-length/--pitch-width if the pitch isn't full-size.

Requires a desktop session (opens a GUI window).

Usage:
    python tools/calibrate_pitch.py --match-id <match_id> --frame 100
    python tools/calibrate_pitch.py --match-id <match_id> --frames 0,900,1800
"""
import argparse

import cv2
import numpy as np

from _common import calibration_path, load_meta, project_points, write_json


def landmark_menu(L, W):
    """(name, (pitch_x, pitch_y)) for standard pitch points. Penalty area:
    16.5 m deep × 40.32 m wide; goal area: 5.5 × 18.32; spots at 11 m;
    center circle radius 9.15 m. 'left'/'right' = as seen in the camera image
    if the pitch map is oriented the same way; 'top'/'bottom' = near/far
    touchline depends on camera side — just be consistent."""
    pw, gw, cr = 40.32 / 2, 18.32 / 2, 9.15
    return [
        ("Corner: top-left", (0, 0)),
        ("Corner: top-right", (L, 0)),
        ("Corner: bottom-left", (0, W)),
        ("Corner: bottom-right", (L, W)),
        ("Halfway line: top touchline", (L / 2, 0)),
        ("Halfway line: bottom touchline", (L / 2, W)),
        ("Center spot", (L / 2, W / 2)),
        ("Center circle: top intersection with halfway line", (L / 2, W / 2 - cr)),
        ("Center circle: bottom intersection with halfway line", (L / 2, W / 2 + cr)),
        ("Center circle: leftmost point", (L / 2 - cr, W / 2)),
        ("Center circle: rightmost point", (L / 2 + cr, W / 2)),
        ("Left penalty area: top-front corner", (16.5, W / 2 - pw)),
        ("Left penalty area: bottom-front corner", (16.5, W / 2 + pw)),
        ("Left penalty area: top corner on goal line", (0, W / 2 - pw)),
        ("Left penalty area: bottom corner on goal line", (0, W / 2 + pw)),
        ("Right penalty area: top-front corner", (L - 16.5, W / 2 - pw)),
        ("Right penalty area: bottom-front corner", (L - 16.5, W / 2 + pw)),
        ("Right penalty area: top corner on goal line", (L, W / 2 - pw)),
        ("Right penalty area: bottom corner on goal line", (L, W / 2 + pw)),
        ("Left goal area: top-front corner", (5.5, W / 2 - gw)),
        ("Left goal area: bottom-front corner", (5.5, W / 2 + gw)),
        ("Right goal area: top-front corner", (L - 5.5, W / 2 - gw)),
        ("Right goal area: bottom-front corner", (L - 5.5, W / 2 + gw)),
        ("Left penalty spot", (11, W / 2)),
        ("Right penalty spot", (L - 11, W / 2)),
    ]


def read_frame(video, frame_idx):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Failed to read frame {frame_idx} from {video}")
    return frame


def print_menu(menu):
    print("Standard pitch landmarks:")
    for i, (name, (px, py)) in enumerate(menu):
        print(f"  {i:2d}. {name}  ({px:.2f}, {py:.2f}) m")
    print()


def collect_points(frame, frame_idx, menu):
    """Interactive click loop on one frame; steps through `menu` in order,
    starting at #1. Returns the recorded points list."""
    click = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    win = f"calibrate_pitch — frame {frame_idx} — click landmarks"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    points = []  # {"landmark", "image", "pitch"}
    history = []  # one entry per advance: True if it recorded a point, False if skipped
    idx = 0

    def redraw(banner):
        canvas = frame.copy()
        for p in points:
            x, y = map(int, p["image"])
            cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(canvas, p["landmark"], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(canvas, banner[0], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(canvas, banner[1], (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow(win, canvas)

    print_menu(menu)
    print("CLICK the landmark named in the window banner to record it and advance.")
    print("SPACE = skip this landmark   BACKSPACE = undo   ENTER = finish (once 4+ points)   Q = abort\n")

    while True:
        if idx < len(menu):
            name, pitch_xy = menu[idx]
            banner = (
                f"[{idx + 1}/{len(menu)}] Click: {name}",
                f"{len(points)} recorded  |  SPACE=skip  BACKSPACE=undo  ENTER=finish (4+)  Q=abort",
            )
        else:
            name, pitch_xy = None, None
            banner = (
                f"All {len(menu)} landmarks done — {len(points)} point(s) recorded",
                "Need 4+ to finish: BACKSPACE to undo and add more, ENTER to finish, Q to abort"
                if len(points) < 4 else
                "BACKSPACE to undo, ENTER to finish, Q to abort",
            )
        redraw(banner)

        key = cv2.waitKey(30) & 0xFF

        if idx < len(menu) and click["pt"] is not None:
            pt = click["pt"]
            click["pt"] = None
            points.append({"landmark": name, "image": list(pt), "pitch": list(pitch_xy)})
            history.append(True)
            print(f"  [{idx + 1}/{len(menu)}] {name}: recorded image {pt} -> pitch {pitch_xy}")
            idx += 1
            continue
        click["pt"] = None

        if key == 255:  # no key pressed this tick
            continue
        if key == ord("q") or key == ord("Q"):
            cv2.destroyAllWindows()
            raise SystemExit("Aborted — no calibration written")
        if key == 32:  # space: skip current landmark
            if idx < len(menu):
                print(f"  [{idx + 1}/{len(menu)}] {name}: skipped")
                history.append(False)
                idx += 1
            continue
        if key in (8, 127):  # backspace: undo last advance
            if history:
                was_recorded = history.pop()
                idx -= 1
                if was_recorded:
                    points.pop()
                print(f"  Undid landmark #{idx + 1}")
            continue
        if key in (13, 10):  # enter: finish early
            if len(points) >= 4:
                break
            print(f"Need at least 4 points to finish (have {len(points)}).")
            continue

    cv2.destroyAllWindows()
    return points


def solve_homography(points):
    """Homography + reprojection error (in meters) from clicked points."""
    src = np.array([p["image"] for p in points], dtype=np.float64)
    dst = np.array([p["pitch"] for p in points], dtype=np.float64)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC if len(points) > 4 else 0)
    if H is None:
        raise SystemExit("cv2.findHomography failed — points may be collinear; pick more spread-out landmarks")
    reproj = project_points(H, src)
    err = np.linalg.norm(reproj - dst, axis=1)
    print(f"Reprojection error: mean {err.mean():.2f} m, max {err.max():.2f} m")
    if err.mean() > 2.0:
        print("WARNING: mean error > 2 m — clicks or landmark choices are probably off; consider redoing")
    return H, err


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None, help="Video path (default: from meta.json)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to calibrate on (static camera)")
    parser.add_argument("--frames", default=None,
                        help="Comma-separated keyframe indices for a moving camera, e.g. 0,900,1800")
    parser.add_argument("--pitch-length", type=float, default=105.0)
    parser.add_argument("--pitch-width", type=float, default=68.0)
    parser.add_argument("--output", default=None, help="Default: config/<match_id>_calibration.json")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    video = args.video or load_meta(args.match_id, args.tmp_dir)["video"]
    frame_indices = ([int(f) for f in args.frames.split(",")] if args.frames else [args.frame])
    frame_indices = sorted(set(frame_indices))

    menu = landmark_menu(args.pitch_length, args.pitch_width)
    if len(frame_indices) > 1:
        print(f"Moving-camera mode: you will calibrate {len(frame_indices)} keyframes: {frame_indices}\n")

    keyframes = []
    for frame_idx in frame_indices:
        if len(frame_indices) > 1:
            print(f"--- Keyframe {frame_idx} ---")
        frame = read_frame(video, frame_idx)
        points = collect_points(frame, frame_idx, menu)
        H, err = solve_homography(points)
        keyframes.append({
            "frame": frame_idx,
            "homography": H.tolist(),
            "points": points,
            "reprojection_error_m": {"mean": float(err.mean()), "max": float(err.max())},
        })

    out_path = args.output or calibration_path(args.match_id, args.config_dir)
    if len(keyframes) == 1:
        kf = keyframes[0]
        payload = {
            "mode": "static",
            "pitch_length": args.pitch_length,
            "pitch_width": args.pitch_width,
            "homography": kf["homography"],
            "points": kf["points"],
            "video": str(video),
            "frame": kf["frame"],
            "reprojection_error_m": kf["reprojection_error_m"],
        }
    else:
        payload = {
            "mode": "keyframes",
            "pitch_length": args.pitch_length,
            "pitch_width": args.pitch_width,
            "video": str(video),
            "keyframes": keyframes,
        }
    write_json(out_path, payload)
    print(f"Wrote {out_path}")
    if len(keyframes) > 1:
        print("Next: run tools/estimate_camera_motion.py to compute per-frame homographies "
              "before any compute_* tool can use this calibration.")


if __name__ == "__main__":
    main()
