"""Hybrid detection over a match video → .tmp/<match_id>/tracks.parquet.

Two passes merged into one parquet (schema per the workflow):
  frame, t_s, track_id, class (player|ball), x1, y1, x2, y2, foot_x, foot_y, conf

- Players: YOLO `person` class + BoT-SORT (config/botsort_football.yaml:
  camera-motion compensation for panning footage, 3 s track buffer, ReID).
  --conf is deliberately low (0.1): the tracker's own thresholds gate quality
  (new tracks need conf ≥ 0.5; low-conf boxes may only extend existing
  tracks), and starving the tracker of low-conf boxes is what fragments IDs.
- Ball: separate *detection-only* pass — no tracker. Trackers silently drop
  detections below their new_track_thresh, which is exactly the confidence
  range a distant ball lives in; possession only needs per-frame positions,
  so ball rows carry track_id -1. Default is the pretrained model's
  `sports ball` class (COCO id 32); a fine-tuned ball model slots in via
  --ball-weights (with --ball-class for its class id).

Inference runs at --imgsz 1280 by default (not YOLO's 640): players on
pitch-scale footage are ~40 px tall and the ball ~10-20 px, and halving the
frame at inference time is the single biggest recall killer for both.

Coordinates are raw pixels; foot_x/foot_y is the bottom-center of the box.
Calibration is applied downstream at compute time, never baked in here.

Player IDs will still fragment across long occlusions — run
tools/stitch_tracks.py afterwards (cheap, artifact-only) before assign_teams.

This is the expensive stage — never re-run without checking with the user.

Usage:
    python tools/run_tracking.py --video match.mp4
    python tools/run_tracking.py --video match.mp4 \
        --ball-weights runs/detect/ball/weights/best.pt --ball-class 0
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

from _common import compute_match_id, match_dir, read_json

PERSON_CLASS = 0       # COCO `person`
SPORTS_BALL_CLASS = 32  # COCO `sports ball`
DEFAULT_TRACKER = Path(__file__).resolve().parent.parent / "config" / "botsort_football.yaml"


def probe_video(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count


def rows_from_boxes(res, frame_idx, class_name, fps, with_ids):
    """Rows for one frame's result. with_ids=False → track_id -1 (detect pass)."""
    rows = []
    if res.boxes is None or len(res.boxes) == 0:
        return rows
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    if with_ids and res.boxes.id is not None:
        track_ids = res.boxes.id.cpu().numpy().astype(int)
    else:
        track_ids = [-1] * len(boxes)
    for (x1, y1, x2, y2), tid, c in zip(boxes, track_ids, confs):
        rows.append({
            "frame": frame_idx,
            "t_s": frame_idx / fps,
            "track_id": int(tid),
            "class": class_name,
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "foot_x": float((x1 + x2) / 2),
            "foot_y": float(y2),
            "conf": float(c),
        })
    return rows


def track_pass(model, video, class_id, class_name, fps, conf, imgsz, tracker, device):
    """Tracking pass (players): boxes come back with persistent track IDs."""
    rows = []
    results = model.track(
        source=str(video), classes=[class_id], conf=conf, imgsz=imgsz,
        tracker=str(tracker), persist=True, stream=True, verbose=False, device=device,
    )
    for frame_idx, res in enumerate(results):
        rows.extend(rows_from_boxes(res, frame_idx, class_name, fps, with_ids=True))
    return rows


def detect_pass(model, video, class_id, class_name, fps, conf, imgsz, device):
    """Detection-only pass (ball): no tracker, so nothing below a tracker's
    new_track_thresh gets silently dropped. track_id is -1 on every row."""
    rows = []
    results = model.predict(
        source=str(video), classes=[class_id], conf=conf, imgsz=imgsz,
        stream=True, verbose=False, device=device,
    )
    for frame_idx, res in enumerate(results):
        rows.extend(rows_from_boxes(res, frame_idx, class_name, fps, with_ids=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to the match video")
    parser.add_argument("--match-id", default=None,
                        help="Override match_id (default: derived from the video, must match ingest_video.py)")
    parser.add_argument("--player-weights", default="yolov8n.pt", help="YOLO weights for the player pass")
    parser.add_argument("--ball-weights", default=None,
                        help="YOLO weights for the ball pass (default: same pretrained model, `sports ball` class)")
    parser.add_argument("--ball-class", type=int, default=None,
                        help="Class id for the ball in the ball weights (default: 32 for pretrained COCO, 0 for custom weights)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Player detection threshold fed to the tracker — keep low; "
                             "the tracker config's thresholds gate track quality")
    parser.add_argument("--ball-conf", type=float, default=0.1,
                        help="Ball confidence threshold (low — the ball is small and rarely confident)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference size. Match or exceed the video width — YOLO's default 640 "
                             "shrinks pitch-scale players/ball below reliable detection size")
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER),
                        help="Ultralytics tracker config (default: config/botsort_football.yaml, "
                             "camera-motion-compensated for panning footage)")
    parser.add_argument("--device", default=None, help="Torch device, e.g. 0 or cpu (default: auto)")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    from ultralytics import YOLO  # deferred: slow import

    video = Path(args.video)
    match_id = args.match_id or compute_match_id(video)
    out_dir = match_dir(match_id, args.tmp_dir, create=True)

    tracker = Path(args.tracker)
    if not tracker.exists():
        print(f"WARNING: tracker config {tracker} not found — falling back to bytetrack.yaml defaults")
        tracker = "bytetrack.yaml"

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        fps, frame_count = meta["fps"], meta["frame_count"]
    else:
        print(f"WARNING: {meta_path} not found — run ingest_video.py first next time; probing video directly")
        fps, frame_count = probe_video(video)

    print(f"[{match_id}] player pass ({args.player_weights}, class {PERSON_CLASS}, "
          f"imgsz {args.imgsz}, tracker {tracker})...")
    player_model = YOLO(args.player_weights)
    player_rows = track_pass(player_model, video, PERSON_CLASS, "player", fps,
                             args.conf, args.imgsz, tracker, args.device)

    ball_weights = args.ball_weights or args.player_weights
    ball_class = args.ball_class if args.ball_class is not None else (
        SPORTS_BALL_CLASS if args.ball_weights is None else 0)
    print(f"[{match_id}] ball pass ({ball_weights}, class {ball_class}, imgsz {args.imgsz}, detect-only)...")
    ball_model = YOLO(ball_weights)
    ball_rows = detect_pass(ball_model, video, ball_class, "ball", fps,
                            args.ball_conf, args.imgsz, args.device)

    df = pd.DataFrame(player_rows + ball_rows,
                      columns=["frame", "t_s", "track_id", "class",
                               "x1", "y1", "x2", "y2", "foot_x", "foot_y", "conf"])
    df = df.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)
    out_path = out_dir / "tracks.parquet"
    df.to_parquet(out_path, index=False)

    n_tracks = df.loc[df["class"] == "player", "track_id"].nunique()
    ball_frames = df.loc[df["class"] == "ball", "frame"].nunique()
    ball_rate = ball_frames / frame_count if frame_count else 0.0
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"  player tracks: {n_tracks}, player rows: {len(player_rows)}")
    print(f"  ball detection rate: {ball_rate:.1%} ({ball_frames}/{frame_count} frames)")
    if ball_rate < 0.30:
        print("WARNING: ball detected in < 30% of frames — possession stats will be low-confidence "
              "(consider fine-tuning a ball model via workflows/train_yolo_model.md)")
    print("Next: python tools/stitch_tracks.py --match-id " + match_id +
          "  (merges fragmented player IDs; run before assign_teams.py)")


if __name__ == "__main__":
    main()
