"""Run a trained YOLO tracker over a video, drawing per-team boxes and
(optionally) movement trajectories. Merges the "Tracking Player on the
Field" and "Tracking the Trajectory of the Player" stages of the notebook
into one parameterized pass.

Usage:
    python tools/track_players.py --weights runs/detect/train/weights/last.pt \
        --video ./Video/2.mp4 --output ./Video/2_tracked.mp4 \
        --team-colors '{"0": [0,0,255], "4": [255,0,0]}' --trajectories
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO


def parse_team_colors(raw):
    """raw: JSON object mapping class_id -> [B, G, R], e.g. {"0": [0,0,255], "4": [255,0,0]}."""
    mapping = json.loads(raw)
    return {int(k): tuple(v) for k, v in mapping.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument(
        "--team-colors",
        default=os.environ.get("TEAM_COLORS", '{"0": [0, 0, 255], "4": [255, 0, 0]}'),
        help='JSON mapping of class_id -> BGR color. Class IDs must match your annotation template.',
    )
    parser.add_argument("--trajectories", action="store_true", help="Draw movement trails in addition to boxes")
    args = parser.parse_args()

    team_colors = parse_team_colors(args.team_colors)
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    trajectories = defaultdict(list)
    results = model.track(source=args.video, persist=True, stream=True)

    for res in results:
        frame = res.orig_img.copy()

        if res.boxes.id is None:
            out.write(frame)
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        track_ids = res.boxes.id.cpu().numpy().astype(int)
        class_ids = res.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), tid, cid in zip(boxes, track_ids, class_ids):
            if cid not in team_colors:
                continue

            color = team_colors[cid]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Player {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if args.trajectories:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trajectories[tid].append((cx, cy))
                for j in range(1, len(trajectories[tid])):
                    cv2.line(frame, trajectories[tid][j - 1], trajectories[tid][j], color, 2)

        out.write(frame)

    out.release()
    print(f"Tracking video saved to: {args.output}")


if __name__ == "__main__":
    main()
