"""Aggregate per-class player positions over a full video into a colorized
heatmap, overlaid on a sample frame.

Usage:
    python tools/generate_heatmap.py --weights runs/detect/train/weights/last.pt \
        --video ./Video/4.mp4 --output class_heatmaps_overlay.png \
        --classes "[0, 4]" --sample-frame 100
"""
import argparse
import json

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="class_heatmaps_overlay.png")
    parser.add_argument("--classes", default="[0, 4]", help="JSON list of class IDs to include, e.g. [0, 4]")
    parser.add_argument("--sample-frame", type=int, default=100, help="Frame index to use as the overlay background")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap blend opacity")
    args = parser.parse_args()

    classes_of_interest = json.loads(args.classes)
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    heatmaps = {cls: np.zeros((height, width), dtype=np.float32) for cls in classes_of_interest}

    results = model.track(source=args.video, persist=True, stream=True)
    for res in results:
        if res.boxes.id is None:
            continue

        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = res.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), cid in zip(boxes, class_ids):
            if cid not in classes_of_interest:
                continue
            heatmaps[cid][y1:y2, x1:x2] += 1

    colored_maps = {}
    for cid, hm in heatmaps.items():
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_maps[cid] = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.sample_frame)
    ret, base = cap.read()
    cap.release()
    if not ret:
        raise SystemExit(f"Failed to read frame {args.sample_frame} from {args.video}")

    overlay = base.copy()
    for cmap in colored_maps.values():
        cv2.addWeighted(cmap, args.alpha, overlay, 1 - args.alpha, 0, overlay)

    cv2.imwrite(args.output, overlay)
    print(f"Saved overlay image with heatmaps: {args.output}")


if __name__ == "__main__":
    main()
