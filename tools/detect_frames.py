"""Person + ball detection over a frame window -> .tmp/<match_id>/detections.parquet.

The ONE expensive stage of the identity-tracking pipeline
(workflows/track_identities.md). Everything after it - calibration, tracking,
identity linking, possession, verification, clips - reads these cached
detections, so the association logic can be re-tuned and re-run in minutes
without touching the detector. That is what makes "put the footage back
through the tracking" cheap.

Per frame: YOLO person + `sports ball` boxes at full resolution. Per person
box: kit colour features (median CIE-Lab of a shirt band and a shorts band,
grass pixels masked out), so team / goalkeeper / referee classification never
needs to decode the video again.

Schema (one row per detection):
    frame, t_s, class (player|ball), x1, y1, x2, y2, foot_x, foot_y, conf,
    shirt_L, shirt_a, shirt_b, shirt_n, shorts_L, shorts_a, shorts_b, shorts_n
Kit columns are NaN (and *_n = 0) for balls or when too few non-grass pixels.
Lab values are OpenCV 8-bit Lab (L 0-255, a/b offset by 128).

Resumable: rows are written in chunks (.tmp/<match_id>/detections/
chunk_<first frame>.parquet); a re-run skips finished chunks. The final merge
writes detections.parquet plus detections_meta.json (weights, device, window,
speed). Ball conf is kept deliberately low (0.05): the ball is 4-8 px on a
tactical camera, and false positives (spare balls on the perimeter plinths,
white boots) are filtered downstream with pitch calibration and motion,
which a confidence threshold can't do.

Device: `--device intel:gpu` runs OpenVINO on the Intel iGPU (yolo26n at
1280x736: ~36 ms/frame on a Core Ultra 5 135U vs ~115 ms with CPU torch, same
boxes). The OpenVINO export is created once under .tmp/models/ (disposable).
`--device cpu` (torch) works everywhere, just slower.

Usage:
    python tools/detect_frames.py --match-id <id>            # window from meta config
    python tools/detect_frames.py --match-id <id> --start-frame 1151 --end-frame 71434
    python tools/detect_frames.py --match-id <id> --limit 500   # quick test
"""
import argparse
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _common import Progress, load_meta, match_dir, read_json, write_json

PERSON_CLASS = 0        # COCO person
SPORTS_BALL_CLASS = 32  # COCO sports ball
KIT_COLS = ["shirt_L", "shirt_a", "shirt_b", "shirt_n",
            "shorts_L", "shorts_a", "shorts_b", "shorts_n"]
COLUMNS = (["frame", "t_s", "class", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "conf"]
           + KIT_COLS)
MODELS_DIR = Path(".tmp") / "models"


def model_input_size(width, height, imgsz):
    """(h, w) multiple-of-32 inference size keeping the video aspect ratio,
    with the long side = imgsz. 1280x720 @ 1280 -> (736, 1280)."""
    scale = imgsz / max(width, height)
    return (int(math.ceil(height * scale / 32) * 32), int(math.ceil(width * scale / 32) * 32))


def resolve_device(device):
    """'auto' -> '0' with CUDA (e.g. the pphub GTX 1060), else 'intel:gpu'
    when OpenVINO is installed (the dev laptop's iGPU), else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    try:
        import openvino  # noqa: F401
        return "intel:gpu"
    except ImportError:
        return "cpu"


def load_model(weights, device, input_hw):
    """YOLO model for `device`. intel:* devices use an OpenVINO export with the
    input size baked in, cached under .tmp/models/<stem>_<h>x<w>_openvino_model
    (ultralytics recognises the format by that directory suffix)."""
    from ultralytics import YOLO  # deferred: slow import

    if not (device or "").startswith("intel:"):
        return YOLO(weights)
    w = Path(weights)
    ov_dir = MODELS_DIR / f"{w.stem}_{input_hw[0]}x{input_hw[1]}_openvino_model"
    if not (ov_dir / f"{w.stem}.xml").exists():
        print(f"Exporting {weights} to OpenVINO at {input_hw} (one-off)...")
        exported = Path(YOLO(weights).export(format="openvino", imgsz=input_hw,
                                             dynamic=False, verbose=False))
        ov_dir.parent.mkdir(parents=True, exist_ok=True)
        if ov_dir.exists():
            shutil.rmtree(ov_dir)
        shutil.move(str(exported), str(ov_dir))
    return YOLO(str(ov_dir), task="detect")


def band_lab(crop_lab, grass, t0, t1, shrink):
    """Median Lab over a horizontal band of a person crop (fractions of its
    height), side margins trimmed, grass pixels excluded."""
    h, w = grass.shape
    ya, yb = int(t0 * h), max(int(t1 * h), int(t0 * h) + 1)
    xa, xb = int(shrink * w), max(int((1 - shrink) * w), int(shrink * w) + 1)
    keep = ~grass[ya:yb, xa:xb]
    pix = crop_lab[ya:yb, xa:xb][keep]
    if len(pix) < 4:
        return (np.nan, np.nan, np.nan, 0)
    m = np.median(pix, axis=0)
    return (float(m[0]), float(m[1]), float(m[2]), int(len(pix)))


def kit_features(frame, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = frame.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 3 or y2 - y1 < 6:
        return (np.nan, np.nan, np.nan, 0) * 2
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = ((hsv[..., 0] > 33) & (hsv[..., 0] < 88) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40))
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return band_lab(lab, grass, 0.15, 0.45, 0.2) + band_lab(lab, grass, 0.48, 0.68, 0.2)


def class_ids(names):
    """(person, ball) class ids from a model's names: COCO (0, 32) for the
    pretrained weights, (0, 1) for one fine-tuned on person + ball (#7)."""
    inv = {str(v).lower(): int(k) for k, v in dict(names).items()}
    person = inv.get("person")
    ball = inv.get("sports ball", inv.get("ball"))
    if person is None or ball is None:
        raise SystemExit(f"Model classes {dict(names)} need a 'person' and a 'sports ball'/'ball' class")
    return person, ball


def frame_rows(res, frame_idx, fps, frame, person_conf, person_class=PERSON_CLASS):
    rows = []
    if res.boxes is None or len(res.boxes) == 0:
        return rows
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, k in zip(boxes, confs, classes):
        if k == person_class:
            if c < person_conf:
                continue
            kit = kit_features(frame, (x1, y1, x2, y2))
            cls = "player"
        else:
            kit = (np.nan, np.nan, np.nan, 0) * 2
            cls = "ball"
        rows.append((frame_idx, frame_idx / fps, cls, float(x1), float(y1), float(x2), float(y2),
                     float((x1 + x2) / 2), float(y2), float(c)) + tuple(kit))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None, help="Video path (default: from meta.json)")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="First frame (default: meta config analysis_window.start_frame, else 0)")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Last frame, inclusive (default: analysis_window.end_frame, else video end)")
    parser.add_argument("--weights", default="yolo26n.pt",
                        help="YOLO weights (auto-downloaded by ultralytics if missing). yolo26n was "
                             "picked on this footage: same person recall as yolov8n, ~2x the ball hits")
    parser.add_argument("--device", default="auto",
                        help="auto (CUDA if available, else intel:gpu if OpenVINO is installed, "
                             "else cpu), intel:gpu / intel:cpu (OpenVINO), cpu, or a CUDA index")
    parser.add_argument("--imgsz", type=int, default=1280, help="Long side of the inference size")
    parser.add_argument("--person-conf", type=float, default=0.10)
    parser.add_argument("--ball-conf", type=float, default=0.05)
    parser.add_argument("--chunk-frames", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=None, help="Only process this many frames (testing)")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    meta = load_meta(args.match_id, args.tmp_dir)
    video = args.video or meta["video"]
    fps = meta["fps"]
    window = meta.get("config", {}).get("analysis_window", {})
    start = args.start_frame if args.start_frame is not None else int(window.get("start_frame", 0))
    end = args.end_frame if args.end_frame is not None else int(window.get("end_frame", meta["frame_count"] - 1))
    if args.limit:
        end = min(end, start + args.limit - 1)
    input_hw = model_input_size(meta["width"], meta["height"], args.imgsz)

    out_dir = match_dir(args.match_id, args.tmp_dir, create=True)
    chunk_dir = out_dir / ("detections_test" if args.limit else "detections")
    chunk_dir.mkdir(exist_ok=True)
    chunks = [(c0, min(c0 + args.chunk_frames - 1, end)) for c0 in range(start, end + 1, args.chunk_frames)]
    todo = [c for c in chunks if not (chunk_dir / f"chunk_{c[0]:07d}.parquet").exists()]
    print(f"[{args.match_id}] frames {start}-{end} ({end - start + 1} frames, "
          f"{(end - start + 1) / fps / 60:.1f} min), {len(chunks)} chunks, {len(todo)} to do")
    print(f"  model {args.weights} on {args.device}, input {input_hw[1]}x{input_hw[0]}")

    if todo:
        model = load_model(args.weights, args.device, input_hw)
        person_cls, ball_cls = class_ids(model.names)
        conf = min(args.person_conf, args.ball_conf)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"OpenCV could not open {video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, todo[0][0])
        pos = todo[0][0]
        t_start, done = time.time(), 0
        total = sum(c1 - c0 + 1 for c0, c1 in todo)
        prog = Progress(total, "detecting")
        for c0, c1 in todo:
            if pos != c0:  # skip over an already-finished chunk
                cap.set(cv2.CAP_PROP_POS_FRAMES, c0)
                pos = c0
            rows = []
            for fi in range(c0, c1 + 1):
                ok, frame = cap.read()
                pos += 1
                if not ok:
                    print(f"WARNING: decode failed at frame {fi}; stopping this chunk early")
                    break
                res = model.predict(frame, imgsz=input_hw, classes=[person_cls, ball_cls],
                                    conf=conf, verbose=False, device=args.device)[0]
                rows.extend(frame_rows(res, fi, fps, frame, args.person_conf, person_cls))
                done += 1
                prog.update(done)
                if done % 500 == 0:
                    el = time.time() - t_start
                    rate = done / el
                    print(f"  {done}/{total} frames  {rate:.1f} fps  ETA {(total - done) / rate / 60:.0f} min",
                          flush=True)
            df = pd.DataFrame(rows, columns=COLUMNS)
            # Write-then-rename: a crash mid-write (e.g. disk full) must never
            # leave a truncated chunk that a resumed run would trust.
            final = chunk_dir / f"chunk_{c0:07d}.parquet"
            tmp = final.with_suffix(".parquet.partial")
            df.to_parquet(tmp, index=False)
            tmp.replace(final)
        cap.release()
        prog.done()
        elapsed = time.time() - t_start
    else:
        elapsed = 0.0

    parts = [pd.read_parquet(chunk_dir / f"chunk_{c0:07d}.parquet") for c0, _ in chunks]
    det = pd.concat(parts, ignore_index=True).sort_values(["frame", "class"]).reset_index(drop=True)
    out_path = out_dir / ("detections_test.parquet" if args.limit else "detections.parquet")
    det.to_parquet(out_path, index=False)
    n_frames = end - start + 1
    players = det[det["class"] == "player"]
    balls = det[det["class"] == "ball"]
    stats = {
        "weights": args.weights, "device": args.device, "input_hw": list(input_hw),
        "person_conf": args.person_conf, "ball_conf": args.ball_conf,
        "start_frame": start, "end_frame": end, "frames": n_frames,
        "rows": int(len(det)),
        "players_per_frame": round(len(players) / n_frames, 2),
        "ball_frame_rate": round(balls["frame"].nunique() / n_frames, 4),
        "ball_frame_rate_conf25": round(balls.loc[balls["conf"] >= 0.25, "frame"].nunique() / n_frames, 4),
        "last_run_seconds": round(elapsed, 1),
    }
    write_json(out_dir / ("detections_test_meta.json" if args.limit else "detections_meta.json"), stats)
    print(f"Wrote {out_path}: {len(det)} rows, {stats['players_per_frame']} players/frame, "
          f"ball in {stats['ball_frame_rate']:.1%} of frames ({stats['ball_frame_rate_conf25']:.1%} at conf>=0.25)")
    print("Next: python tools/calibrate_auto.py --match-id " + args.match_id)


if __name__ == "__main__":
    main()
