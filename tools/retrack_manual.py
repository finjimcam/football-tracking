"""Turn the roster's manual boxes into real tracks and merge manual ball
marks -> updates .tmp/<match_id>/tracks.parquet (new rows, source="manual").

Run AFTER label_roster.py whenever it reported manual boxes or ball marks,
and BEFORE assign_identities.py (which resolves manual labels against the
tracks this tool creates).

How a manual box becomes a track — per-frame YOLO re-detection on a zoomed
ROI, forward AND backward from the anchor frame:
  crop a region around the predicted position (last box + velocity), run the
  player model on just that crop at small imgsz, and keep the detection that
  best overlaps the prediction. Zooming is the point: pitch-scale players are
  ~40 px tall in the full frame but fill a 4x-box crop, so the detector that
  missed them at full-frame scale finds them here — and unlike a generic
  correlation tracker it can't silently drift onto grass, because every frame
  must re-detect an actual person. (The workflow's original CSRT plan was
  dropped: this OpenCV build (5.0, no contrib) doesn't ship CSRT, and ROI
  re-detection was already the agreed escalation for CSRT drift.)

A run stops when:
  - handoff: the box overlaps one automatic track (IoU >= --handoff-iou) for
    --handoff-frames consecutive frames — the person is auto-tracked from
    there, so rows are trimmed back to the handoff point and
    assign_identities.py carries the identity across the join;
  - lost: no acceptable re-detection for --max-miss frames (coasting on the
    velocity prediction in between);
  - the video ends.

Ball marks are merged as class="ball" rows with conf 1.0; automatic ball
detections on a marked frame are dropped (manual beats auto).

The first run backs up tracks.parquet to tracks_auto.parquet and re-runs
always rebuild from that backup (idempotent — tweak flags freely). If
stitch_tracks.py is re-run later it rewrites tracks.parquet without manual
rows; just run this tool again (it notices the fresh base and re-backs-up).

Usage:
    python tools/retrack_manual.py --match-id <match_id>
    python tools/retrack_manual.py --match-id <match_id> --max-miss 20
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from _common import load_meta, load_roster, match_dir


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


class VideoFrames:
    """Sequential reads when frames come in order (forward pass), seeks when
    they don't (backward pass — per-frame seek, fine for the short runs a
    manual track usually is)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"OpenCV could not open {path}")
        self.pos = 0

    def get(self, idx):
        if idx != self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        ok, img = self.cap.read()
        self.pos += 1
        return img if ok else None

    def release(self):
        self.cap.release()


def roi_detect(model, frame, pred_box, roi_scale, imgsz, conf):
    """Person detections inside a crop around the predicted box, mapped back
    to full-frame coords -> [((x1,y1,x2,y2), conf), ...]."""
    H, W = frame.shape[:2]
    cx, cy = (pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2
    rw = max((pred_box[2] - pred_box[0]) * roi_scale, 96)
    rh = max((pred_box[3] - pred_box[1]) * roi_scale, 96)
    x1, y1 = int(max(0, cx - rw / 2)), int(max(0, cy - rh / 2))
    x2, y2 = int(min(W, cx + rw / 2)), int(min(H, cy + rh / 2))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return []
    res = model.predict(frame[y1:y2, x1:x2], classes=[0], imgsz=imgsz,
                        conf=conf, verbose=False)[0]
    out = []
    if res.boxes is not None:
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            out.append(((float(b[0]) + x1, float(b[1]) + y1,
                         float(b[2]) + x1, float(b[3]) + y1), float(c)))
    return out


def shift_box(box, vel, steps=1):
    dx, dy = vel[0] * steps, vel[1] * steps
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def track_direction(model, frames, start_frame, start_box, step, last_frame,
                    auto_by_frame, args):
    """ROI-redetect from start_frame in one direction (step = +1/-1).
    Returns (rows [(frame, box, conf)], stop_reason string)."""
    box, vel = tuple(start_box), (0.0, 0.0)
    miss, streak, streak_tid, streak_start = 0, 0, None, None
    rows, reason = [], "video end"
    f = start_frame + step
    while 0 <= f <= last_frame:
        img = frames.get(f)
        if img is None:
            reason = "video read failed"
            break
        pred = shift_box(box, vel, 1 + miss)
        dets = roi_detect(model, img, pred, args.roi_scale, args.roi_imgsz, args.det_conf)
        area = (pred[2] - pred[0]) * (pred[3] - pred[1])
        best, best_iou = None, args.min_iou
        for b, c in dets:
            a = (b[2] - b[0]) * (b[3] - b[1])
            if not (0.4 <= a / max(area, 1e-6) <= 2.5):  # size jump = wrong person
                continue
            i = box_iou(b, pred)
            if i > best_iou:
                best, best_iou = (b, c), i
        if best is None:
            miss += 1
            if miss > args.max_miss:
                reason = f"lost (no re-detection for {args.max_miss} frames)"
                break
            f += step
            continue
        newbox, conf = best
        cx0, cy0 = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        cx1, cy1 = (newbox[0] + newbox[2]) / 2, (newbox[1] + newbox[3]) / 2
        vel = ((cx1 - cx0) / (1 + miss), (cy1 - cy0) / (1 + miss))
        box, miss = newbox, 0
        rows.append((f, box, conf))

        # Handoff: sustained overlap with one automatic track -> its problem now.
        hit_tid, hit_iou = None, args.handoff_iou
        for tid, abox in auto_by_frame.get(f, []):
            i = box_iou(box, abox)
            if i >= hit_iou:
                hit_tid, hit_iou = tid, i
        if hit_tid is not None:
            if hit_tid == streak_tid:
                streak += 1
            else:
                streak_tid, streak, streak_start = hit_tid, 1, f
            if streak >= args.handoff_frames:
                keep = (lambda fr: fr < streak_start) if step > 0 else (lambda fr: fr > streak_start)
                rows = [r for r in rows if keep(r[0])]
                reason = f"handoff to track #{streak_tid}"
                break
        else:
            streak, streak_tid = 0, None
        f += step
    return rows, reason


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None, help="Video path (default: from meta.json)")
    parser.add_argument("--roster", default=None, help="Default: config/<match_id>_roster.json")
    parser.add_argument("--weights", default="yolov8n.pt", help="Player detection weights")
    parser.add_argument("--roi-scale", type=float, default=4.0,
                        help="ROI crop size as a multiple of the box size")
    parser.add_argument("--roi-imgsz", type=int, default=320, help="Inference size on the crop")
    parser.add_argument("--det-conf", type=float, default=0.15)
    parser.add_argument("--min-iou", type=float, default=0.10,
                        help="Min IoU between a re-detection and the prediction to accept it")
    parser.add_argument("--max-miss", type=int, default=12,
                        help="Stop after this many consecutive frames without a re-detection")
    parser.add_argument("--handoff-iou", type=float, default=0.5)
    parser.add_argument("--handoff-frames", type=int, default=5,
                        help="Consecutive frames overlapping one auto track before handing off")
    parser.add_argument("--device", default=None, help="Torch device (default: auto)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    video = args.video or meta["video"]
    fps, last_frame = meta["fps"], meta["frame_count"] - 1
    roster = load_roster(args.match_id, args.config_dir, args.roster)
    if roster is None:
        raise SystemExit(f"No roster found for {args.match_id} — run tools/label_roster.py first")

    manual_jobs = [(a["frame"], mb) for a in roster["anchors"]
                   for mb in a.get("manual_boxes", [])]
    ball_jobs = [(a["frame"], bm) for a in roster["anchors"]
                 for bm in a.get("ball_marks", [])]
    if not manual_jobs and not ball_jobs:
        print("Roster has no manual boxes or ball marks — nothing to do.")
        print("Next: python tools/assign_identities.py --match-id", args.match_id)
        return

    mdir = match_dir(args.match_id, args.tmp_dir)
    tracks_path = mdir / "tracks.parquet"
    backup_path = mdir / "tracks_auto.parquet"
    if not tracks_path.exists():
        raise SystemExit(f"{tracks_path} not found — run tools/run_tracking.py first")
    current = pd.read_parquet(tracks_path)
    has_manual = "source" in current.columns and (current["source"] == "manual").any()
    if has_manual and backup_path.exists():
        tracks = pd.read_parquet(backup_path)  # idempotent re-run
        print(f"Rebuilding from {backup_path} (previous manual rows discarded)")
    else:
        tracks = current  # fresh base (first run, or stitch_tracks was re-run)
        tracks.to_parquet(backup_path, index=False)
        print(f"Backed up pre-manual tracks to {backup_path}")
    if "source" not in tracks.columns:
        tracks["source"] = "auto"

    new_rows = []
    next_tid = int(tracks.loc[tracks["class"] == "player", "track_id"].max()) + 1

    if manual_jobs:
        from ultralytics import YOLO  # deferred: slow import

        model = YOLO(args.weights)
        if args.device:
            model.to(args.device)
        auto = tracks[(tracks["class"] == "player") & (tracks["track_id"] >= 0)]
        auto_by_frame = {
            int(f): [(int(r["track_id"]), (r["x1"], r["y1"], r["x2"], r["y2"]))
                     for _, r in grp.iterrows()]
            for f, grp in auto.groupby("frame")
        }
        frames = VideoFrames(video)
        for anchor_frame, mb in manual_jobs:
            box = tuple(float(v) for v in mb["box"])
            tag = mb.get("key") or "unlabeled"
            fwd, fwd_reason = track_direction(model, frames, anchor_frame, box, +1,
                                              last_frame, auto_by_frame, args)
            bwd, bwd_reason = track_direction(model, frames, anchor_frame, box, -1,
                                              last_frame, auto_by_frame, args)
            span = sorted(bwd, key=lambda r: r[0]) + [(anchor_frame, box, 1.0)] \
                + sorted(fwd, key=lambda r: r[0])
            for f, b, c in span:
                new_rows.append({
                    "frame": int(f), "t_s": f / fps, "track_id": next_tid,
                    "class": "player",
                    "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                    "foot_x": (b[0] + b[2]) / 2, "foot_y": b[3],
                    "conf": float(c), "source": "manual",
                })
            print(f"Manual box {tag} @ frame {anchor_frame} -> track #{next_tid}: "
                  f"{len(bwd)} frames back (stopped: {bwd_reason}), "
                  f"{len(fwd)} forward (stopped: {fwd_reason})")
            if not mb.get("key"):
                print("  NOTE: this box has no identity label — it will stay unidentified "
                      "(re-open label_roster.py on that frame to label it)")
            next_tid += 1
        frames.release()

    if ball_jobs:
        marked_frames = sorted({f for f, _ in ball_jobs})
        drop = (tracks["class"] == "ball") & tracks["frame"].isin(marked_frames)
        if int(drop.sum()):
            print(f"Dropping {int(drop.sum())} automatic ball row(s) on marked frames "
                  "(manual beats auto)")
        tracks = tracks[~drop]
        for f, (bx, by) in ball_jobs:
            r = 8.0
            new_rows.append({
                "frame": int(f), "t_s": f / fps, "track_id": -1, "class": "ball",
                "x1": bx - r, "y1": by - r, "x2": bx + r, "y2": by + r,
                "foot_x": float(bx), "foot_y": by + r,
                "conf": 1.0, "source": "manual",
            })
        print(f"Merged {len(ball_jobs)} manual ball mark(s)")

    out = pd.concat([tracks, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)
    out.to_parquet(tracks_path, index=False)
    n_manual = int((out["source"] == "manual").sum())
    print(f"Wrote {tracks_path} ({len(out)} rows, {n_manual} manual)")
    print("Next: python tools/assign_identities.py --match-id", args.match_id)


if __name__ == "__main__":
    main()
