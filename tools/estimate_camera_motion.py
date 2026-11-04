"""Per-frame camera-motion homographies for a moving camera →
.tmp/<match_id>/frame_homographies.npz (+ updates the calibration file to
"mode": "dynamic").

Requires a keyframe calibration first (tools/calibrate_pitch.py --frames ...).
Between keyframes the camera motion is estimated automatically: sparse
optical flow (Shi-Tomasi corners + pyramidal Lucas-Kanade with a
forward-backward check) on background features, then a RANSAC homography per
consecutive frame pair. Pair homographies are chained from each manual
keyframe and blended between neighboring keyframes so drift is pulled back
to zero at every anchor.

This assumes the camera pans/tilts/zooms from a fixed position (tripod,
mast, Veo-style unit) — then inter-frame motion of the *static scene* is an
exact homography. A camera that physically travels (drone flyover, handheld
walking) introduces parallax this model can't fully capture.

Static overlays (letterbox borders, watermarks, scoreboards) are detected
automatically (pixels that never change across sampled frames) and masked
out of feature selection, as are player/ball boxes when tracks.parquet
exists (run tools/run_tracking.py first for best results).

Health metrics printed at the end — take them seriously:
- pairs that fell back to identity (flow failure) — a few isolated ones are
  fine, clusters mean trouble (cuts, motion blur, occlusion);
- drift at each keyframe: where the chain from the previous keyframe
  disagrees with your manual clicks, in meters. > ~3 m means keyframes are
  too far apart for how fast the camera moves — add one in between.

Usage:
    python tools/estimate_camera_motion.py --match-id <match_id>
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import (Progress, calibration_path, load_meta, match_dir, project_points,
                     read_json, write_json)

MIN_PAIR_INLIERS = 15
FB_ERR_PX = 1.0  # forward-backward flow consistency threshold


def detect_overlay_mask(video, frame_count, samples=7):
    """Boolean mask (H, W): True where pixels are usable (not a static
    overlay). Samples frames across the clip; pixels whose gray value barely
    changes while the camera moves are overlays (borders, watermark)."""
    idxs = np.linspace(0, max(frame_count - 1, 0), samples).astype(int)
    cap = cv2.VideoCapture(str(video))
    grays = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16))
    cap.release()
    if len(grays) < 2:
        raise SystemExit(f"Could not sample frames from {video}")
    stack = np.stack(grays)
    ptp = stack.max(axis=0) - stack.min(axis=0)
    usable = ptp > 6
    usable = cv2.erode(usable.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    frac = usable.mean()
    if frac < 0.2:
        # Camera barely moved across the samples — can't tell overlay from
        # scene. Fall back to masking only near-black pixels (letterbox).
        print(f"WARNING: overlay detection found only {frac:.0%} usable pixels "
              "(camera may be nearly static) — falling back to black-border mask only")
        usable = stack.mean(axis=0) > 12
        usable = cv2.erode(usable.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    return usable


def load_track_boxes(match_id, tmp_dir):
    """{frame: [(x1, y1, x2, y2), ...]} of detected boxes, or None if no
    tracks.parquet yet."""
    path = match_dir(match_id, tmp_dir) / "tracks.parquet"
    if not path.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(path, columns=["frame", "x1", "y1", "x2", "y2"])
    boxes = {}
    for frame, g in df.groupby("frame"):
        boxes[int(frame)] = g[["x1", "y1", "x2", "y2"]].to_numpy()
    return boxes


def frame_mask(base_mask, boxes, pad=8):
    """Feature-selection mask for one frame: overlay mask minus detection
    boxes (padded — flow around a moving player's edges is unreliable)."""
    mask = (base_mask * 255).astype(np.uint8)
    if boxes is not None:
        h, w = mask.shape
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(mask,
                          (max(int(x1) - pad, 0), max(int(y1) - pad, 0)),
                          (min(int(x2) + pad, w - 1), min(int(y2) + pad, h - 1)),
                          0, -1)
    return mask


def pair_homography(prev_gray, cur_gray, mask):
    """Homography mapping cur pixels → prev pixels, or (None, 0) on failure."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01,
                                  minDistance=12, blockSize=7, mask=mask)
    if pts is None or len(pts) < 20:
        return None, 0
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, nxt, None, **lk)
    fb_err = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_err < FB_ERR_PX)
    if good.sum() < 20:
        return None, 0
    p_prev = pts.reshape(-1, 2)[good]
    p_cur = nxt.reshape(-1, 2)[good]
    H, inl = cv2.findHomography(p_cur, p_prev, cv2.RANSAC, 3.0)
    n_inliers = int(inl.sum()) if inl is not None else 0
    if H is None or n_inliers < MIN_PAIR_INLIERS:
        return None, 0
    return H, n_inliers


def normalize_h(H):
    """Scale-normalize a homography so matrices can be blended element-wise."""
    n = np.linalg.norm(H)
    if n == 0:
        return H
    H = H / n
    return H if H[2, 2] >= 0 else -H


def blend_h(Ha, Hb, w):
    """Element-wise blend of two scale-normalized homographies. Valid for
    nearby transforms, which anchored chains to the same frame always are."""
    A, B = normalize_h(Ha), normalize_h(Hb)
    if np.sum(A * B) < 0:
        B = -B
    return (1 - w) * A + w * B


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None, help="Video path (default: from meta.json)")
    parser.add_argument("--calibration", default=None,
                        help="Keyframe calibration file (default: config/<match_id>_calibration.json)")
    parser.add_argument("--no-mask-tracks", action="store_true",
                        help="Don't mask player/ball boxes even if tracks.parquet exists")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    calib_path = Path(args.calibration) if args.calibration else calibration_path(args.match_id, args.config_dir)
    if not calib_path.exists():
        raise SystemExit(f"{calib_path} not found — run tools/calibrate_pitch.py --frames ... first")
    calib = read_json(calib_path)
    if calib.get("mode") not in ("keyframes", "dynamic"):
        raise SystemExit(f"{calib_path} has mode '{calib.get('mode')}' — moving-camera estimation needs "
                         "a keyframe calibration (tools/calibrate_pitch.py --frames N1,N2,...)")
    keyframes = sorted(calib["keyframes"], key=lambda k: k["frame"])
    kf_idx = [k["frame"] for k in keyframes]
    kf_H = {k["frame"]: np.array(k["homography"], dtype=np.float64) for k in keyframes}

    meta = load_meta(args.match_id, args.tmp_dir)
    video = args.video or meta["video"]
    frame_count = meta["frame_count"]

    print(f"[{args.match_id}] detecting static overlays...")
    base_mask = detect_overlay_mask(video, frame_count)
    print(f"  usable image area: {base_mask.mean():.0%}")

    boxes = None if args.no_mask_tracks else load_track_boxes(args.match_id, args.tmp_dir)
    print("  masking detection boxes from tracks.parquet" if boxes is not None else
          "  no tracks.parquet — proceeding without player masking (RANSAC still rejects moving players)")

    # Pass over the video: H_back[f] maps frame f pixels → frame f-1 pixels.
    cap = cv2.VideoCapture(str(video))
    ok, prev = cap.read()
    if not ok:
        raise SystemExit(f"Could not read {video}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    n = 1
    H_back = [np.eye(3)]
    inlier_counts = [0]
    failed = []
    prog = Progress(frame_count, "camera motion")
    while True:
        ok, cur = cap.read()
        if not ok:
            break
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        mask = frame_mask(base_mask, boxes.get(n - 1) if boxes else None)
        H, n_inl = pair_homography(prev_gray, cur_gray, mask)
        if H is None:
            failed.append(n)
            H = np.eye(3)
        H_back.append(H)
        inlier_counts.append(n_inl)
        prev_gray = cur_gray
        n += 1
        prog.update(n)
        if n % 500 == 0:
            print(f"  {n}/{frame_count} frames...")
    cap.release()
    prog.done()
    print(f"  estimated {n - 1} frame pairs, {len(failed)} fell back to identity")

    # Cumulative chain: C[f] maps frame f pixels → frame 0 pixels.
    C = np.empty((n, 3, 3))
    C[0] = np.eye(3)
    for f in range(1, n):
        C[f] = C[f - 1] @ H_back[f]

    # Anchor at keyframes: pitch homography from anchor k is
    # K[k] @ C[k]^-1 @ C[f]; blend between the two surrounding anchors.
    def anchored(f, k):
        return kf_H[k] @ np.linalg.inv(C[k]) @ C[f]

    H_pitch = np.empty((n, 3, 3))
    for f in range(n):
        prev_ks = [k for k in kf_idx if k <= f]
        next_ks = [k for k in kf_idx if k > f]
        if prev_ks and next_ks:
            k1, k2 = prev_ks[-1], next_ks[0]
            w = (f - k1) / (k2 - k1)
            H_pitch[f] = blend_h(anchored(f, k1), anchored(f, k2), w)
        else:
            k = prev_ks[-1] if prev_ks else next_ks[0]
            H_pitch[f] = anchored(f, k)

    # Drift check: chain from each keyframe to the next, compare against the
    # manual homography there (image center + quarter points, in meters).
    h_img, w_img = base_mask.shape
    probe = np.array([[w_img / 2, h_img / 2], [w_img / 4, h_img / 2],
                      [3 * w_img / 4, h_img / 2], [w_img / 2, 3 * h_img / 4]])
    drifts = []
    for k1, k2 in zip(kf_idx, kf_idx[1:]):
        chained = project_points(anchored(k2, k1), probe)
        manual = project_points(kf_H[k2], probe)
        drift = float(np.linalg.norm(chained - manual, axis=1).mean())
        drifts.append((k1, k2, drift))
        flag = "  <-- consider an extra keyframe in this span" if drift > 3.0 else ""
        print(f"  drift keyframe {k1} -> {k2}: {drift:.2f} m{flag}")

    out_dir = match_dir(args.match_id, args.tmp_dir, create=True)
    npz_path = out_dir / "frame_homographies.npz"
    np.savez_compressed(npz_path, frames=np.arange(n), H=H_pitch,
                        inliers=np.array(inlier_counts), failed=np.array(failed, dtype=int))

    calib["mode"] = "dynamic"
    calib["frame_homographies"] = str(npz_path)
    calib["motion_summary"] = {
        "frames": n,
        "failed_pairs": len(failed),
        "keyframe_drift_m": [{"from": a, "to": b, "drift_m": round(d, 3)} for a, b, d in drifts],
    }
    write_json(calib_path, calib)
    print(f"Wrote {npz_path} ({n} frames) and updated {calib_path} (mode: dynamic)")
    if failed:
        runs = np.split(np.array(failed), np.where(np.diff(failed) > 1)[0] + 1)
        worst = max(runs, key=len)
        if len(worst) > 5:
            print(f"WARNING: {len(worst)} consecutive pair failures around frame {worst[0]} — "
                  "positions there will smear; check the video for cuts or heavy blur")


if __name__ == "__main__":
    main()
