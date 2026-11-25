"""Automatic per-frame pitch calibration for a fixed-position pan/tilt/zoom
camera -> .tmp/<match_id>/frame_homographies.npz, and the calibration file
flips to "mode": "dynamic" (the format every compute_* tool already reads).

Why not tools/estimate_camera_motion.py: that chains optical flow between
hand-clicked keyframes, so error accumulates between anchors - fine for a
5-minute Veo clip, tens of metres of drift over a 47-minute broadcast half
that pans and zooms non-stop. Here every frame is re-anchored to the pitch
itself, so drift can't build up:

  1. Seed: fit a physical camera to the landmark clicks in
     config/<id>_calibration.json ("keyframes", calibrate_pitch.py format):
     one fixed optical centre C + roll, and per-keyframe pan / tilt / focal
     length (see tools/_pitch.py). 4 numbers are global, 3 per frame.
  2. Self-calibrate: global pan/tilt/zoom search (coarse-to-fine chamfer
     score) on --self-cal-frames frames spread across the window, then one
     joint least-squares over C, roll and every sampled view so the camera
     position is pinned down from many pan angles, not one.
  3. Track every frame: optical flow (background features, players and
     overlays masked) predicts the next view; the projected pitch-line model
     is then snapped onto the detected white line pixels (truncated chamfer
     distance, 3 parameters, weak prior to the flow prediction so views
     with few visible lines stay stable). If the lines stop agreeing for a
     few frames (cut, heavy occlusion, lost lock) the global search
     re-acquires the view.

Quality per frame (stored in the npz and summarised at the end):
  n_pts  - projected model points that fall on valid image area
  frac2  - share of them within 2 px of a detected line pixel
  status - 0 snapped to lines, 1 flow-only (too few lines visible),
           2 re-acquired by global search, 3 lost (homography is a guess)
Downstream tools should not trust positions on status-3 frames.

Player boxes from detections.parquet are masked out of both line detection
(white kits read as paint) and flow features. Run tools/detect_frames.py
first; without it the tool still runs, slightly noisier.

Usage:
    python tools/calibrate_auto.py --match-id <id>
    python tools/calibrate_auto.py --match-id <id> --max-frames 3000 --check-frames 1200,2500
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from _common import calibration_path, load_meta, match_dir, read_json, write_json
from _pitch import (distance_map, draw_pitch, line_pixels, pitch_polylines, project,
                    projected_pitch_region, rotation, sample_map, sample_polylines,
                    world_to_image_h, decompose_homography, angles_from_rotation)
from estimate_camera_motion import detect_overlay_mask, pair_homography

TAU = 6.0           # chamfer truncation, px
EDGE = 6            # ignore model points this close to the image border
GOOD_FRAC2 = 0.45   # a view whose lines agree this well is trusted
BAD_FRAC2 = 0.25    # below this (with enough points) the lock is suspect


class Camera:
    def __init__(self, C, roll, pp):
        self.C = np.asarray(C, dtype=np.float64)
        self.roll = float(roll)
        self.pp = tuple(pp)

    def H(self, x):
        """x = (pan, tilt, log f) -> pitch->image homography."""
        return world_to_image_h(self.C, self.roll, x[0], x[1], math.exp(x[2]), self.pp)


# --------------------------------------------------------------------------
# Per-frame image evidence
# --------------------------------------------------------------------------
def frame_evidence(frame, overlay_ok, boxes, region=None):
    """(distance map to line pixels, valid-pixel mask). Valid = not a static
    overlay and not inside a (padded) player box."""
    valid = overlay_ok.copy()
    if boxes is not None and len(boxes):
        h, w = valid.shape
        for x1, y1, x2, y2 in boxes:
            valid[max(int(y1) - 3, 0):min(int(y2) + 3, h), max(int(x1) - 3, 0):min(int(x2) + 3, w)] = 0
    if region is not None:
        region = region & overlay_ok
    lines = line_pixels(frame, region=region, exclude_boxes=boxes)
    return distance_map(lines), valid


def visible(cam, x, model, valid):
    """Model points that project inside the image onto valid pixels."""
    h, w = valid.shape
    p, ok = project(cam.H(x), model)
    ok &= np.isfinite(p[:, 0]) & (p[:, 0] >= EDGE) & (p[:, 0] < w - EDGE) & (p[:, 1] >= EDGE) & (p[:, 1] < h - EDGE)
    idx = np.where(ok)[0]
    keep = valid[p[idx, 1].astype(int), p[idx, 0].astype(int)] > 0
    return model[idx[keep]]


def quality(dt, cam, x, model, valid):
    pts = visible(cam, x, model, valid)
    if len(pts) == 0:
        return {"n": 0, "frac2": 0.0, "med": TAU}
    p, _ = project(cam.H(x), pts)
    d, _ = sample_map(dt, p, TAU)
    d = np.minimum(d, TAU)
    return {"n": int(len(pts)), "frac2": float((d < 2).mean()), "med": float(np.median(d))}


def refine(dt, valid, cam, x0, model, prior=None, sigma=(0.004, 0.004, 0.006), rounds=2, max_nfev=30):
    """Snap the projected line model onto line pixels: min sum min(d, TAU)^2
    over visible model points (+ prior to the flow prediction)."""
    x = np.array(x0, dtype=np.float64)
    sig = np.asarray(sigma)
    for _ in range(rounds):
        pts = visible(cam, x, model, valid)
        if len(pts) < 12:
            break

        def resid(v, pts=pts):
            p, _ = project(cam.H(v), pts)
            d, _ = sample_map(dt, p, TAU)
            r = np.minimum(d, TAU)
            if prior is not None:
                r = np.concatenate([r, (v - prior) / sig])
            return r

        x = least_squares(resid, x, diff_step=1e-4, max_nfev=max_nfev, ftol=1e-4, xtol=1e-7).x
    return x, quality(dt, cam, x, model, valid)


def grid_scores(dt, valid, cam, pts, pan, tilt, fs, sigma):
    """Chamfer score of one (pan, tilt) for many focal lengths at once."""
    R = rotation(pan, tilt, cam.roll)
    pc = (np.column_stack([pts, np.zeros(len(pts))]) - cam.C) @ R.T
    ok = pc[:, 2] > 1e-6
    n = pc[ok, :2] / pc[ok, 2:3]
    xy = np.asarray(cam.pp)[None, None, :] + fs[:, None, None] * n[None]
    flat = xy.reshape(-1, 2)
    d, inside = sample_map(dt, flat, 1e9)
    vi = np.zeros(len(flat), bool)
    vi[inside] = valid[flat[inside, 1].astype(int), flat[inside, 0].astype(int)] > 0
    d = d.reshape(len(fs), -1)
    vi = vi.reshape(len(fs), -1)
    s = np.exp(-d ** 2 / (2 * sigma ** 2)) - 0.1 * (d > 3 * sigma)
    return np.where(vi, s, 0.0).sum(axis=1)


def search_view(dt, valid, cam, model_coarse, model, pan_deg=(-50, 50), tilt_deg=(12, 36), f_px=(800, 3200)):
    """Global coarse-to-fine pan/tilt/zoom search, then refine()."""
    fs = np.exp(np.linspace(np.log(f_px[0]), np.log(f_px[1]), 20))
    best = (-1e18, None)
    for pan in np.radians(np.arange(pan_deg[0], pan_deg[1] + 1e-6, 1.0)):
        for tilt in np.radians(np.arange(tilt_deg[0], tilt_deg[1] + 1e-6, 0.75)):
            s = grid_scores(dt, valid, cam, model_coarse, pan, tilt, fs, 14.0)
            k = int(np.argmax(s))
            if s[k] > best[0]:
                best = (s[k], (pan, tilt, fs[k]))
    p0, t0, f0 = best[1]
    fs2 = f0 * np.exp(np.linspace(-0.12, 0.12, 13))
    for pan in p0 + np.radians(np.arange(-1.5, 1.51, 0.25)):
        for tilt in t0 + np.radians(np.arange(-1.0, 1.01, 0.2)):
            s = grid_scores(dt, valid, cam, model, pan, tilt, fs2, 4.0)
            k = int(np.argmax(s))
            if s[k] > best[0]:
                best = (s[k], (pan, tilt, fs2[k]))
    pan, tilt, f = best[1]
    return refine(dt, valid, cam, (pan, tilt, math.log(f)), model, rounds=3)


# --------------------------------------------------------------------------
# Seed fit and self-calibration
# --------------------------------------------------------------------------
def fit_seed(keyframes, pp):
    """Camera globals + per-keyframe views from landmark clicks."""
    inits, data = [], []
    for kf in keyframes:
        img = np.array([p["image"] for p in kf["points"]], dtype=np.float64)
        pit = np.array([p["pitch"] for p in kf["points"]], dtype=np.float64)
        Hw, _ = cv2.findHomography(pit, img, 0)
        dec = decompose_homography(Hw, pp)
        if dec is None:
            raise SystemExit(f"keyframe {kf['frame']}: landmarks don't determine a camera - add more spread-out points")
        f, R, C = dec
        pan, tilt, roll = angles_from_rotation(R)
        inits.append((C, roll, (pan, tilt, math.log(f))))
        data.append((img, pit))
    C0 = np.mean([c for c, _, _ in inits], axis=0)
    roll0 = float(np.mean([r for _, r, _ in inits]))
    x0 = np.concatenate([C0, [roll0]] + [np.array(v) for _, _, v in inits])

    def resid(x):
        cam = Camera(x[:3], x[3], pp)
        out = []
        for k, (img, pit) in enumerate(data):
            p, _ = project(cam.H(x[4 + 3 * k: 7 + 3 * k]), pit)
            out.append((p - img).ravel())
        return np.concatenate(out)

    r = least_squares(resid, x0, loss="soft_l1", f_scale=2.0)
    err = np.linalg.norm(resid(r.x).reshape(-1, 2), axis=1)
    cam = Camera(r.x[:3], r.x[3], pp)
    views = [r.x[4 + 3 * k: 7 + 3 * k] for k in range(len(keyframes))]
    return cam, views, data, err


def self_calibrate(cam, samples, seeds, pts_per_frame=150):
    """Joint least squares over C, roll and every sampled view's (pan, tilt,
    log f): truncated chamfer residuals + landmark residuals for seeds."""
    blocks = []  # (kind, dt-or-landmarks, pts)
    x_views = []
    rng = np.random.default_rng(0)
    for dt, valid, x, model in samples:
        pts = visible(cam, x, model, valid)
        if len(pts) > pts_per_frame:
            pts = pts[rng.choice(len(pts), pts_per_frame, replace=False)]
        blocks.append(("chamfer", dt, pts))
        x_views.append(x)
    for (img, pit), x in seeds:
        blocks.append(("landmarks", img, pit))
        x_views.append(x)
    x0 = np.concatenate([cam.C, [cam.roll]] + [np.asarray(v) for v in x_views])
    sizes = [len(b[2]) * (2 if b[0] == "landmarks" else 1) for b in blocks]

    def resid(x):
        c = Camera(x[:3], x[3], cam.pp)
        out = []
        for k, (kind, a, pts) in enumerate(blocks):
            v = x[4 + 3 * k: 7 + 3 * k]
            p, _ = project(c.H(v), pts)
            if kind == "chamfer":
                d, _ = sample_map(a, p, TAU)
                out.append(np.minimum(d, TAU))
            else:
                out.append(2.0 * (p - a).ravel())  # landmarks weigh double
        return np.concatenate(out)

    sp = lil_matrix((sum(sizes), len(x0)), dtype=int)
    row = 0
    for k, n in enumerate(sizes):
        sp[row:row + n, :4] = 1
        sp[row:row + n, 4 + 3 * k: 7 + 3 * k] = 1
        row += n
    r = least_squares(resid, x0, jac_sparsity=sp, diff_step=1e-4, max_nfev=60)
    return Camera(r.x[:3], r.x[3], cam.pp), [r.x[4 + 3 * k: 7 + 3 * k] for k in range(len(blocks))]


# --------------------------------------------------------------------------
# Flow prediction
# --------------------------------------------------------------------------
def fit_view_to_h(cam, Hw_target, x_init, probe_pitch):
    """PTZ view whose projection of probe points matches a target homography."""
    tgt, ok = project(Hw_target, probe_pitch)
    good = ok & np.all(np.abs(tgt) < 5000, axis=1)
    if good.sum() < 6:
        return np.asarray(x_init, dtype=np.float64)
    pts, tgt = probe_pitch[good], tgt[good]

    def resid(v):
        p, _ = project(cam.H(v), pts)
        return (p - tgt).ravel()

    return least_squares(resid, x_init, diff_step=1e-5, max_nfev=20).x


def probe_points(cam, x, shape, L, W):
    """Pitch points under a 6x4 grid over the image (only those on the pitch)."""
    h, w = shape
    gx, gy = np.meshgrid(np.linspace(0.08 * w, 0.92 * w, 6), np.linspace(0.15 * h, 0.95 * h, 4))
    img = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    Hinv = np.linalg.inv(cam.H(x))
    q = img @ Hinv.T
    q = q[:, :2] / q[:, 2:3]
    on = (q[:, 0] > -15) & (q[:, 0] < L + 15) & (q[:, 1] > -15) & (q[:, 1] < W + 15)
    return q[on]


# --------------------------------------------------------------------------
class BoxSource:
    """Player boxes (for masking) from detections.parquet, or - while
    tools/detect_frames.py is still running - from its chunk files, loaded
    lazily as tracking advances. If tracking catches up with detection it
    waits for the next chunk (up to --wait-min), so the two stages can run
    concurrently: detection on the iGPU, calibration on the CPU."""

    COLS = ["frame", "class", "x1", "y1", "x2", "y2", "conf"]

    def __init__(self, match_id, tmp_dir, start, chunk_frames=2500, min_conf=0.25, wait_min=30.0):
        self.mdir = match_dir(match_id, tmp_dir)
        self.start, self.chunk_frames, self.min_conf, self.wait_s = start, chunk_frames, min_conf, wait_min * 60
        self.boxes, self.loaded = {}, set()
        self.complete = (self.mdir / "detections.parquet").exists()
        if self.complete:
            self._ingest(pd.read_parquet(self.mdir / "detections.parquet", columns=self.COLS))
        self.available = self.complete or (self.mdir / "detections").exists()

    def _ingest(self, det):
        det = det[(det["class"] == "player") & (det["conf"] >= self.min_conf)]
        for f, g in det.groupby("frame"):
            self.boxes[int(f)] = g[["x1", "y1", "x2", "y2"]].to_numpy()

    def get(self, frame):
        if self.complete or not self.available:
            return self.boxes.get(frame)
        c0 = self.start + ((frame - self.start) // self.chunk_frames) * self.chunk_frames
        if c0 not in self.loaded:
            path = self.mdir / "detections" / f"chunk_{c0:07d}.parquet"
            waited = 0.0
            while not path.exists() and waited < self.wait_s:
                if waited == 0:
                    print(f"  waiting for detection chunk {path.name}...", flush=True)
                time.sleep(15)
                waited += 15
            if path.exists():
                self._ingest(pd.read_parquet(path, columns=self.COLS))
            else:
                print(f"WARNING: {path.name} never appeared - frames {c0}+ calibrated without player masking")
            self.loaded.add(c0)
        return self.boxes.get(frame)

    def peek(self, frame):
        """Non-blocking get: boxes if that chunk is already on disk, else None."""
        if self.complete or not self.available:
            return self.boxes.get(frame)
        c0 = self.start + ((frame - self.start) // self.chunk_frames) * self.chunk_frames
        path = self.mdir / "detections" / f"chunk_{c0:07d}.parquet"
        if c0 not in self.loaded and path.exists():
            self._ingest(pd.read_parquet(path, columns=self.COLS))
            self.loaded.add(c0)
        return self.boxes.get(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--calibration", default=None, help="Default: config/<match_id>_calibration.json")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Track only this many frames (testing)")
    parser.add_argument("--self-cal-frames", type=int, default=16,
                        help="Views sampled across the window to pin down the camera position (0 = skip)")
    parser.add_argument("--check-frames", default="",
                        help="Comma-separated frames to save overlay JPEGs for (.tmp/<id>/calib_check/)")
    parser.add_argument("--wait-min", type=float, default=30.0,
                        help="How long to wait for a detection chunk that detect_frames.py hasn't written yet")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    video = args.video or meta["video"]
    shape = (meta["height"], meta["width"])
    pp = (meta["width"] / 2.0, meta["height"] / 2.0)
    window = meta.get("config", {}).get("analysis_window", {})
    start = args.start_frame if args.start_frame is not None else int(window.get("start_frame", 0))
    end = args.end_frame if args.end_frame is not None else int(window.get("end_frame", meta["frame_count"] - 1))
    if args.max_frames:
        end = min(end, start + args.max_frames - 1)
    calib_file = Path(args.calibration) if args.calibration else calibration_path(args.match_id, args.config_dir)
    calib = read_json(calib_file)
    L, W = float(calib.get("pitch_length", 105.0)), float(calib.get("pitch_width", 68.0))
    model = sample_polylines(pitch_polylines(L, W), step=0.75)
    model_coarse = sample_polylines(pitch_polylines(L, W), step=1.5)
    check = {int(f) for f in args.check_frames.split(",") if f.strip()}
    out_dir = match_dir(args.match_id, args.tmp_dir, create=True)
    if check:
        (out_dir / "calib_check").mkdir(exist_ok=True)

    # 1. seed
    cam, seed_views, seed_data, err = fit_seed(calib["keyframes"], pp)
    print(f"[{args.match_id}] seed fit on {len(seed_views)} keyframe(s): landmark error mean {err.mean():.2f} px, "
          f"max {err.max():.2f} px")
    print(f"  camera C = ({cam.C[0]:.1f}, {cam.C[1]:.1f}, {cam.C[2]:.1f}) m, roll {math.degrees(cam.roll):.2f} deg")

    boxes = BoxSource(args.match_id, args.tmp_dir, start, wait_min=args.wait_min)
    print("  player boxes for masking: " + ("detections.parquet" if boxes.complete else
                                             "detection chunks, loaded as tracking advances" if boxes.available
                                             else "none (run tools/detect_frames.py first)"))
    print("  detecting static overlays...")
    overlay_ok = detect_overlay_mask(video, meta["frame_count"]).astype(np.uint8)

    cap = cv2.VideoCapture(str(video))

    def read(fi):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        return fr if ok else None

    # 2. self-calibration across the window
    if args.self_cal_frames > 0:
        t0 = time.time()
        samples = []
        seed_frames = {kf["frame"] for kf in calib["keyframes"]}
        for fi in np.linspace(start, end, args.self_cal_frames).astype(int):
            fr = read(int(fi))
            if fr is None:
                continue
            dt, valid = frame_evidence(fr, overlay_ok, boxes.peek(int(fi)))
            x, q = search_view(dt, valid, cam, model_coarse, model)
            if q["frac2"] >= GOOD_FRAC2 and q["n"] >= 80:
                samples.append((dt, valid, x, model))
        before = cam.C.copy()
        cam, _ = self_calibrate(cam, samples, list(zip(seed_data, seed_views)))
        print(f"  self-calibration on {len(samples)} searched views + {len(seed_frames)} seed(s) "
              f"({time.time() - t0:.0f}s): C ({before[0]:.1f}, {before[1]:.1f}, {before[2]:.1f}) -> "
              f"({cam.C[0]:.1f}, {cam.C[1]:.1f}, {cam.C[2]:.1f}), roll {math.degrees(cam.roll):.2f} deg")

    # 3. track every frame
    n = end - start + 1
    params = np.zeros((n, 3))
    q_n = np.zeros(n, np.int32)
    q_frac = np.zeros(n, np.float32)
    q_med = np.zeros(n, np.float32)
    status = np.zeros(n, np.int8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev_small, prev_valid_small, x_prev, x_prev2 = None, None, None, None
    bad_run, t0, searches = 0, time.time(), 0
    S = np.diag([0.5, 0.5, 1.0])
    S_inv = np.linalg.inv(S)
    for k in range(n):
        fi = start + k
        ok, fr = cap.read()
        if not ok:
            print(f"WARNING: decode failed at frame {fi}; remaining frames reuse the last view")
            params[k:] = x_prev
            status[k:] = 3
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (shape[1] // 2, shape[0] // 2), interpolation=cv2.INTER_AREA)
        bx = boxes.get(fi)
        st = 0
        if x_prev is None:
            dt, valid = frame_evidence(fr, overlay_ok, bx)
            x, q = search_view(dt, valid, cam, model_coarse, model)
            searches += 1
            st = 2
        else:
            Hfl, _ = pair_homography(prev_small, small, prev_valid_small)  # cur -> prev (half res)
            if Hfl is not None:
                Hc2p = S_inv @ Hfl @ S
                H_pred = np.linalg.inv(Hc2p) @ cam.H(x_prev)
                x_pred = fit_view_to_h(cam, H_pred, x_prev, probe_points(cam, x_prev, shape, L, W))
            else:
                x_pred = x_prev + 0.5 * (x_prev - x_prev2) if x_prev2 is not None else x_prev
            region = projected_pitch_region(cam.H(x_pred), shape, L, W, margin=1.5)
            dt, valid = frame_evidence(fr, overlay_ok, bx, region=region)
            x, q = refine(dt, valid, cam, x_pred, model, prior=x_pred)
            if q["n"] < 25:
                st = 1  # too few lines in view: the flow prediction carries it
            elif q["frac2"] < BAD_FRAC2:
                bad_run += 1
                st = 1
            else:
                bad_run = 0
            if Hfl is None or bad_run >= 3:
                dt_full, valid_full = frame_evidence(fr, overlay_ok, bx)
                xs, qs = search_view(dt_full, valid_full, cam, model_coarse, model)
                searches += 1
                if qs["frac2"] >= GOOD_FRAC2 and qs["n"] >= 60:
                    x, q, st, bad_run = xs, qs, 2, 0
                elif bad_run >= 25:
                    st = 3
        params[k], q_n[k], q_frac[k], q_med[k], status[k] = x, q["n"], q["frac2"], q["med"], st
        vs = cv2.resize(valid, (shape[1] // 2, shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        prev_small, prev_valid_small, x_prev2, x_prev = small, vs, x_prev, x
        if fi in check:
            vis = fr.copy()
            draw_pitch(vis, cam.H(x), L, W, color=(0, 0, 255))
            cv2.putText(vis, f"f{fi} status {st} n {q['n']} frac2 {q['frac2']:.2f}", (10, 700),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(out_dir / "calib_check" / f"f{fi}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if (k + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{n} frames  {(k + 1) / el:.1f} fps  ETA {(n - k - 1) / ((k + 1) / el) / 60:.0f} min  "
                  f"median frac2 (last 1000) {np.median(q_frac[k - 999:k + 1]):.2f}  searches {searches}", flush=True)
    cap.release()

    # 4. save: image->pitch homographies (the format _common.apply_calibration reads)
    Hs = np.empty((n, 3, 3))
    for k in range(n):
        Hi = np.linalg.inv(cam.H(params[k]))
        Hs[k] = Hi / Hi[2, 2]
    npz_path = out_dir / "frame_homographies.npz"
    np.savez_compressed(npz_path, frames=np.arange(start, end + 1), H=Hs, params=params,
                        n_pts=q_n, frac2=q_frac, med_px=q_med, status=status)
    counts = {s: int((status == s).sum()) for s in range(4)}
    summary = {
        "frames": n, "start_frame": start, "end_frame": end,
        "snapped": counts[0], "flow_only": counts[1], "reacquired": counts[2], "lost": counts[3],
        "median_frac2": round(float(np.median(q_frac)), 3),
        "frac2_below_0.25": round(float((q_frac < 0.25).mean()), 4),
        "global_searches": searches,
    }
    calib["mode"] = "dynamic"
    calib["frame_homographies"] = Path(npz_path).as_posix()  # same on Windows and Linux
    calib["camera_model"] = {
        "method": "fixed-position PTZ, per-frame chamfer snap to pitch lines (tools/calibrate_auto.py)",
        "C": [round(float(v), 3) for v in cam.C], "roll_deg": round(math.degrees(cam.roll), 4),
        "principal_point": list(pp),
    }
    calib["motion_summary"] = summary
    write_json(calib_file, calib)
    print(f"Wrote {npz_path} ({n} frames) and updated {calib_file} (mode: dynamic)")
    print(f"  snapped {counts[0]}, flow-only {counts[1]}, re-acquired {counts[2]}, LOST {counts[3]}; "
          f"median frac2 {summary['median_frac2']}")
    if counts[3]:
        print("WARNING: some frames lost lock - positions there are unreliable (status 3 in the npz)")


if __name__ == "__main__":
    main()
