"""Chain image->pitch homographies backwards from a calibrated frame, for a
camera shot that calibrate_auto.py can't lock onto (e.g. the zoomed-out wide
view at a kick-off, where it locks onto the stands instead of the pitch).

The camera is fixed in place (pan/tilt/zoom only), so two frames of the same
shot are related by an image homography. Each frame before --anchor-frame is
matched to a reference frame whose pitch homography is known: SIFT features,
with player boxes (from the detections) and static overlays (pixels that
never change over the shot: score bug, logos) masked out, then RANSAC. The
reference starts at the anchor and steps back to the last chained frame when
the match gets weak, so the error grows only at those steps, not every frame.

    python tools/chain_homographies.py --match-id <id> \\
        --npz .tmp/<id>/periods/h2/<id>/frame_homographies.npz \\
        --detections .tmp/<id>/periods/h2/<id>/det_86760.parquet \\
        --start-frame 86760 --anchor-frame 87110

Writes the chained frames into --npz (or --out) with status 1 (flow-only, as
calibrate_auto.py marks frames without enough lines) and n_pts = RANSAC
inliers. params / frac2 / med_px are NaN for them. --check-frames saves
images with the pitch lines drawn through the chained homography.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _common import load_meta, match_dir

SCALE = 0.5  # features are found at half resolution


def read_frames(video, start, end):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(start, end + 1):
        ok, img = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(out) != end - start + 1:
        raise SystemExit(f"read {len(out)} of {end - start + 1} frames from {video}")
    return out


def static_mask(frames, std_max=2.0):
    """255 where the image changes over the shot; 0 on static overlays."""
    stack = np.stack(frames[:: max(1, len(frames) // 60)]).astype(np.float32)
    return np.where(stack.std(axis=0) > std_max, 255, 0).astype(np.uint8)


def frame_mask(base, boxes, pad=4):
    m = base.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(m, (int(x1 * SCALE) - pad, int(y1 * SCALE) - pad),
                      (int(x2 * SCALE) + pad, int(y2 * SCALE) + pad), 0, -1)
    return m


def to_full(A):
    """A homography between half-resolution images -> between full-resolution ones."""
    S = np.diag([SCALE, SCALE, 1.0])
    return np.linalg.inv(S) @ A @ S


def match(sift, bf, kd_a, kd_b, min_inliers):
    """Image homography taking frame a's pixels to frame b's (half resolution), inliers."""
    (ka, da), (kb, db) = kd_a, kd_b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    pairs = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None, len(good)
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    A, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    n = int(inl.sum()) if inl is not None else 0
    return (A if n >= min_inliers else None), n


def chain(frames, boxes, H_anchor, min_inliers=150, refresh=400):
    """frames[0..n-1] are the frames before the anchor (frames[n] is the anchor).
    Returns ({index: image->pitch H}, {index: inliers})."""
    sift = cv2.SIFT_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    base = static_mask(frames)
    feats = {}

    def kd(i):
        if i not in feats:
            feats[i] = sift.detectAndCompute(frames[i], frame_mask(base, boxes.get(i, ())))
        return feats[i]

    n = len(frames) - 1
    H = {n: H_anchor}
    inliers = {}
    ref = n
    for i in range(n - 1, -1, -1):
        A, k = match(sift, bf, kd(i), kd(ref), min_inliers)
        if (A is None or k < refresh) and ref != i + 1:
            ref = i + 1  # the reference is too far away now: step it back to the last chained frame
            A, k = match(sift, bf, kd(i), kd(ref), min_inliers)
        if A is None:
            print(f"  stopped at index {i}: only {k} matches to index {ref}")
            break
        Hi = H[ref] @ to_full(A)
        H[i] = Hi / Hi[2, 2]
        inliers[i] = k
        for j in [j for j in feats if j > ref]:
            del feats[j]
    del H[n]
    return H, inliers


def draw_check(img, H, out):
    """Pitch lines (105 x 68) projected through image->pitch H."""
    Hinv = np.linalg.inv(H)
    lines = [[(0, 0), (105, 0)], [(105, 0), (105, 68)], [(105, 68), (0, 68)], [(0, 68), (0, 0)],
             [(52.5, 0), (52.5, 68)], [(0, 13.84), (16.5, 13.84)], [(16.5, 13.84), (16.5, 54.16)],
             [(16.5, 54.16), (0, 54.16)], [(105, 13.84), (88.5, 13.84)], [(88.5, 13.84), (88.5, 54.16)],
             [(88.5, 54.16), (105, 54.16)]]
    for (x0, y0), (x1, y1) in lines:
        pts = np.array([[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, 1] for t in np.linspace(0, 1, 30)]).T
        q = Hinv @ pts
        q = (q[:2] / q[2]).T.astype(int)
        cv2.polylines(img, [q], False, (0, 255, 255), 2)
    c = Hinv @ np.array([[52.5 + 9.15 * np.cos(t), 34 + 9.15 * np.sin(t), 1] for t in np.linspace(0, 2 * np.pi, 60)]).T
    cv2.polylines(img, [(c[:2] / c[2]).T.astype(int)], True, (0, 255, 255), 2)
    cv2.imwrite(str(out), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--tmp-dir", default=".tmp")
    ap.add_argument("--npz", required=True, help="calibrated frame_homographies.npz that holds --anchor-frame")
    ap.add_argument("--out", help="npz to write (default: update --npz in place)")
    ap.add_argument("--detections", help="detections parquet (default: .tmp/<id>/detections.parquet)")
    ap.add_argument("--start-frame", type=int, required=True)
    ap.add_argument("--anchor-frame", type=int, required=True, help="first calibrated frame after the shot")
    ap.add_argument("--min-inliers", type=int, default=150)
    ap.add_argument("--check-frames", default="", help="comma-separated frames to draw the pitch on")
    a = ap.parse_args()

    meta = load_meta(a.match_id, a.tmp_dir)
    z = dict(np.load(a.npz))
    fr = z["frames"]
    k_anchor = np.searchsorted(fr, a.anchor_frame)
    if k_anchor >= len(fr) or fr[k_anchor] != a.anchor_frame:
        raise SystemExit(f"frame {a.anchor_frame} is not in {a.npz}")
    det_p = Path(a.detections) if a.detections else match_dir(a.match_id, a.tmp_dir) / "detections.parquet"
    det = pd.read_parquet(det_p, columns=["frame", "x1", "y1", "x2", "y2"],
                          filters=[("frame", ">=", a.start_frame), ("frame", "<=", a.anchor_frame)])
    boxes = {int(f) - a.start_frame: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in det.groupby("frame")}

    print(f"[{a.match_id}] chaining frames {a.start_frame}-{a.anchor_frame - 1} back from {a.anchor_frame}", flush=True)
    frames = read_frames(meta["video"], a.start_frame, a.anchor_frame)
    H, inl = chain(frames, boxes, z["H"][k_anchor], a.min_inliers)
    if not H:
        raise SystemExit("nothing chained")
    idx = sorted(H)
    new_f = np.array([a.start_frame + i for i in idx])
    print(f"  chained {len(idx)} frames ({new_f[0]}-{new_f[-1]}); inliers min {min(inl.values())}, "
          f"median {int(np.median(list(inl.values())))}")

    part = {"frames": new_f, "H": np.stack([H[i] for i in idx]),
            "params": np.full((len(idx), 3), np.nan), "n_pts": np.array([inl[i] for i in idx], np.int32),
            "frac2": np.full(len(idx), np.nan, np.float32), "med_px": np.full(len(idx), np.nan, np.float32),
            "status": np.ones(len(idx), np.int8)}
    keep = ~np.isin(fr, new_f)
    out = {k: np.concatenate([z[k][keep], part[k]]) for k in part if k in z}
    order = np.argsort(out["frames"], kind="stable")
    out = {k: v[order] for k, v in out.items()}
    dst = Path(a.out or a.npz)
    np.savez_compressed(dst, **out)
    print(f"  wrote {dst} ({len(out['frames'])} frames)")

    if a.check_frames:
        cdir = dst.parent / "chain_check"
        cdir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(meta["video"]))
        for f in [int(x) for x in a.check_frames.split(",") if x.strip()]:
            if f - a.start_frame not in H:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                draw_check(img, H[f - a.start_frame], cdir / f"f{f}.jpg")
        cap.release()
        print(f"  check images in {cdir}")


if __name__ == "__main__":
    main()
