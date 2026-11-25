"""Pitch-line model + fixed-position PTZ camera geometry (not a CLI tool).

Shared by tools/calibrate_auto.py and the stages that render or verify
calibration. Conventions match tools/calibrate_pitch.py / _common.py:

  pitch x: along the length, 0 = left goal line as seen from the camera, L = right
  pitch y: across, 0 = FAR touchline (top of the image), W = NEAR touchline
  z      : DOWN (the camera sits at negative z). x right / y towards the
           camera / z down is right-handed, so the "TV view" pitch map
           (origin top-left, y downward) needs no mirroring.

Camera model: one fixed optical centre C (a broadcast tactical camera on the
main stand) that only pans, tilts and zooms. Globals: C (3), roll (1).
Per frame: pan, tilt, focal length f (pixels). Principal point = image
centre, square pixels, no lens distortion.

    world -> camera:  p_cam = R(pan, tilt, roll) (P - C)
    R = Rz_cam(roll) @ Rx_cam(tilt) @ R_BASE @ Rz_world(pan)

R_BASE is the camera looking horizontally at the far touchline (-y) with
image-right = +x and image-down = +z. Positive tilt looks down.
"""
import math

import cv2
import numpy as np

R_BASE = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [0.0, -1.0, 0.0]])

PENALTY_DEPTH, PENALTY_HALF_W = 16.5, 40.32 / 2
GOAL_AREA_DEPTH, GOAL_AREA_HALF_W = 5.5, 18.32 / 2
CIRCLE_R, SPOT_DIST = 9.15, 11.0


# --------------------------------------------------------------------------
# Pitch model
# --------------------------------------------------------------------------
def pitch_polylines(L=105.0, W=68.0, arc_step_deg=4.0):
    """Every painted pitch line as a named polyline in pitch metres."""
    cy = W / 2
    lines = {
        "touchline_far": [(0, 0), (L, 0)],
        "touchline_near": [(0, W), (L, W)],
        "goal_line_left": [(0, 0), (0, W)],
        "goal_line_right": [(L, 0), (L, W)],
        "halfway": [(L / 2, 0), (L / 2, W)],
    }
    for side, x0, s in (("left", 0.0, 1.0), ("right", L, -1.0)):
        pd_, gd = x0 + s * PENALTY_DEPTH, x0 + s * GOAL_AREA_DEPTH
        lines[f"penalty_{side}"] = [(x0, cy - PENALTY_HALF_W), (pd_, cy - PENALTY_HALF_W),
                                    (pd_, cy + PENALTY_HALF_W), (x0, cy + PENALTY_HALF_W)]
        lines[f"goal_area_{side}"] = [(x0, cy - GOAL_AREA_HALF_W), (gd, cy - GOAL_AREA_HALF_W),
                                      (gd, cy + GOAL_AREA_HALF_W), (x0, cy + GOAL_AREA_HALF_W)]
        # Penalty arc: the part of the spot-centred circle outside the box.
        spot = x0 + s * SPOT_DIST
        half = math.degrees(math.acos((PENALTY_DEPTH - SPOT_DIST) / CIRCLE_R))
        base = 0.0 if s > 0 else 180.0
        angs = np.radians(np.arange(base - half, base + half + 1e-6, arc_step_deg / 2))
        lines[f"penalty_arc_{side}"] = [(spot + CIRCLE_R * math.cos(a), cy + CIRCLE_R * math.sin(a))
                                        for a in angs]
    angs = np.radians(np.arange(0, 360 + 1e-6, arc_step_deg))
    lines["centre_circle"] = [(L / 2 + CIRCLE_R * math.cos(a), cy + CIRCLE_R * math.sin(a)) for a in angs]
    return {k: np.asarray(v, dtype=np.float64) for k, v in lines.items()}


def sample_polylines(polylines, step=0.5):
    """Dense points along every polyline (pitch metres) -> (N, 2)."""
    pts = []
    for poly in polylines.values():
        for a, b in zip(poly[:-1], poly[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + t * (b - a))
        pts.append(poly[-1:])
    return np.vstack(pts)


# --------------------------------------------------------------------------
# Camera model
# --------------------------------------------------------------------------
def rotation(pan, tilt, roll):
    cp, sp = math.cos(pan), math.sin(pan)
    ct, st = math.cos(tilt), math.sin(tilt)
    cr, sr = math.cos(roll), math.sin(roll)
    rz_world = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    rx_cam = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    rz_cam = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz_cam @ rx_cam @ R_BASE @ rz_world


def world_to_image_h(C, roll, pan, tilt, f, pp):
    """3x3 homography: pitch (x, y, 1) -> image pixels (homogeneous)."""
    R = rotation(pan, tilt, roll)
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    C = np.asarray(C, dtype=np.float64)
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def project(Hw, pts):
    """Pitch points (N, 2) -> image pixels (N, 2) (NaN behind the camera), mask."""
    pts = np.asarray(pts, dtype=np.float64)
    h = np.column_stack([pts, np.ones(len(pts))]) @ Hw.T
    ok = h[:, 2] > 1e-9
    out = np.full((len(pts), 2), np.nan)
    out[ok] = h[ok, :2] / h[ok, 2:3]
    return out, ok


def decompose_homography(Hw, pp):
    """Pitch->image homography -> (f, R, C) for a camera with square pixels and
    a known principal point. None if the focal length isn't recoverable."""
    T = np.array([[1, 0, -pp[0]], [0, 1, -pp[1]], [0, 0, 1.0]])
    H = T @ Hw
    h11, h21, h31 = H[:, 0]
    h12, h22, h32 = H[:, 1]
    cands = []
    if abs(h31 * h32) > 1e-12:
        f2 = -(h11 * h12 + h21 * h22) / (h31 * h32)
        if f2 > 0:
            cands.append(f2)
    den = h31 ** 2 - h32 ** 2
    if abs(den) > 1e-12:
        f2 = (h12 ** 2 + h22 ** 2 - h11 ** 2 - h21 ** 2) / den
        if f2 > 0:
            cands.append(f2)
    if not cands:
        return None
    f = math.sqrt(float(np.median(cands)))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    a1, a2, a3 = Kinv @ H[:, 0], Kinv @ H[:, 1], Kinv @ H[:, 2]
    lam = 2.0 / (np.linalg.norm(a1) + np.linalg.norm(a2))
    r1, r2, t = lam * a1, lam * a2, lam * a3
    if t[2] < 0:  # the pitch must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    return f, R, -R.T @ t


def angles_from_rotation(R):
    """(pan, tilt, roll) whose rotation() best matches R (numeric fit)."""
    from scipy.optimize import least_squares

    best = None
    for pan0 in np.radians([-60, -20, 0, 20, 60]):
        res = least_squares(lambda p: (rotation(*p) - R).ravel(), x0=[pan0, 0.4, 0.0])
        if best is None or res.cost < best.cost:
            best = res
    return tuple(best.x)


# --------------------------------------------------------------------------
# Image side: line pixels + distance maps
# --------------------------------------------------------------------------
def line_pixels(frame_bgr, ridge_thresh=7, region=None, exclude_boxes=None, box_pad=3):
    """uint8 mask of painted-line pixels: thin structures brighter than their
    surroundings (morphological top-hat), unsaturated, inside `region`
    (default: the largest grass blob). `exclude_boxes` (N, 4) masks players,
    whose white kit otherwise reads as line.

    Top-hat (L minus its 9x9 opening) rather than L minus a median blur:
    identical line pixels on the pitch (both ignore the broad edges of the
    mowing stripes), ~20x faster (3-5 ms vs ~80 ms per 720p frame); its
    extra responses are crowd/boards/players, which the masks remove."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if region is None:
        grass = ((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 40)).astype(np.uint8)
        pitch = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(pitch)
        if n > 1:
            pitch = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        region = cv2.erode(pitch, np.ones((7, 7), np.uint8))
    L = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)[..., 0]
    ridge = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    m = ((ridge > ridge_thresh) & (hsv[..., 1] < 140) & (region > 0)).astype(np.uint8)
    if exclude_boxes is not None and len(exclude_boxes):
        h, w = m.shape
        for x1, y1, x2, y2 in exclude_boxes:
            m[max(int(y1) - box_pad, 0):min(int(y2) + box_pad, h),
              max(int(x1) - box_pad, 0):min(int(x2) + box_pad, w)] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 6
    return keep[lab].astype(np.uint8)


def distance_map(mask):
    """Distance (px) from every pixel to the nearest set pixel of `mask`."""
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)


def sample_map(dmap, xy, fill):
    """Bilinear sample of a distance map at float pixel coords; `fill` outside.
    Returns (values, inside-mask)."""
    h, w = dmap.shape
    x, y = xy[:, 0], xy[:, 1]
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= w - 2) & (y <= h - 2)
    out = np.full(len(xy), fill, dtype=np.float64)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    fx, fy = x[ok] - xi, y[ok] - yi
    out[ok] = (dmap[yi, xi] * (1 - fx) * (1 - fy) + dmap[yi, xi + 1] * fx * (1 - fy)
               + dmap[yi + 1, xi] * (1 - fx) * fy + dmap[yi + 1, xi + 1] * fx * fy)
    return out, ok


def projected_pitch_region(Hw, shape, L, W, margin=2.0):
    """uint8 mask of the image area covered by the pitch (+margin metres)."""
    corners = np.array([[-margin, -margin], [L + margin, -margin],
                        [L + margin, W + margin], [-margin, W + margin]])
    edge = [a + t * (b - a) for a, b in zip(corners, np.roll(corners, -1, axis=0))
            for t in np.linspace(0, 1, 40, endpoint=False)]
    img, ok = project(Hw, np.array(edge))
    region = np.zeros(shape[:2], np.uint8)
    pts = img[ok]
    if len(pts) >= 3:
        pts = np.clip(pts, -4 * shape[1], 4 * shape[1])
        cv2.fillPoly(region, [np.round(pts).astype(np.int32)], 1)
    return region


def draw_pitch(img, Hw, L=105.0, W=68.0, color=(0, 255, 255), thickness=1):
    """Overlay the projected pitch model on an image (in place)."""
    for poly in pitch_polylines(L, W).values():
        dense = sample_polylines({"p": poly}, step=0.5)
        pts, ok = project(Hw, dense)
        good = ok & np.all(np.abs(pts) < 1e5, axis=1)
        if good.sum() >= 2:
            seg = np.round(pts[good]).astype(np.int32)
            cv2.polylines(img, [seg], False, color, thickness, cv2.LINE_AA)
    return img
