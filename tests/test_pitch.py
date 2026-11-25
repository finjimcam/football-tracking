import math

import cv2
import numpy as np

import _pitch as P

C_TRUE = np.array([52.5, 120.8, -33.8])  # the Etihad tactical camera solved in the 2026-11-03 run
ROLL = math.radians(-0.65)
PP = (640.0, 360.0)


def H_of(pan_deg, tilt_deg, f):
    return P.world_to_image_h(C_TRUE, ROLL, math.radians(pan_deg), math.radians(tilt_deg), f, PP)


def test_polylines_cover_standard_markings():
    lines = P.pitch_polylines()
    assert {"touchline_far", "halfway", "centre_circle", "penalty_left", "goal_area_right",
            "penalty_arc_left"} <= set(lines)
    pts = P.sample_polylines(lines, step=1.0)
    assert pts[:, 0].min() >= -1e-9 and pts[:, 0].max() <= 105 + 1e-9
    assert pts[:, 1].min() >= -1e-9 and pts[:, 1].max() <= 68 + 1e-9


def test_decompose_homography_recovers_camera():
    f, R, C = P.decompose_homography(H_of(-13, 24, 1600), PP)
    assert abs(f - 1600) / 1600 < 1e-3
    assert np.allclose(C, C_TRUE, atol=0.05)


def test_angles_from_rotation_roundtrip():
    R = P.rotation(math.radians(10), math.radians(22), math.radians(-0.5))
    assert np.allclose(P.rotation(*P.angles_from_rotation(R)), R, atol=1e-6)


def test_centre_spot_projects_into_the_kickoff_view():
    uv, ok = P.project(H_of(0.6, 21.3, 1619), [[52.5, 34.0]])
    assert ok[0] and 0 < uv[0, 0] < 1280 and 0 < uv[0, 1] < 720


def test_line_pixels_find_paint_and_ignore_mowing_stripes():
    img = np.zeros((200, 300, 3), np.uint8)
    img[:] = (40, 140, 40)
    img[:, 150:] = (45, 155, 45)  # broad stripe edge: not a line
    cv2.line(img, (20, 100), (280, 110), (235, 235, 235), 2)
    m = P.line_pixels(img, region=np.ones(img.shape[:2], np.uint8))
    assert m[98:114, 40:260].sum() > 100
    assert m[:80, :].sum() == 0


def test_distance_map_and_bilinear_sampling():
    m = np.zeros((50, 50), np.uint8)
    m[25, :] = 1
    vals, ok = P.sample_map(P.distance_map(m), np.array([[10.0, 25.0], [10.0, 30.0], [-5.0, 0.0]]), fill=99.0)
    assert ok.tolist() == [True, True, False]
    assert vals[0] < 0.5 and 4.5 < vals[1] < 5.5 and vals[2] == 99.0


def test_projected_pitch_region_covers_the_view():
    region = P.projected_pitch_region(H_of(0.6, 21.3, 1619), (720, 1280), 105, 68)
    assert region.mean() > 0.3
