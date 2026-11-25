import math

import numpy as np

import _pitch as P
from calibrate_auto import Camera, fit_seed, frame_evidence, refine

PP = (640.0, 360.0)
C_TRUE, ROLL = [52.5, 120.8, -33.8], math.radians(-0.65)


def synthetic_frame(cam, x):
    """Green canvas with the pitch lines painted where the camera sees them."""
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:] = (40, 140, 40)
    P.draw_pitch(img, cam.H(x), color=(235, 235, 235), thickness=2)
    return img


def test_fit_seed_recovers_the_camera_from_landmark_clicks():
    cam = Camera(C_TRUE, ROLL, PP)
    x = (math.radians(-13), math.radians(24.3), math.log(1608))
    pit = np.array([(52.5, 0), (52.5, 68), (88.5, 13.84), (88.5, 54.16),
                    (105, 13.84), (105, 0), (99.5, 24.84), (99.5, 43.16)], float)
    img, _ = P.project(cam.H(x), pit)
    keyframes = [{"frame": 0, "points": [{"image": list(i), "pitch": list(p)} for i, p in zip(img, pit)]}]
    fitted, views, _, err = fit_seed(keyframes, PP)
    assert err.max() < 0.5
    assert np.allclose(fitted.C, C_TRUE, atol=0.5)


def test_refine_snaps_a_perturbed_view_back_onto_the_lines():
    cam = Camera(C_TRUE, ROLL, PP)
    x_true = np.array([math.radians(0.6), math.radians(21.3), math.log(1619)])
    dt, valid = frame_evidence(synthetic_frame(cam, x_true), np.ones((720, 1280), np.uint8), None)
    model = P.sample_polylines(P.pitch_polylines(), step=0.75)
    x0 = x_true + np.array([math.radians(0.1), math.radians(-0.1), 0.005])  # a few px off
    x, q = refine(dt, valid, cam, x0, model)
    assert q["frac2"] > 0.8
    assert abs(x[0] - x_true[0]) < math.radians(0.03)
    assert abs(x[1] - x_true[1]) < math.radians(0.03)
