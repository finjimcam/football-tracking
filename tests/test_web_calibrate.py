"""tools/_web_calibrate.py: the browser calibration page's live fit and overlay."""
import numpy as np

import _web_calibrate as wc


def _points(H_pitch_to_img):
    pitch = [(0, 0), (105, 0), (105, 68), (0, 68), (52.5, 34), (16.5, 13.84)]
    out = []
    for x, y in pitch:
        u = H_pitch_to_img @ np.array([x, y, 1.0])
        out.append({"image": [u[0] / u[2], u[1] / u[2]], "pitch": [x, y]})
    return out


def test_try_solve_recovers_the_homography():
    Hp2i = np.array([[9.0, 1.5, 200.0], [0.3, 6.0, 150.0], [0.0, 0.004, 1.0]])
    H, err = wc.try_solve(_points(Hp2i))
    assert H is not None and err.max() < 1e-3
    assert wc.try_solve(_points(Hp2i)[:3]) == (None, None)  # fewer than 4 points: no fit yet


def test_project_lines_splits_behind_the_camera():
    lines = wc.pitch_lines(105.0, 68.0)
    assert len(lines) > 10 and all(len(line) >= 2 for line in lines)
    out = wc.project_lines(np.eye(3), [[[0, 0], [10, 0]]])  # identity: pitch metres = pixels
    assert out == [[[0.0, 0.0], [10.0, 0.0]]]
    H = np.linalg.inv(np.array([[1.0, 0, 0], [0, 1.0, 0], [-0.1, 0, 1.0]]))  # w <= 0 from x >= 10
    out = wc.project_lines(H, [[[0, 0], [5, 0], [12, 0], [20, 0], [30, 0]]])
    assert len(out) == 1 and len(out[0]) == 2  # the points behind the camera are dropped
