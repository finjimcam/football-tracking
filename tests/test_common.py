import numpy as np
import pandas as pd
import pytest

import _common as C


def test_match_id_is_path_safe(tmp_path):
    video = tmp_path / "Man City vs Spurs (1).mp4"
    video.write_bytes(b"not really a video")
    mid = C.compute_match_id(video)
    assert mid.startswith("Man_City_vs_Spurs_1_")
    assert all(ch.isalnum() or ch in "_-" for ch in mid)


def test_match_id_unchanged_for_clean_stems(tmp_path):
    video = tmp_path / "testvideo.mp4"
    video.write_bytes(b"abc")
    mid = C.compute_match_id(video)
    assert mid.startswith("testvideo_") and len(mid.split("_")[-1]) == 8


def test_project_points_identity():
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(C.project_points(np.eye(3), xy), xy)


def test_apply_calibration_dynamic_matches_per_frame_projection():
    rng = np.random.default_rng(0)
    frames = np.arange(100, 160)
    Hs = np.stack([np.array([[1 + 0.01 * k, 0.02, k], [0.01, 1.0, -k], [1e-4, 2e-4, 1.0]])
                   for k in range(len(frames))])
    df = pd.DataFrame({"frame": rng.integers(100, 160, 500),
                       "foot_x": rng.uniform(0, 1280, 500), "foot_y": rng.uniform(0, 720, 500)})
    out, unit = C.apply_calibration(df, {"frame_index": frames, "frame_H": Hs})
    assert unit == "m"
    for r in out.sample(25, random_state=1).itertuples():
        expected = C.project_points(Hs[r.frame - 100], [[r.foot_x, r.foot_y]])[0]
        assert np.allclose([r.pitch_x, r.pitch_y], expected)


def test_apply_calibration_uncalibrated_passthrough():
    df = pd.DataFrame({"frame": [0], "foot_x": [5.0], "foot_y": [6.0]})
    out, unit = C.apply_calibration(df, None)
    assert unit == "px" and out["pitch_x"].iat[0] == 5.0 and out["pitch_y"].iat[0] == 6.0


def test_clock_label():
    cfg = {"clock_sync": {"kickoff_frame": 834, "fps": 25}}
    assert C.clock_label(834 + 25 * 236, cfg) == "03:56"  # Kulusevski's goal, Opta 236 s
    assert C.clock_label(100, cfg) is None                 # before kick-off
    assert C.clock_label(100, {}) is None


def test_resolve_track_teams_identity_flip_and_ignore():
    team_map = {1: "team_a", 2: "team_b", 3: "team_a"}
    pm = {"track_to_player": {2: "a7"}, "players": {"a7": {"team": "team_a"}},
          "ignored_tracks": [{"track_id": 3}], "team_map_alignment": "flipped"}
    assert C.resolve_track_teams([1, 2, 3], team_map, pm) == {1: "team_b", 2: "team_a", 3: "ignored"}


def test_video_writer_roundtrip(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "x.mp4"
    w = C.VideoWriter(path, 25, 64, 48)
    for k in range(10):
        w.write(np.full((48, 64, 3), k * 20, np.uint8))
    w.close()
    assert w.codec in ("h264", "mp4v")
    with av.open(str(path)) as c:
        assert sum(1 for _ in c.decode(video=0)) == 10
