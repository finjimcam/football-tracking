"""Issue #37: choose which phase clips to export, trim each one's start/end,
and get one MP4 per phase in a zip (for PowerPoint)."""
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from test_lines import FPS, MID, N, build_synthetic  # noqa: E402

from cut_phase_clips import export_range  # noqa: E402


@pytest.fixture
def root(tmp_path):
    r = build_synthetic(tmp_path)
    # two City out-of-possession phases: Spurs on the ball 100-299 and 380-469
    state = ["team_b" if (100 <= f < 300 or 380 <= f < 470) else "team_a" for f in range(N)]
    pd.DataFrame({"frame": range(N), "state": state}).to_parquet(
        r / ".tmp" / MID / "possession_timeline.parquet")
    return r


@pytest.fixture
def client(root):
    from fastapi.testclient import TestClient

    from app.jobs import JobRunner
    from app.server import create_app
    return TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))


def test_export_range_clamps_and_ignores_crossed_trims():
    assert export_range((100, 200), None) == (True, 100, 200)
    assert export_range((100, 200), {"keep": False}) == (False, 100, 200)
    assert export_range((100, 200), {"in": 50, "out": 150}) == (True, 100, 150)
    assert export_range((100, 200), {"in": 180, "out": 120}) == (True, 100, 200)


def test_keep_drop_and_trim_api(client, root):
    base = f"/api/matches/{MID}/lines"
    ph = client.get(base).json()["phases"]
    assert len(ph) == 2 and all(p["export"]["keep"] and not p["export"]["trimmed"] for p in ph)
    p1, p2 = ph
    r = client.put(f"{base}/{p1['start']}/export", json={"keep": False})
    assert r.status_code == 200 and r.json()["export"]["keep"] is False
    r = client.put(f"{base}/{p2['start']}/export", json={"trim_in": p2["start"] + 10, "trim_out": p2["end"] - 20})
    ex = r.json()["export"]
    assert (ex["in"], ex["out"], ex["trimmed"]) == (p2["start"] + 10, p2["end"] - 20, True)
    # outside the phase / shorter than a second -> 400, nothing saved
    assert client.put(f"{base}/{p2['start']}/export", json={"trim_in": p2["end"] + 5}).status_code == 400
    assert client.put(f"{base}/{p2['start']}/export", json={"trim_in": p2["end"] - 30}).status_code == 400
    after = client.get(base).json()["phases"]
    assert after[0]["export"]["keep"] is False and after[1]["export"]["in"] == p2["start"] + 10
    # keep all / drop all keep the trims
    client.put(f"{base}/export", json={"keep": False})
    assert not any(p["export"]["keep"] for p in client.get(base).json()["phases"])
    client.put(f"{base}/export", json={"keep": True})
    after = client.get(base).json()["phases"]
    assert all(p["export"]["keep"] for p in after) and after[1]["export"]["trimmed"]
    # "Full phase" undoes the trim
    ex = client.put(f"{base}/{p2['start']}/export", json={"reset_trim": True}).json()["export"]
    assert (ex["in"], ex["out"], ex["trimmed"]) == (p2["start"], p2["end"], False)
    assert (root / "config" / f"{MID}_team_a_out_of_possession_export.json").exists()


def test_export_stage_args():
    from app import catalog
    st = catalog.get_stage("b_clip_export")
    args = catalog.build_args(st, {"team": "team_b"}, match_id=MID)
    assert args[:2] == ["--match-id", MID]
    assert {"--shape-selection", "--export-selection", "--separate"} <= set(args)
    assert args[-2:] == ["--team", "team_b"]


def test_separate_export_zips_only_kept_trimmed_clips(client, root):
    base = f"/api/matches/{MID}/lines"
    p1, p2 = client.get(base).json()["phases"]
    client.put(f"{base}/{p1['start']}/export", json={"keep": False})
    a, b = p2["start"] + 10, p2["start"] + 10 + 2 * FPS - 1  # a 2-second clip
    client.put(f"{base}/{p2['start']}/export", json={"trim_in": a, "trim_out": b})
    # leftovers from an earlier export must not survive
    old = root / "outputs" / "videos" / f"{MID}_team_a_out_of_possession_shape_selected_clips"
    old.mkdir(parents=True)
    (old / "phase_01_old.mp4").write_bytes(b"x")
    run = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "cut_phase_clips.py"), "--match-id", MID,
         "--tmp-dir", str(root / ".tmp"), "--config-dir", str(root / "config"),
         "--shape-selection", "--export-selection", "--separate"],
        cwd=root, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr + run.stdout
    zpath = old.with_suffix(".zip")
    assert zpath.exists() and not old.exists()
    with zipfile.ZipFile(zpath) as z:
        names = sorted(z.namelist())
        assert names == ["phase_02_" + "{:02d}m{:02d}s".format(*divmod(380 // FPS, 60)) + ".mp4", "phases.csv"]
        z.extractall(root / "unz")
    cap = cv2.VideoCapture(str(root / "unz" / names[0]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == b - a + 1
    idx = pd.read_csv(root / "unz" / "phases.csv")
    assert idx["phase"].tolist() == [2] and idx["file"].tolist() == [names[0]]
    assert (idx["start_frame"].iat[0], idx["end_frame"].iat[0]) == (a, b)
    # the match page can now offer the zip
    r = client.get(f"/api/matches/{MID}/files/clips_zip")
    assert r.status_code == 200 and r.content[:2] == b"PK"


def test_dropping_everything_fails_clearly(client, root):
    client.put(f"/api/matches/{MID}/lines/export", json={"keep": False})
    run = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "cut_phase_clips.py"), "--match-id", MID,
         "--tmp-dir", str(root / ".tmp"), "--config-dir", str(root / "config"),
         "--export-selection", "--separate"],
        cwd=root, capture_output=True, text=True)
    assert run.returncode != 0 and "keep at least one" in (run.stderr + run.stdout)
