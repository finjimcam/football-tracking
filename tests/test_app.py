"""Tests for the Stopframe web app (app/): param whitelisting, match status,
the job runner (real subprocesses on a stub tool) and the HTTP API."""
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import catalog, matches  # noqa: E402
from app.catalog import ParamError, Stage, build_args, get_stage  # noqa: E402
from app.jobs import JobRunner, parse_progress  # noqa: E402

STUB = """\
import sys, time
args = sys.argv[1:]
print("hello", " ".join(args), flush=True)
for i in range(1, 4):
    print(f"PROGRESS {i}/3 step", flush=True)
if "--sleep" in args:
    time.sleep(30)
if "--fail" in args:
    print("boom", file=sys.stderr)
    sys.exit(3)
"""


@pytest.fixture
def root(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "stub.py").write_text(STUB)
    (tmp_path / "videos").mkdir()
    return tmp_path


def wait_for(runner, job_id, states, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        job = runner.get(job_id)
        if job["status"] in states:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck in {runner.get(job_id)['status']}")


def make_match(root, match_id="m_1234", video="videos/a.mp4", **files):
    d = root / ".tmp" / match_id
    d.mkdir(parents=True)
    meta = {"match_id": match_id, "video": str((root / video).resolve()), "fps": 25.0,
            "frame_count": 250, "duration_s": 10.0, "width": 1920, "height": 1080,
            "config": {"team_a_name": "Reds"}}
    (d / "meta.json").write_text(json.dumps(meta))
    for name, content in files.items():
        (d / name).write_text(content)
    return match_id


# ---- catalog ----------------------------------------------------------

def test_build_args_basic_and_types():
    stage = get_stage("possession")
    args = build_args(stage, {"radius_m": "1.5", "switch_s": ""}, match_id="m_1")
    assert args == ["--match-id", "m_1", "--radius-m", "1.5"]
    with pytest.raises(ParamError):
        build_args(stage, {"radius_m": "far"}, match_id="m_1")


def test_build_args_rejects_unknown_and_bad_match_id():
    stage = get_stage("possession")
    with pytest.raises(ParamError, match="Unknown parameter"):
        build_args(stage, {"output": "/etc/passwd"}, match_id="m_1")
    for bad in ("", "..", "../x", "a b", "a;rm"):
        with pytest.raises(ParamError):
            build_args(stage, {}, match_id=bad)


def test_build_args_flags_and_int():
    args = build_args(get_stage("render"), {"hide_unidentified": True, "scale": 0.5}, match_id="m")
    assert "--hide-unidentified" in args and args[args.index("--scale") + 1] == "0.5"
    args = build_args(get_stage("render"), {"hide_unidentified": False}, match_id="m")
    assert "--hide-unidentified" not in args
    with pytest.raises(ParamError):
        build_args(get_stage("track"), {"imgsz": "12.5"}, match_id="m")


def test_ingest_video_must_be_inside_videos(root):
    (root / "videos" / "a.mp4").write_bytes(b"x")
    (root / "outside.mp4").write_bytes(b"x")
    stage = get_stage("ingest")
    args = build_args(stage, {"video": "a.mp4", "team_a_name": "Reds"}, videos_dir=root / "videos")
    assert args[:2] == ["--video", str((root / "videos" / "a.mp4").resolve())]
    assert json.loads(args[args.index("--config") + 1]) == {"team_a_name": "Reds"}
    for bad in ("../outside.mp4", "missing.mp4", str(root / "outside.mp4")):
        with pytest.raises(ParamError):
            build_args(stage, {"video": bad}, videos_dir=root / "videos")
    with pytest.raises(ParamError, match="required"):
        build_args(stage, {}, videos_dir=root / "videos")


def test_track_gets_video_from_meta():
    args = build_args(get_stage("track"), {}, match_id="m", meta_video="/v/a.mp4")
    assert args[:4] == ["--match-id", "m", "--video", "/v/a.mp4"]
    with pytest.raises(ParamError, match="re-ingest"):
        build_args(get_stage("track"), {}, match_id="m")


def test_progress_helper_is_throttled(capsys):
    sys.path.insert(0, str(ROOT / "tools"))
    from _common import Progress
    prog = Progress(1000, "x")
    for i in range(1, 1001):
        prog.update(i)
    prog.done()
    lines = capsys.readouterr().out.splitlines()
    assert 150 <= len(lines) <= 205
    assert lines[-1] == "PROGRESS 1000/1000 x"
    assert all(parse_progress(line) for line in lines)


def test_interactive_stage_cannot_run():
    with pytest.raises(ParamError, match="interactive"):
        build_args(get_stage("calibrate"), {}, match_id="m")


# ---- match status -----------------------------------------------------

def test_match_status_from_artifacts(root):
    (root / "videos" / "a.mp4").write_bytes(b"x")
    (root / "videos" / "b.mp4").write_bytes(b"x")
    mid = make_match(root)
    m = matches.match_summary(root, mid)
    assert m["team_a_name"] == "Reds" and m["team_b_name"] == "Team B"
    assert m["stages"]["track"]["state"] == "ready"
    assert m["stages"]["teams"]["state"] == "blocked"
    assert m["stages"]["camera_motion"]["state"] == "blocked"

    (root / ".tmp" / mid / "tracks.parquet").write_bytes(b"x")
    m = matches.match_summary(root, mid)
    assert m["stages"]["track"]["state"] == "done"
    assert m["stages"]["teams"]["state"] == "ready"

    vids = {v["name"]: v["match_id"] for v in matches.list_videos(root)}
    assert vids == {"a.mp4": mid, "b.mp4": None}


def test_stale_when_input_newer(root):
    mid = make_match(root, **{"tracks.parquet": "x", "team_map.json": "{}"})
    d = root / ".tmp" / mid
    now = time.time()
    import os
    os.utime(d / "team_map.json", (now - 100, now - 100))
    os.utime(d / "tracks.parquet", (now, now))
    assert matches.match_summary(root, mid)["stages"]["teams"]["state"] == "stale"


def test_camera_motion_depends_on_calibration_mode(root):
    mid = make_match(root)
    calib = root / "config" / f"{mid}_calibration.json"
    calib.parent.mkdir()
    for mode, want in (("static", "skip"), ("keyframes", "ready"), ("dynamic", "done")):
        calib.write_text(json.dumps({"mode": mode}))
        assert matches.match_summary(root, mid)["stages"]["camera_motion"]["state"] == want


# ---- job runner -------------------------------------------------------

STUB_STAGE = Stage("stub", "Stub", "stub.py", 1, "test", per_match=False)
HEAVY_STAGE = Stage("stubh", "Stub heavy", "stub.py", 1, "test", heavy=True, per_match=False)


def test_parse_progress():
    assert parse_progress("PROGRESS 5/10 tracking") == {"done": 5, "total": 10, "label": "tracking"}
    assert parse_progress("PROGRESS 5 / 10") == {"done": 5, "total": 10, "label": ""}
    assert parse_progress("progress was made") is None


def test_job_runs_logs_and_progress(root):
    r = JobRunner(root, nice_heavy=False)
    job = r.submit(STUB_STAGE, ["--x", "1"])
    job = wait_for(r, job["id"], {"done", "failed"})
    assert job["status"] == "done" and job["returncode"] == 0
    assert job["progress"] == {"done": 3, "total": 3, "label": "step"}
    log, _ = r.read_log(job["id"])
    assert "hello --x 1" in log and "PROGRESS" not in log and "[exit 0]" in log
    # history survives a restart
    r2 = JobRunner(root, nice_heavy=False)
    assert r2.get(job["id"])["status"] == "done"


def test_job_failure_captures_stderr(root):
    r = JobRunner(root, nice_heavy=False)
    job = wait_for(r, r.submit(STUB_STAGE, ["--fail"])["id"], {"done", "failed"})
    assert job["status"] == "failed" and job["returncode"] == 3
    assert "boom" in r.read_log(job["id"])[0]


def test_cancel_running_job(root):
    r = JobRunner(root, nice_heavy=False)
    job = r.submit(STUB_STAGE, ["--sleep"])
    wait_for(r, job["id"], {"running"})
    time.sleep(0.3)
    assert r.cancel(job["id"])
    job = wait_for(r, job["id"], {"cancelled", "failed", "done"}, timeout=10)
    assert job["status"] == "cancelled"


def test_busy_check_holds_heavy_jobs_until_forced(root):
    flag = root / "busy"
    flag.write_text("1")
    check = [sys.executable, "-c", f"import os,sys; sys.exit(0 if os.path.exists({str(flag)!r}) else 1)"]
    r = JobRunner(root, busy_check=check, nice_heavy=False)
    heavy = r.submit(HEAVY_STAGE, [])
    light = r.submit(STUB_STAGE, [])
    assert wait_for(r, light["id"], {"done"})["status"] == "done"  # light lane unaffected
    assert wait_for(r, heavy["id"], {"waiting"})["note"]
    assert r.force(heavy["id"])
    assert wait_for(r, heavy["id"], {"done"})["status"] == "done"


def test_interrupted_on_restart(root):
    jobs = root / ".tmp" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "old.json").write_text(json.dumps({"id": "old", "status": "running", "created": 1}))
    r = JobRunner(root, nice_heavy=False)
    assert r.get("old")["status"] == "interrupted"


# ---- HTTP API ---------------------------------------------------------

@pytest.fixture
def client(root):
    from fastapi.testclient import TestClient

    from app.server import create_app
    return TestClient(create_app(root, runner=JobRunner(root, nice_heavy=False)))


def test_api_stages_and_matches(client, root):
    stages = client.get("/api/stages").json()
    assert [s["id"] for s in stages][:2] == ["ingest", "track"]
    mid = make_match(root)
    data = client.get("/api/matches").json()
    assert data["matches"][0]["match_id"] == mid
    assert client.get(f"/api/matches/{mid}").json()["jobs"] == []
    assert client.get("/api/matches/nope").status_code == 404
    assert client.get("/").status_code == 200


def test_api_rejects_bad_jobs(client, root):
    mid = make_match(root)
    assert client.post("/api/jobs", json={"stage": "nope"}).status_code == 400
    assert client.post("/api/jobs", json={"stage": "possession", "match_id": "missing"}).status_code == 404
    r = client.post("/api/jobs", json={"stage": "possession", "match_id": mid, "params": {"bogus": 1}})
    assert r.status_code == 400 and "Unknown parameter" in r.json()["detail"]


def test_api_files_only_whitelisted(client, root):
    mid = make_match(root)
    assert client.get(f"/api/matches/{mid}/files/meta").status_code == 404
    assert client.get(f"/api/matches/{mid}/files/report").status_code == 404
    rep = root / "outputs" / "reports" / f"{mid}_report.pdf"
    rep.parent.mkdir(parents=True)
    rep.write_bytes(b"%PDF-1.4")
    r = client.get(f"/api/matches/{mid}/files/report")
    assert r.status_code == 200 and r.content == b"%PDF-1.4"


def test_api_runs_a_job_end_to_end(client, root, monkeypatch):
    mid = make_match(root)
    monkeypatch.setitem(catalog.BY_ID, "stub", Stage("stub", "Stub", "stub.py", 1, "t"))
    job = client.post("/api/jobs", json={"stage": "stub", "match_id": mid}).json()
    runner = client.app.state.runner
    wait_for(runner, job["id"], {"done", "failed"})
    body = client.get(f"/api/jobs/{job['id']}").json()
    assert body["status"] == "done" and f"--match-id {mid}" in body["log"]
    tail = client.get(f"/api/jobs/{job['id']}?offset={body['offset']}").json()
    assert tail["log"] == ""
