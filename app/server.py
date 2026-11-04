"""FastAPI app: JSON API over the pipeline + the static front end.

    GET  /api/stages                      stage catalog (forms are built from it)
    GET  /api/matches                     matches + not-yet-ingested videos
    GET  /api/matches/{match_id}          one match with per-stage status
    GET  /api/matches/{match_id}/files/{key}   download stats / report / tracking video
    GET  /api/jobs[?match_id=]            job history, newest first
    POST /api/jobs                        {stage, match_id?, params{}, force?}
    GET  /api/jobs/{id}?offset=N          job record + log text from byte N
    POST /api/jobs/{id}/cancel | /force
    GET  /api/system                      disk space + whether heavy jobs are held

Everything else under / is the single-page app in app/static/.
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import catalog, matches
from app.jobs import JobRunner
from app.lines import FrameSource, LinesCache, LinesError

STATIC = Path(__file__).parent / "static"


class LinesEdit(BaseModel):
    team: str = "team_a"
    slots: list = Field(default_factory=list)
    action: str = "chains"
    frame: int | None = None
    tid: int | None = None
    chain: int | None = None


class LinesSave(BaseModel):
    team: str = "team_a"
    slots: list = Field(default_factory=list)
    keyframe: int | None = None
    skip: bool = False


class JobRequest(BaseModel):
    stage: str
    match_id: str | None = None
    params: dict = Field(default_factory=dict)
    force: bool = False


def _disk(path):
    try:
        u = shutil.disk_usage(path)
    except OSError:
        return None
    return {"path": str(path), "free": u.free, "total": u.total}


def create_app(root, runner=None, busy_check=None, warm=False):
    root = Path(root).resolve()
    runner = runner or JobRunner(root, busy_check=busy_check)
    app = FastAPI(title="Stopframe", docs_url="/api/docs", redoc_url=None)
    app.state.runner = runner
    app.state.root = root

    def _match_or_404(match_id):
        m = matches.match_summary(root, match_id) if _safe_id(match_id) else None
        if not m:
            raise HTTPException(404, f"No match '{match_id}'")
        return m

    @app.get("/api/stages")
    def stages():
        return [s.to_dict() for s in catalog.STAGES]

    @app.get("/api/matches")
    def list_matches():
        ms = matches.list_matches(root)
        active = {}
        for j in runner.list(limit=500):
            if j["status"] in ("queued", "waiting", "running") and j.get("match_id"):
                active.setdefault(j["match_id"], []).append(j["stage"])
        for m in ms:
            m["active_stages"] = active.get(m["match_id"], [])
        return {"matches": ms, "videos": matches.list_videos(root, ms)}

    @app.get("/api/matches/{match_id}")
    def get_match(match_id: str):
        m = _match_or_404(match_id)
        m["jobs"] = runner.list(match_id=match_id, limit=50)
        return m

    @app.get("/api/matches/{match_id}/files/{key}")
    def match_file(match_id: str, key: str):
        _match_or_404(match_id)
        if key not in matches.DOWNLOADABLE:
            raise HTTPException(404, "Not a downloadable output")
        path = matches.artifact_paths(root, match_id)[key]
        if not path.exists():
            raise HTTPException(404, "Not produced yet")
        return FileResponse(path, filename=path.name)

    lines_cache = LinesCache(root)
    frames = FrameSource()

    def _warm():
        """Load line-picker data for broadcast matches in the background, so the
        first visit to the Lines tab doesn't wait ~30 s reading a whole half."""
        for m in matches.list_matches(root):
            if m["pipeline"] == catalog.BROADCAST and m["artifacts"]["possession"]["exists"]:
                try:
                    lines_cache.get(m["match_id"], "team_a")
                except Exception:  # noqa: BLE001 - warming is best-effort
                    pass

    if warm:
        import threading
        threading.Thread(target=_warm, daemon=True, name="lines-warm").start()

    def _lines(match_id, team):
        _match_or_404(match_id)
        try:
            return lines_cache.get(match_id, team)
        except LinesError as e:
            raise HTTPException(409, str(e)) from None

    def _lines_call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except LinesError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/matches/{match_id}/lines")
    def lines_summary(match_id: str, team: str = "team_a"):
        return _lines(match_id, team).summary()

    @app.get("/api/matches/{match_id}/lines/{start}")
    def lines_phase(match_id: str, start: int, team: str = "team_a"):
        return _lines_call(_lines(match_id, team).phase, start)

    @app.post("/api/matches/{match_id}/lines/{start}/edit")
    def lines_edit(match_id: str, start: int, req: LinesEdit):
        return _lines_call(_lines(match_id, req.team).edit, start, req.slots, req.action, req.frame, req.tid,
                           req.chain)

    @app.put("/api/matches/{match_id}/lines/{start}")
    def lines_save(match_id: str, start: int, req: LinesSave):
        return _lines_call(_lines(match_id, req.team).save, start, req.slots, req.keyframe, req.skip)

    @app.get("/api/matches/{match_id}/frame/{frame}.jpg")
    def frame_jpg(match_id: str, frame: int, w: int | None = None):
        m = _match_or_404(match_id)
        if not m["video"] or frame < 0 or (m["frame_count"] and frame >= m["frame_count"]):
            raise HTTPException(404, "No such frame")
        if w is not None and not 160 <= w <= 3840:
            raise HTTPException(400, "w out of range")
        try:
            data = frames.jpeg(m["video"], frame, w)
        except LinesError as e:
            raise HTTPException(404, str(e)) from None
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/jobs")
    def list_jobs(match_id: str | None = None):
        return runner.list(match_id=match_id)

    @app.post("/api/jobs", status_code=201)
    def submit(req: JobRequest):
        try:
            stage = catalog.get_stage(req.stage)
            match = _match_or_404(req.match_id or "") if stage.per_match else {}
            args = catalog.build_args(stage, req.params, match_id=req.match_id,
                                      videos_dir=root / "videos", meta_video=match.get("video"),
                                      config_dir=root / "config")
        except catalog.ParamError as e:
            raise HTTPException(400, str(e)) from None
        return runner.submit(stage, args, match_id=req.match_id if stage.per_match else None,
                             force=req.force)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, offset: int = 0):
        job = runner.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        text, new_offset = runner.read_log(job_id, offset)
        job["log"] = text
        job["offset"] = new_offset
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        if not runner.cancel(job_id):
            raise HTTPException(409, "Job is not queued or running")
        return runner.get(job_id)

    @app.post("/api/jobs/{job_id}/force")
    def force(job_id: str):
        if not runner.force(job_id):
            raise HTTPException(409, "Job is not waiting")
        return runner.get(job_id)

    @app.get("/api/system")
    def system():
        data_dir = (root / ".tmp").resolve()
        return {"disk": [d for d in (_disk(data_dir), _disk("/")) if d],
                "busy_check": bool(runner.busy_check)}

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _safe_id(match_id):
    return bool(match_id) and all(c.isalnum() or c in "_-." for c in match_id) \
        and not match_id.startswith(".")
