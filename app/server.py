"""FastAPI app: JSON API over the pipeline + the static front end.

    GET  /api/stages                      stage catalog (forms are built from it)
    GET  /api/matches                     matches + not-yet-ingested videos
    GET  /api/matches/{match_id}          one match with per-stage status
    GET  /api/matches/{match_id}/files/{key}   download stats / report / tracking video
    GET  /api/jobs[?match_id=]            job history, newest first
    POST /api/jobs                        {stage, match_id?, params{}, force?}
    GET  /api/jobs/{id}?offset=N          job record + log text from byte N
    POST /api/jobs/{id}/cancel | /force
    GET  /api/matches/{match_id}/calibration            calibration clips + lineups + run state (issue #46)
    GET  /api/matches/{match_id}/calibration/{clip}     a clip's boxes per frame, with the tracker's guesses
    POST /api/matches/{match_id}/calibration/{clip}/names   name players: {items: [{frame, box, team, number}]}
    DELETE /api/matches/{match_id}/calibration/{clip}/names/{id}
    POST /api/matches/{match_id}/calibration-run        re-run identities and every stage after it, chained
    PUT  /api/matches/{match_id}/lines/{start}/export   keep/drop + trim one phase clip
    PUT  /api/matches/{match_id}/lines/export           keep/drop every phase
    POST /api/matches/{match_id}/lines/identify         anchor "this box is No. N" (issue #16)
    DELETE /api/matches/{match_id}/lines/identify/{id}  remove an anchor
    GET  /api/matches/{match_id}/clips?team=             detected clips by heading (issue #58)
    PUT  /api/matches/{match_id}/clips/{clip_id}/export  keep/drop + trim one clip
    PUT  /api/matches/{match_id}/clips/export            keep/drop a team's clips (optionally one heading / type)
    GET  /api/matches/{match_id}/clips/{clip_id}         players per frame + the clip's lines / highlights
    POST /api/matches/{match_id}/clips/{clip_id}/marks   tap a player: line / highlight / relink, or clear
    GET  /api/matches/{match_id}/clips/{clip_id}/video.mp4   small preview video (made on first use)
    GET  /api/matches/{match_id}/lines/others?frame=     other detected players at a frame (Identify mode)
    GET  /api/matches/{match_id}/players                every player's full-match stats (issue #16)
    GET  /api/matches/{match_id}/players/{pid}          one player's path, heatmap and stats (?start=&end= frames)
    GET  /api/library                                   teams, players, competitions, seasons, squads, matches
    POST/PUT/DELETE /api/library/{kind}[/{id}]          edit teams / players / competitions / seasons
    PUT  /api/library/squads/{season}/{team}            replace a squad (numbers per team and season)
    PUT  /api/library/matches/{match_id}                link a match to a season, teams and data
    POST /api/library/import/{match_id}                 fill the library from the match's config files
    GET  /api/matches/{match_id}/tracking               Tracking tab: players, problems, fixes
    GET  /api/matches/{match_id}/tracking/lanes?start=&end=&bins=   status lanes per player
    GET  /api/matches/{match_id}/tracking/frames?start=&end=        boxes + pitch spots per frame
    POST /api/matches/{match_id}/tracking/edits         save one hand fix (config/<id>_track_edits.json)
    DELETE /api/matches/{match_id}/tracking/edits/{id}  undo one fix
    POST /api/matches/{match_id}/tracking/autofix       Auto-fix: referee look-alikes off, gaps bridged, tracks named by elimination
    DELETE /api/matches/{match_id}/tracking/autofix     undo the auto-fix
    GET  /api/system                      disk space + whether heavy jobs are held
    GET  /api/matches/{match_id}/video    the source video (HTTP Range, for the Code tab)
    GET|PUT|DELETE /api/code/templates[/{name}]   code-window button layouts (issue #3)
    GET|POST /api/matches/{match_id}/instances     tagged instances (+ /instances.csv)
    PATCH|DELETE /api/matches/{match_id}/instances/{id}

Everything else under / is the single-page app in app/static/.
"""
import shutil
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import catalog, matches
from app.coding import CodeStore, CodingError
from app.jobs import JobRunner
from app.clips import Clips, ClipsError
from app.lines import FrameSource, LinesCache, LinesError
from app.players import PlayersCache, PlayersError
from app import library as libapi
from app.tracking import TrackingCache, TrackingError
from app.calibration import RUN_CHAIN, CalibrationCache, CalibrationError

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


class ExportEdit(BaseModel):
    team: str = "team_a"
    keep: bool | None = None
    trim_in: int | None = None
    trim_out: int | None = None
    reset_trim: bool = False


class Identify(BaseModel):
    team: str = "team_a"
    frame: int
    box: list
    number: int


class ExportAll(BaseModel):
    team: str = "team_a"
    keep: bool


class ClipMarks(BaseModel):
    action: str
    frame: int | None = None
    tid: int | None = None
    slots: list | None = None


class ClipsExportMany(BaseModel):
    team: str = "team_a"
    keep: bool
    group: str | None = None
    kind: str | None = None


class TrackEdit(BaseModel):
    edit: dict


class CalibName(BaseModel):
    items: list


class CalibRun(BaseModel):
    force: bool = False
    only_identities: bool = False  # "Update guesses": just re-track, so the other clips' guesses use the names so far
class InstanceNew(BaseModel):
    code: str
    color: str | None = None
    t: float
    lead_s: float = 5
    lag_s: float = 5
    labels: dict = Field(default_factory=dict)
    template: str | None = None


class InstanceEdit(BaseModel):
    labels: dict | None = None
    set_label: dict | None = None
    start: float | None = None
    end: float | None = None


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
    runner = runner or JobRunner(root, busy_check=busy_check, stage_lookup=catalog.get_stage)
    if getattr(runner, "stage_lookup", None) is None:
        runner.stage_lookup = catalog.get_stage
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
    def match_file(match_id: str, key: str, team: str = "team_a"):
        _match_or_404(match_id)
        if key not in matches.DOWNLOADABLE:
            raise HTTPException(404, "Not a downloadable output")
        if team not in ("team_a", "team_b"):
            raise HTTPException(400, "team must be team_a or team_b")
        path = matches.artifact_paths(root, match_id, team)[key]
        if not path.exists():
            raise HTTPException(404, "Not produced yet")
        return FileResponse(path, filename=path.name)

    lines_cache = LinesCache(root)
    tracking_cache = TrackingCache(root)
    frames = FrameSource()

    def _warm():
        """Load line-picker and Tracking-tab data for broadcast matches in the background, so the
        first visit to the Lines tab doesn't wait ~30 s reading a whole half."""
        for m in matches.list_matches(root):
            if m["pipeline"] == catalog.BROADCAST and m["artifacts"]["possession"]["exists"]:
                try:
                    lines_cache.get(m["match_id"], "team_a")
                except Exception:  # noqa: BLE001 - warming is best-effort
                    pass
            if m["pipeline"] == catalog.BROADCAST and m["artifacts"]["player_map"]["exists"]:
                try:
                    tracking_cache.get(m["match_id"])
                except Exception:  # noqa: BLE001
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

    def _lib(fn, *a):
        try:
            return fn(root, *a)
        except libapi.LibraryError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/library")
    def library_get():
        return libapi.summary(root, matches.list_matches(root))

    @app.post("/api/library/import/{match_id}")
    def library_import(match_id: str):
        _match_or_404(match_id)
        return _lib(libapi.import_match, match_id)

    @app.put("/api/library/squads/{season}/{team}")
    def library_squad(season: str, team: str, body: dict = Body(...)):
        return _lib(libapi.set_squad, season, team, body.get("entries") or [])

    @app.put("/api/library/matches/{match_id}")
    def library_match(match_id: str, body: dict = Body(...)):
        _match_or_404(match_id)
        return _lib(libapi.link_match, match_id, body)

    @app.post("/api/library/{kind}", status_code=201)
    def library_create(kind: str, body: dict = Body(...)):
        return _lib(libapi.upsert, kind, body, None)

    @app.put("/api/library/{kind}/{item_id}")
    def library_update(kind: str, item_id: str, body: dict = Body(...)):
        return _lib(libapi.upsert, kind, body, item_id)

    @app.delete("/api/library/{kind}/{item_id}")
    def library_delete(kind: str, item_id: str):
        return _lib(libapi.delete, kind, item_id)

    players_cache = PlayersCache(root)

    def _players(match_id):
        _match_or_404(match_id)
        try:
            return players_cache.get(match_id)
        except PlayersError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/api/matches/{match_id}/players")
    def players_summary(match_id: str):
        return _players(match_id).summary()

    @app.get("/api/matches/{match_id}/players/{pid}")
    def player_detail(match_id: str, pid: str, start: int | None = None, end: int | None = None):
        mp = _players(match_id)
        try:
            return mp.player(pid, start, end)
        except PlayersError as e:
            raise HTTPException(404 if "Unknown" in str(e) else 400, str(e)) from None

    @app.get("/api/matches/{match_id}/lines")
    def lines_summary(match_id: str, team: str = "team_a"):
        return _lines(match_id, team).summary()

    @app.post("/api/matches/{match_id}/lines/identify")
    def lines_identify(match_id: str, req: Identify):
        return _lines_call(_lines(match_id, req.team).identify, req.frame, req.box, req.number)

    @app.get("/api/matches/{match_id}/lines/others")
    def lines_others(match_id: str, frame: int, team: str = "team_a"):
        return _lines_call(_lines(match_id, team).others, frame)

    @app.delete("/api/matches/{match_id}/lines/identify/{anchor_id}")
    def lines_unidentify(match_id: str, anchor_id: str, team: str = "team_a"):
        return _lines_call(_lines(match_id, team).unidentify, anchor_id)

    @app.put("/api/matches/{match_id}/lines/export")
    def lines_export_all(match_id: str, req: ExportAll):
        return _lines_call(_lines(match_id, req.team).set_export_all, req.keep)

    def _clips_call(match_id, fn, *a):
        _match_or_404(match_id)
        try:
            return fn(Clips(root, match_id), *a)
        except ClipsError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/api/matches/{match_id}/clips")
    def clips_summary(match_id: str, team: str = "team_a"):
        return _clips_call(match_id, Clips.summary, team)

    @app.put("/api/matches/{match_id}/clips/export")
    def clips_export_many(match_id: str, req: ClipsExportMany):
        return _clips_call(match_id, Clips.set_export_many, req.team, req.keep, req.group, req.kind)

    @app.get("/api/matches/{match_id}/clips/{clip_id}")
    def clips_one(match_id: str, clip_id: str):
        return _clips_call(match_id, Clips.clip, clip_id)

    @app.post("/api/matches/{match_id}/clips/{clip_id}/marks")
    def clips_marks(match_id: str, clip_id: str, req: ClipMarks):
        return _clips_call(match_id, Clips.edit, clip_id, req.action, req.frame, req.tid, req.slots)

    @app.get("/api/matches/{match_id}/clips/{clip_id}/video.mp4")
    def clips_video(match_id: str, clip_id: str, prefetch: str = ""):
        path = _clips_call(match_id, Clips.preview, clip_id)
        nxt = [c for c in prefetch.split(",") if c][:2]
        if nxt:
            Clips(root, match_id).prefetch(nxt)
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})

    @app.put("/api/matches/{match_id}/clips/{clip_id}/export")
    def clips_export(match_id: str, clip_id: str, req: ExportEdit):
        return _clips_call(match_id, Clips.set_export, clip_id, req.keep, req.trim_in, req.trim_out, req.reset_trim)

    @app.put("/api/matches/{match_id}/lines/{start}/export")
    def lines_export(match_id: str, start: int, req: ExportEdit):
        return _lines_call(_lines(match_id, req.team).set_export, start, req.keep, req.trim_in, req.trim_out,
                           req.reset_trim)

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

    def _tracking(match_id):
        _match_or_404(match_id)
        try:
            return tracking_cache.get(match_id)
        except TrackingError as e:
            raise HTTPException(409, str(e)) from None

    def _tracking_call(fn, *a):
        try:
            return fn(*a)
        except TrackingError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/matches/{match_id}/tracking")
    def tracking_summary(match_id: str):
        out = _tracking(match_id).summary()
        out["run_jobs"] = _run_chain_jobs(match_id)
        return out

    @app.get("/api/matches/{match_id}/tracking/lanes")
    def tracking_lanes(match_id: str, start: int, end: int, bins: int = 600):
        return _tracking_call(_tracking(match_id).lanes, start, end, bins)

    @app.get("/api/matches/{match_id}/tracking/frames")
    def tracking_frames(match_id: str, start: int, end: int):
        return _tracking_call(_tracking(match_id).frames, start, end)

    @app.post("/api/matches/{match_id}/tracking/edits", status_code=201)
    def tracking_add(match_id: str, req: TrackEdit):
        return _tracking_call(_tracking(match_id).add_edit, req.edit)

    @app.post("/api/matches/{match_id}/tracking/autofix")
    def tracking_autofix(match_id: str):
        return _tracking_call(_tracking(match_id).autofix)

    @app.delete("/api/matches/{match_id}/tracking/autofix")
    def tracking_autofix_undo(match_id: str):
        return _tracking_call(_tracking(match_id).undo_autofix)

    @app.delete("/api/matches/{match_id}/tracking/edits/{edit_id}")
    def tracking_remove(match_id: str, edit_id: int):
        return _tracking_call(_tracking(match_id).remove_edit, edit_id)

    # ---- calibration clips (issue #46) ----
    calib_cache = CalibrationCache(root)

    def _calib(match_id):
        _match_or_404(match_id)
        try:
            return calib_cache.get(match_id)
        except CalibrationError as e:
            raise HTTPException(409, str(e)) from None

    def _calib_call(fn, *a):
        try:
            return fn(*a)
        except CalibrationError as e:
            raise HTTPException(400, str(e)) from None

    def _run_chain_jobs(match_id):
        """The latest run: the newest Track identities job and the jobs chained after it."""
        start = next((j for j in runner.list(match_id=match_id, limit=200) if j["stage"] == RUN_CHAIN[0]), None)
        chain = []
        while start is not None and len(chain) < len(RUN_CHAIN):
            chain.append(start)
            start = runner.get(start["next_job"]) if start.get("next_job") else None
        if chain and chain[-1].get("then") and chain[-1]["status"] in ("queued", "waiting", "running", "done"):
            chain += [{"title": catalog.get_stage(t["stage"]).title, "status": "waiting for the step before"}
                      for t in chain[-1]["then"]]
        return chain[::-1]

    @app.get("/api/matches/{match_id}/calibration")
    def calib_summary(match_id: str):
        out = _calib(match_id).summary()
        out["jobs"] = _run_chain_jobs(match_id)
        return out

    @app.get("/api/matches/{match_id}/calibration/{clip_id}")
    def calib_clip(match_id: str, clip_id: str):
        return _calib_call(_calib(match_id).clip, clip_id)

    @app.post("/api/matches/{match_id}/calibration/{clip_id}/names")
    def calib_name(match_id: str, clip_id: str, req: CalibName):
        return _calib_call(_calib(match_id).name, clip_id, req.items)

    @app.delete("/api/matches/{match_id}/calibration/{clip_id}/names/{anchor_id}")
    def calib_unname(match_id: str, clip_id: str, anchor_id: str):
        return _calib_call(_calib(match_id).unname, clip_id, anchor_id)

    @app.post("/api/matches/{match_id}/calibration-run", status_code=201)
    def calib_run(match_id: str, req: CalibRun):
        """Re-run identities with the clip names, then everything after it, in order."""
        m = _match_or_404(match_id)
        steps = []
        for sid in (RUN_CHAIN[:1] if req.only_identities else RUN_CHAIN):
            stage = catalog.get_stage(sid)
            args = catalog.build_args(stage, {}, match_id=match_id, videos_dir=root / "videos",
                                      meta_video=m.get("video"), config_dir=root / "config")
            steps.append({"stage": sid, "args": args, "obj": stage})
        first = steps[0]
        return runner.submit(first["obj"], first["args"], match_id=match_id, force=req.force,
                             then=[{"stage": s["stage"], "args": s["args"]} for s in steps[1:]])

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

    # ---------- code window (issue #3) ----------
    codes = CodeStore(root)

    def _code_call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except CodingError as e:
            raise HTTPException(400, str(e)) from None
        except KeyError:
            raise HTTPException(404, "Not found") from None

    def _clock(m):
        from _common import clock_label  # tools/ is on sys.path via app.lines
        cfg = matches._read_json(matches.artifact_paths(root, m["match_id"])["match_config"]) or {}
        fps = m["fps"] or 25.0
        return lambda t: clock_label(round(t * fps), cfg, fps)

    @app.get("/api/matches/{match_id}/video")
    def match_video(match_id: str):
        m = _match_or_404(match_id)
        path = Path(m["video"] or "")
        if not m["video"] or not path.is_file():
            raise HTTPException(404, "Video file not found")
        return FileResponse(path, media_type="video/mp4" if path.suffix.lower() in (".mp4", ".m4v") else None,
                            headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/code/templates")
    def code_templates():
        return codes.list_templates()

    @app.get("/api/code/templates/{name}")
    def code_template(name: str):
        return _code_call(codes.get_template, name)

    @app.put("/api/code/templates/{name}")
    def code_template_save(name: str, body: dict):
        if (body.get("name") or "").strip() != name:
            raise HTTPException(400, "Template name in the URL and the body differ")
        return _code_call(codes.save_template, body)

    @app.delete("/api/code/templates/{name}", status_code=204)
    def code_template_delete(name: str):
        _code_call(codes.delete_template, name)

    @app.get("/api/matches/{match_id}/instances")
    def instances(match_id: str):
        m = _match_or_404(match_id)
        d = codes.instances(match_id)
        clock = _clock(m)
        for i in d["instances"]:
            i["clock"] = clock(i["t"])
        d["match_labels"] = codes.match_labels(match_id, {"team_a": m["team_a_name"], "team_b": m["team_b_name"]})
        return d

    @app.get("/api/matches/{match_id}/instances.csv")
    def instances_csv(match_id: str):
        m = _match_or_404(match_id)
        return Response(codes.csv_text(match_id, _clock(m)), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{match_id}_instances.csv"'})

    @app.post("/api/matches/{match_id}/instances", status_code=201)
    def instance_add(match_id: str, req: InstanceNew):
        m = _match_or_404(match_id)
        inst = _code_call(codes.add, match_id, m["duration_s"], req.code, req.color, req.t, req.lead_s, req.lag_s,
                          req.labels, req.template)
        inst["clock"] = _clock(m)(inst["t"])
        return inst

    @app.patch("/api/matches/{match_id}/instances/{iid}")
    def instance_edit(match_id: str, iid: str, req: InstanceEdit):
        m = _match_or_404(match_id)
        inst = _code_call(codes.update, match_id, iid, m["duration_s"], req.labels, req.set_label, req.start, req.end)
        inst["clock"] = _clock(m)(inst["t"])
        return inst

    @app.delete("/api/matches/{match_id}/instances/{iid}", status_code=204)
    def instance_delete(match_id: str, iid: str):
        _match_or_404(match_id)
        _code_call(codes.delete, match_id, iid)

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

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        # the UI is plain ES modules with no build step or file hashes: make browsers
        # check for a newer copy (a cheap 304 when unchanged) instead of running stale code
        resp = await call_next(request)
        if not request.url.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-cache")
        return resp

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _safe_id(match_id):
    return bool(match_id) and all(c.isalnum() or c in "_-." for c in match_id) \
        and not match_id.startswith(".")
