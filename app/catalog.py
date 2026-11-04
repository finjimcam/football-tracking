"""Pipeline stages the web app can run, one per tools/ script.

Each Stage maps a whitelisted set of form params to CLI args for one tool, so
the browser never sends raw argv: only known flags, type-checked values, and
video paths that resolve inside videos/.

Two pipelines, picked per match by app.matches.match_pipeline():
  report     workflows/generate_match_report.md (fixed/panning camera, e.g.
             Veo; manual calibration + roster)
  broadcast  workflows/track_identities.md (broadcast/tactical camera with a
             hand-checked config/<id>_match.json: cached detection, automatic
             calibration, persistent identities, phase clips)
Some compute stages belong to both.

Interactive stages (calibrate_pitch.py, label_roster.py) are listed with
`interactive=True` and no runner: the app shows their status, and the user
still runs them by hand (see the workflow) until they get web front-ends.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
REPORT, BROADCAST = "report", "broadcast"
BOTH = (REPORT, BROADCAST)


class ParamError(ValueError):
    """A submitted value failed validation (shown to the user as-is)."""


@dataclass(frozen=True)
class Param:
    name: str            # form key, also the flag: name "max_gap_s" -> --max-gap-s
    kind: str            # "video" | "text" | "int" | "float" | "flag" | "choice"
    label: str
    help: str = ""
    required: bool = False
    default: object = None  # shown in the form only; the tool owns real defaults
    options: tuple = ()     # for "choice"

    @property
    def flag(self):
        return "--" + self.name.replace("_", "-")

    def to_dict(self):
        return {"name": self.name, "kind": self.kind, "label": self.label, "help": self.help,
                "required": self.required, "default": self.default, "options": list(self.options)}


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    tool: str | None          # tools/<tool>; None for interactive-only stages
    step: int                 # workflow step number, for ordering
    summary: str
    heavy: bool = False       # GPU pass: one at a time, niced, waits for the busy check
    cpu_heavy: bool = False   # long CPU/memory pass: same rules, its own one-at-a-time lane
    per_match: bool = True    # passes --match-id
    interactive: bool = False
    video_from_meta: bool = False  # tool requires --video: pass meta.json's path
    pipelines: tuple = (REPORT,)
    fixed_args: tuple = ()    # always passed (workflow-recommended settings)
    produces: tuple = ()      # artifact keys from app.matches.artifact_paths
    needs: tuple = ()         # artifact keys that must exist first
    params: tuple = field(default_factory=tuple)

    @property
    def lane(self):
        return "heavy" if self.heavy else "cpu" if self.cpu_heavy else "light"

    def to_dict(self):
        return {"id": self.id, "title": self.title, "tool": self.tool, "step": self.step,
                "summary": self.summary, "heavy": self.heavy, "cpu_heavy": self.cpu_heavy,
                "lane": self.lane, "per_match": self.per_match, "interactive": self.interactive,
                "pipelines": list(self.pipelines), "fixed_args": list(self.fixed_args),
                "produces": list(self.produces), "needs": list(self.needs),
                "params": [p.to_dict() for p in self.params]}


def _f(name, label, help="", default=None):
    return Param(name, "float", label, help, default=default)


TEAM = Param("team", "choice", "Team", "Whose phases", default="team_a", options=("team_a", "team_b"))

STAGES = (
    Stage("ingest", "Ingest video", "ingest_video.py", 1,
          "Probe the video, mint its match_id and write meta.json. A matching "
          "config/<id>_match.json is picked up automatically.",
          per_match=False, pipelines=BOTH, produces=("meta",),
          params=(Param("video", "video", "Video", "A file in videos/", required=True),
                  Param("team_a_name", "text", "Team A name"),
                  Param("team_b_name", "text", "Team B name"),
                  Param("kickoff", "text", "Kickoff", "e.g. 2026-07-05T15:00"))),

    # ---- report pipeline (workflows/generate_match_report.md) ----
    Stage("track", "Track players and ball", "run_tracking.py", 2,
          "YOLO + BoT-SORT over every frame. Slow: never re-run without a reason.",
          heavy=True, video_from_meta=True, produces=("tracks",), needs=("meta",),
          params=(_f("conf", "Player conf", default=0.1),
                  _f("ball_conf", "Ball conf"),
                  Param("imgsz", "int", "Inference size", default=1280))),
    Stage("stitch", "Stitch track fragments", "stitch_tracks.py", 3,
          "Merge fragmented player IDs. Re-run after calibrating.",
          produces=("tracks",), needs=("tracks",),
          params=(_f("max_gap_s", "Max gap (s)"),
                  Param("min_track_frames", "int", "Min track rows"))),
    Stage("teams", "Assign teams", "assign_teams.py", 4,
          "Cluster shirt colours into team_a / team_b.",
          produces=("team_map",), needs=("tracks",),
          params=(Param("samples_per_track", "int", "Samples per track"),
                  _f("outlier_scale", "Outlier scale"))),
    Stage("calibrate", "Calibrate pitch", "calibrate_pitch.py", 5,
          "Click pitch landmarks. Interactive: run it by hand for now.",
          interactive=True, produces=("calibration",), needs=("meta",)),
    Stage("camera_motion", "Estimate camera motion", "estimate_camera_motion.py", 6,
          "Moving camera only: per-frame homographies from keyframes.",
          heavy=True, produces=("calibration",), needs=("calibration",),
          params=(Param("no_mask_tracks", "flag", "Don't mask player boxes"),)),
    Stage("roster", "Label roster", "label_roster.py", 7,
          "Click team + shirt number on anchor frames. Interactive: run it by hand for now.",
          interactive=True, produces=("roster",), needs=("tracks",)),
    Stage("identities", "Assign identities", "assign_identities.py", 7,
          "Propagate roster labels across track fragments.",
          produces=("player_map",), needs=("tracks", "roster"),
          params=(_f("max_gap_s", "Max gap (s)"),
                  _f("pitch_margin_m", "Pitch margin (m)"))),

    # ---- broadcast pipeline (workflows/track_identities.md) ----
    Stage("b_detect", "Detect players and ball", "detect_frames.py", 1,
          "YOLO on every frame of the analysis window (cached, resumable). The one slow step.",
          heavy=True, pipelines=(BROADCAST,), produces=("detections",), needs=("meta",),
          params=(Param("limit", "int", "Only N frames", "quick test"),
                  Param("imgsz", "int", "Inference size", default=1280))),
    Stage("b_calibrate", "Calibrate pitch (automatic)", "calibrate_auto.py", 2,
          "Per-frame homographies from the seed clicks in the calibration file. "
          "Can run alongside detection: it waits for detection chunks.",
          cpu_heavy=True, pipelines=(BROADCAST,), produces=("homographies",),
          needs=("meta", "calibration"),
          params=(Param("check_frames", "text", "Check frames", "e.g. 5000,30000,60000"),)),
    Stage("b_identities", "Track identities", "track_identities.py", 3,
          "Persistent identities for all 22 players: tracks, team_map and player_map.",
          cpu_heavy=True, pipelines=(BROADCAST,), produces=("tracks", "player_map", "team_map"),
          needs=("detections", "homographies"),
          params=(Param("reuse_tracklets", "flag", "Reuse cached tracklets", "re-link only (~3 min)"),)),
    Stage("b_ball", "Track ball", "track_ball.py", 4,
          "Clean ball track; replaces the ball rows of tracks.parquet.",
          pipelines=(BROADCAST,), produces=("ball_track",), needs=("tracks",)),

    # ---- shared compute / outputs ----
    Stage("possession", "Compute possession", "compute_possession.py", 9,
          "Ball-to-player possession with hysteresis.",
          produces=("possession",), needs=("tracks", "team_map"),
          params=(_f("radius_m", "Radius (m)"), _f("switch_s", "Switch delay (s)"))),
    Stage("b_possession", "Compute possession", "compute_possession.py", 5,
          "Ball-to-player possession (incl. unidentified tracks). Phases come from this.",
          pipelines=(BROADCAST,), fixed_args=("--include-unidentified",),
          produces=("possession",), needs=("tracks", "team_map"),
          params=(_f("radius_m", "Radius (m)"), _f("switch_s", "Switch delay (s)"))),
    Stage("positions", "Compute positions", "compute_positions.py", 9,
          "Average positions and heatmap figures.",
          pipelines=BOTH, produces=("positions",), needs=("tracks", "team_map")),
    Stage("physical", "Compute physical", "compute_physical.py", 9,
          "Distance and speed per player / team.",
          produces=("physical",), needs=("tracks", "team_map"),
          params=(_f("smooth_s", "Smoothing (s)"), _f("max_speed", "Max speed (m/s)"))),
    Stage("b_physical", "Compute physical", "compute_physical.py", 6,
          "Distance and speed per player / team (1 s smoothing).",
          pipelines=(BROADCAST,), fixed_args=("--smooth-s", "1.0"),
          produces=("physical",), needs=("tracks", "team_map"),
          params=(_f("max_speed", "Max speed (m/s)"),)),
    Stage("b_verify", "Verify tracking", "verify_tracking.py", 7,
          "Self-checks: identity hand-offs, long gaps, ball. Contact sheets read the video.",
          cpu_heavy=True, pipelines=(BROADCAST,), produces=("verification",), needs=("tracks",),
          params=(Param("no_sheets", "flag", "Skip contact sheets"),)),
    Stage("b_compare", "Compare with published stats", "compare_reference_stats.py", 8,
          "outputs/stats/<match_id>_vs_reference.csv from config/<id>_reference.json.",
          pipelines=(BROADCAST,), produces=("vs_reference",), needs=("possession", "reference")),
    Stage("render", "Render tracking video", "render_tracking_video.py", 10,
          "Annotated video for checking identities.",
          heavy=True, pipelines=BOTH, produces=("tracking_video",), needs=("tracks",),
          params=(_f("start_s", "Start (s)"), _f("end_s", "End (s)"),
                  _f("scale", "Scale", "e.g. 0.5 for half size"),
                  _f("trail_s", "Trail (s)"),
                  Param("hide_unidentified", "flag", "Hide unidentified tracks"),
                  Param("minimap", "flag", "Pitch minimap", "broadcast: needs calibration"),
                  Param("possession", "flag", "Possession banner"))),
    Stage("b_phases", "Cut phase clips", "cut_phase_clips.py", 10,
          "One team's out-of-possession phases as a compilation + phase index CSV.",
          cpu_heavy=True, pipelines=(BROADCAST,), produces=("phase_index", "phase_video"),
          needs=("possession",),
          params=(TEAM, Param("shape", "flag", "Draw automatic lines (flickers)"))),
    Stage("b_shape_render", "Render picked lines", "cut_phase_clips.py", 11,
          "The phase compilation with the back / midfield lines you picked in the Lines tab.",
          cpu_heavy=True, pipelines=(BROADCAST,), fixed_args=("--shape-selection",),
          produces=("shape_video",), needs=("possession", "shape_selection"),
          params=(TEAM, Param("unselected", "choice", "Unpicked phases", default="none",
                              options=("none", "auto")))),
    Stage("b_clip_export", "Export phase clips (zip)", "cut_phase_clips.py", 12,
          "The phases you kept in the Lines tab, trimmed, with your picked lines: one MP4 per phase, "
          "zipped for PowerPoint.",
          cpu_heavy=True, pipelines=(BROADCAST,),
          fixed_args=("--shape-selection", "--export-selection", "--separate"),
          produces=("clips_zip",), needs=("possession",), params=(TEAM,)),
    Stage("export", "Export stats CSV", "export_stats.py", 12,
          "outputs/stats/<match_id>_stats.csv", pipelines=BOTH, produces=("stats",), needs=("meta",)),
    Stage("report", "Build PDF report", "build_report.py", 12,
          "outputs/reports/<match_id>_report.pdf", pipelines=BOTH, produces=("report",), needs=("meta",),
          params=(Param("team_a_name", "text", "Team A name"),
                  Param("team_b_name", "text", "Team B name"))),
)

BY_ID = {s.id: s for s in STAGES}


def stages_for(pipeline):
    """Per-match stages of one pipeline, in workflow order."""
    return sorted((s for s in STAGES if s.per_match and pipeline in s.pipelines),
                  key=lambda s: (s.step, STAGES.index(s)))


def get_stage(stage_id):
    try:
        return BY_ID[stage_id]
    except KeyError:
        raise ParamError(f"Unknown stage '{stage_id}'") from None


def safe_stem(stem):
    """Same sanitising as _common.compute_match_id, so a video's match config
    can be found before it's ingested."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


def find_match_config(video_path, config_dir):
    """config/<stem>_<sha8>_match.json for this video, if one exists. The sha
    is only checked by ingest itself; a stem match is enough to pick the file."""
    stem = safe_stem(Path(video_path).stem)
    hits = sorted(Path(config_dir).glob(f"{stem}_????????_match.json"))
    return hits[0] if len(hits) == 1 else None


def _video_arg(value, videos_dir):
    videos_dir = Path(videos_dir).resolve()
    path = (videos_dir / str(value)).resolve()
    if videos_dir not in path.parents:
        raise ParamError("Video must be a file inside videos/")
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
        raise ParamError(f"Not a video in videos/: {value}")
    return str(path)


def build_args(stage, values, match_id=None, videos_dir="videos", meta_video=None, config_dir=None):
    """Form values -> argv list (after `python tools/<tool>`). Unknown keys
    are rejected; blank optional values are left to the tool's defaults.
    `meta_video` is the match's video path from meta.json, for stages with
    video_from_meta (it's server-side data, not a form value)."""
    if stage.interactive or not stage.tool:
        raise ParamError(f"'{stage.title}' is interactive and can't run from the app yet")
    values = dict(values or {})
    known = {p.name for p in stage.params}
    extra = set(values) - known
    if extra:
        raise ParamError(f"Unknown parameter(s): {', '.join(sorted(extra))}")
    args = []
    if stage.per_match:
        if (not match_id or match_id.startswith(".")
                or not all(c.isalnum() or c in "_-." for c in match_id)):
            raise ParamError("A valid match_id is required")
        args += ["--match-id", match_id]
    if stage.video_from_meta:
        if not meta_video:
            raise ParamError("This match's meta.json has no video path: re-ingest it")
        args += ["--video", str(meta_video)]
    args += list(stage.fixed_args)
    config = {}
    video_path = None
    for p in stage.params:
        v = values.get(p.name)
        blank = v is None or (isinstance(v, str) and not v.strip())
        if blank or (p.kind == "flag" and not v):
            if p.required:
                raise ParamError(f"{p.label} is required")
            continue
        if p.kind == "flag":
            args.append(p.flag)
            continue
        if p.kind == "video":
            v = video_path = _video_arg(v, videos_dir)
        elif p.kind == "int":
            try:
                v = str(int(v))
            except (TypeError, ValueError):
                raise ParamError(f"{p.label} must be a whole number") from None
        elif p.kind == "float":
            try:
                v = repr(float(v))
            except (TypeError, ValueError):
                raise ParamError(f"{p.label} must be a number") from None
        elif p.kind == "choice":
            if v not in p.options:
                raise ParamError(f"{p.label} must be one of {', '.join(p.options)}")
        else:
            v = str(v).strip()
            if p.name == "check_frames" and not re.fullmatch(r"\d+(,\d+)*", v):
                raise ParamError("Check frames must be comma-separated frame numbers")
        # ingest takes team names/kickoff as one --config JSON, not flags
        if stage.id == "ingest" and p.name in ("team_a_name", "team_b_name", "kickoff"):
            config[p.name] = v
            continue
        args += [p.flag, v]
    if stage.id == "ingest":
        found = find_match_config(video_path, config_dir) if (video_path and config_dir) else None
        if found:
            # the hand-checked match config wins; form values only fill gaps
            base = json.loads(Path(found).read_text(encoding="utf-8"))
            config = {**config, **base}
        if config:
            args += ["--config", json.dumps(config)]
    return args
