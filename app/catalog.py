"""Pipeline stages the web app can run, one per tools/ script.

Each Stage maps a whitelisted set of form params to CLI args for one tool, so
the browser never sends raw argv: only known flags, type-checked values, and
video paths that resolve inside videos/. Order and wording follow
workflows/generate_match_report.md (Steps 1-11).

Interactive stages (calibrate_pitch.py, label_roster.py) are listed with
`interactive=True` and no runner: the app shows their status, and the user
still runs them by hand (see the workflow) until they get web front-ends.
"""
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


class ParamError(ValueError):
    """A submitted value failed validation (shown to the user as-is)."""


@dataclass(frozen=True)
class Param:
    name: str            # form key, also the flag: name "max_gap_s" -> --max-gap-s
    kind: str            # "video" | "text" | "int" | "float" | "flag"
    label: str
    help: str = ""
    required: bool = False
    default: object = None  # shown in the form only; the tool owns real defaults

    @property
    def flag(self):
        return "--" + self.name.replace("_", "-")

    def to_dict(self):
        return {"name": self.name, "kind": self.kind, "label": self.label, "help": self.help,
                "required": self.required, "default": self.default}


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    tool: str | None          # tools/<tool>; None for interactive-only stages
    step: int                 # workflow step number, for ordering
    summary: str
    heavy: bool = False       # GPU / full-video pass: one at a time, niced
    per_match: bool = True    # passes --match-id
    interactive: bool = False
    video_from_meta: bool = False  # tool requires --video: pass meta.json's path
    produces: tuple = ()      # artifact keys from app.matches.ARTIFACTS
    needs: tuple = ()         # artifact keys that must exist first
    params: tuple = field(default_factory=tuple)

    def to_dict(self):
        return {"id": self.id, "title": self.title, "tool": self.tool, "step": self.step,
                "summary": self.summary, "heavy": self.heavy, "per_match": self.per_match,
                "interactive": self.interactive, "produces": list(self.produces),
                "needs": list(self.needs), "params": [p.to_dict() for p in self.params]}


def _f(name, label, help="", default=None):
    return Param(name, "float", label, help, default=default)


STAGES = (
    Stage("ingest", "Ingest video", "ingest_video.py", 1,
          "Probe the video, mint its match_id and write meta.json.",
          per_match=False, produces=("meta",),
          params=(Param("video", "video", "Video", "A file in videos/", required=True),
                  Param("team_a_name", "text", "Team A name"),
                  Param("team_b_name", "text", "Team B name"),
                  Param("kickoff", "text", "Kickoff", "e.g. 2026-07-05T15:00"))),
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
    Stage("render", "Render tracking video", "render_tracking_video.py", 8,
          "Annotated video for checking identities.",
          heavy=True, produces=("tracking_video",), needs=("tracks",),
          params=(_f("start_s", "Start (s)"), _f("end_s", "End (s)"),
                  _f("scale", "Scale", "e.g. 0.5 for half size"),
                  _f("trail_s", "Trail (s)"),
                  Param("hide_unidentified", "flag", "Hide unidentified tracks"))),
    Stage("possession", "Compute possession", "compute_possession.py", 9,
          "Ball-to-player possession with hysteresis.",
          produces=("possession",), needs=("tracks", "team_map"),
          params=(_f("radius_m", "Radius (m)"), _f("switch_s", "Switch delay (s)"))),
    Stage("positions", "Compute positions", "compute_positions.py", 9,
          "Average positions and heatmap figures.",
          produces=("positions",), needs=("tracks", "team_map")),
    Stage("physical", "Compute physical", "compute_physical.py", 9,
          "Distance and speed per player / team.",
          produces=("physical",), needs=("tracks", "team_map"),
          params=(_f("smooth_s", "Smoothing (s)"), _f("max_speed", "Max speed (m/s)"))),
    Stage("export", "Export stats CSV", "export_stats.py", 10,
          "outputs/stats/<match_id>_stats.csv", produces=("stats",), needs=("meta",)),
    Stage("report", "Build PDF report", "build_report.py", 11,
          "outputs/reports/<match_id>_report.pdf", produces=("report",), needs=("meta",),
          params=(Param("team_a_name", "text", "Team A name"),
                  Param("team_b_name", "text", "Team B name"))),
)

BY_ID = {s.id: s for s in STAGES}


def get_stage(stage_id):
    try:
        return BY_ID[stage_id]
    except KeyError:
        raise ParamError(f"Unknown stage '{stage_id}'") from None


def _video_arg(value, videos_dir):
    videos_dir = Path(videos_dir).resolve()
    path = (videos_dir / str(value)).resolve()
    if videos_dir not in path.parents:
        raise ParamError("Video must be a file inside videos/")
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
        raise ParamError(f"Not a video in videos/: {value}")
    return str(path)


def build_args(stage, values, match_id=None, videos_dir="videos", meta_video=None):
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
    config = {}
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
            v = _video_arg(v, videos_dir)
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
        else:
            v = str(v).strip()
        # ingest takes team names/kickoff as one --config JSON, not flags
        if stage.id == "ingest" and p.name in ("team_a_name", "team_b_name", "kickoff"):
            config[p.name] = v
            continue
        args += [p.flag, v]
    if config:
        import json
        args += ["--config", json.dumps(config)]
    return args
