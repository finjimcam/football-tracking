"""Match discovery and per-stage status, read straight from the artifact
layout owned by tools/_common.py (no database yet, that's issue #18).

A match exists once ingest_video.py has written .tmp/<match_id>/meta.json.
Videos in videos/ that no meta.json points at are listed as "not ingested".
A match uses the broadcast pipeline when config/<id>_match.json exists
(workflows/track_identities.md), else the report pipeline.
"""
import json
from pathlib import Path

from app.catalog import BROADCAST, REPORT, VIDEO_EXTS, stages_for

PHASE = "out_of_possession"
STALE_SLACK_S = 120


def artifact_paths(root, match_id, team="team_a"):
    root = Path(root)
    tmp = root / ".tmp" / match_id
    cfg = root / "config"
    out = root / "outputs"
    return {
        "meta": tmp / "meta.json",
        "match_config": cfg / f"{match_id}_match.json",
        "reference": cfg / f"{match_id}_reference.json",
        "detections": tmp / "detections.parquet",
        "homographies": tmp / "frame_homographies.npz",
        "tracks": tmp / "tracks.parquet",
        "team_map": tmp / "team_map.json",
        "calibration": cfg / f"{match_id}_calibration.json",
        "roster": cfg / f"{match_id}_roster.json",
        "player_map": tmp / "player_map.json",
        "ball_track": tmp / "ball_track.parquet",
        "possession": tmp / "possession.json",
        "positions": tmp / "positions.json",
        "physical": tmp / "physical.json",
        "verification": tmp / "verification.json",
        "vs_reference": out / "stats" / f"{match_id}_vs_reference.csv",
        "phase_index": out / "stats" / f"{match_id}_{team}_{PHASE}_phases.csv",
        "phase_video": out / "videos" / f"{match_id}_{team}_{PHASE}.mp4",
        "shape_selection": cfg / f"{match_id}_{team}_{PHASE}_shape_selection.json",
        "shape_video": out / "videos" / f"{match_id}_{team}_{PHASE}_shape_selected.mp4",
        "stats": out / "stats" / f"{match_id}_stats.csv",
        "report": out / "reports" / f"{match_id}_report.pdf",
        "tracking_video": out / "videos" / f"{match_id}_tracking.mp4",
    }


# Outputs the browser may download, by artifact key (served read-only).
DOWNLOADABLE = ("stats", "report", "tracking_video", "vs_reference", "phase_index",
                "phase_video", "shape_video")

# Artifacts that are inputs, never "produced" by a stage: committed configs
# can be older than everything else without making stages stale.
INPUTS = {"match_config", "reference", "calibration", "roster", "shape_selection"}


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _artifacts(root, match_id):
    paths = artifact_paths(root, match_id)
    out = {}
    for key, path in paths.items():
        if path.exists():
            st = path.stat()
            out[key] = {"exists": True, "mtime": st.st_mtime, "size": st.st_size}
        else:
            out[key] = {"exists": False}
    calib = out["calibration"]
    if calib["exists"]:
        data = _read_json(paths["calibration"]) or {}
        calib["mode"] = data.get("mode", "static")
        calib["keyframes"] = len(data.get("keyframes") or [])
    return out


def match_pipeline(artifacts):
    return BROADCAST if artifacts["match_config"]["exists"] else REPORT


def _stage_status(stage, artifacts):
    """done / ready / blocked / stale / skip, from artifact presence only. A
    stage whose output is older than one of its (produced) inputs is `stale`
    (re-run suggested)."""
    missing = [k for k in stage.needs if not artifacts[k]["exists"]]
    if stage.id == "camera_motion":
        # only meaningful for keyframe calibrations; "dynamic" means done
        mode = artifacts["calibration"].get("mode")
        if missing:
            return {"state": "blocked", "missing": missing}
        if mode == "dynamic":
            return {"state": "done"}
        if mode == "keyframes":
            return {"state": "ready"}
        return {"state": "skip", "note": "Fixed camera: not needed"}
    if stage.id == "b_calibrate" and not missing and not artifacts["calibration"].get("keyframes"):
        return {"state": "blocked", "missing": ["calibration seed clicks"]}
    if missing:
        return {"state": "blocked", "missing": missing}
    produced = [artifacts[k] for k in stage.produces if k not in stage.needs]
    if stage.produces and produced and all(a["exists"] for a in produced):
        ins = [artifacts[k]["mtime"] for k in stage.needs if k not in INPUTS]
        # slack: track_ball.py rewrites tracks.parquet just after writing its own output
        if ins and min(a["mtime"] for a in produced) < max(ins) - STALE_SLACK_S:
            return {"state": "stale"}
        return {"state": "done"}
    return {"state": "ready"}


def match_summary(root, match_id):
    paths = artifact_paths(root, match_id)
    meta = _read_json(paths["meta"])
    if meta is None:
        return None
    arts = _artifacts(root, match_id)
    # the live hand-checked match config wins over the copy frozen into meta.json
    cfg = _read_json(paths["match_config"]) or meta.get("config") or {}
    pipeline = match_pipeline(arts)
    stages = stages_for(pipeline)
    return {
        "match_id": match_id,
        "pipeline": pipeline,
        "video": meta.get("video"),
        "video_name": Path(meta.get("video") or "").name,
        "team_a_name": cfg.get("team_a_name") or "Team A",
        "team_b_name": cfg.get("team_b_name") or "Team B",
        "kickoff": cfg.get("kickoff"),
        "fps": meta.get("fps"),
        "duration_s": meta.get("duration_s"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "frame_count": meta.get("frame_count"),
        "ingested_at": arts["meta"]["mtime"],
        "artifacts": arts,
        "stage_order": [s.id for s in stages],
        "stages": {s.id: _stage_status(s, arts) for s in stages},
    }


def list_matches(root):
    tmp = Path(root) / ".tmp"
    out = []
    if tmp.is_dir():
        for d in sorted(tmp.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                m = match_summary(root, d.name)
                if m:
                    out.append(m)
    out.sort(key=lambda m: m["ingested_at"], reverse=True)
    return out


def list_videos(root, matches=None):
    """Videos in videos/ with the match they were ingested as (or None)."""
    from app.catalog import find_match_config

    vdir = Path(root) / "videos"
    if matches is None:
        matches = list_matches(root)
    by_path = {}
    for m in matches:
        if m["video"]:
            by_path[str(Path(m["video"]).resolve())] = m["match_id"]
    out = []
    if vdir.is_dir():
        for p in sorted(vdir.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                cfg = find_match_config(p, Path(root) / "config")
                out.append({"name": p.name, "size": p.stat().st_size,
                            "match_id": by_path.get(str(p.resolve())),
                            "match_config": cfg.name if cfg else None})
    return out
