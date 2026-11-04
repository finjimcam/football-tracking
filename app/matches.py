"""Match discovery and per-stage status, read straight from the artifact
layout owned by tools/_common.py (no database yet, that's issue #18).

A match exists once ingest_video.py has written .tmp/<match_id>/meta.json.
Videos in videos/ that no meta.json points at are listed as "not ingested".
"""
import json
from pathlib import Path

from app.catalog import STAGES, VIDEO_EXTS


def artifact_paths(root, match_id):
    root = Path(root)
    tmp = root / ".tmp" / match_id
    return {
        "meta": tmp / "meta.json",
        "tracks": tmp / "tracks.parquet",
        "team_map": tmp / "team_map.json",
        "calibration": root / "config" / f"{match_id}_calibration.json",
        "roster": root / "config" / f"{match_id}_roster.json",
        "player_map": tmp / "player_map.json",
        "possession": tmp / "possession.json",
        "positions": tmp / "positions.json",
        "physical": tmp / "physical.json",
        "stats": root / "outputs" / "stats" / f"{match_id}_stats.csv",
        "report": root / "outputs" / "reports" / f"{match_id}_report.pdf",
        "tracking_video": root / "outputs" / "videos" / f"{match_id}_tracking.mp4",
    }


# Outputs the browser may download, by artifact key (served read-only).
DOWNLOADABLE = ("stats", "report", "tracking_video")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _artifacts(root, match_id):
    out = {}
    for key, path in artifact_paths(root, match_id).items():
        if path.exists():
            st = path.stat()
            out[key] = {"exists": True, "mtime": st.st_mtime, "size": st.st_size}
        else:
            out[key] = {"exists": False}
    calib = out["calibration"]
    if calib["exists"]:
        data = _read_json(artifact_paths(root, match_id)["calibration"]) or {}
        calib["mode"] = data.get("mode", "static")
    return out


def _stage_status(stage, artifacts):
    """done / ready / blocked, from artifact presence only. A stage whose
    output is older than one of its inputs is `stale` (re-run suggested)."""
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
    if missing:
        return {"state": "blocked", "missing": missing}
    produced = [artifacts[k] for k in stage.produces if k not in stage.needs]
    if stage.produces and all(a["exists"] for a in produced) and produced:
        newest_in = max((artifacts[k]["mtime"] for k in stage.needs), default=0)
        oldest_out = min(a["mtime"] for a in produced)
        return {"state": "stale" if oldest_out < newest_in else "done"}
    return {"state": "ready"}


def match_summary(root, match_id):
    paths = artifact_paths(root, match_id)
    meta = _read_json(paths["meta"])
    if meta is None:
        return None
    arts = _artifacts(root, match_id)
    cfg = meta.get("config") or {}
    return {
        "match_id": match_id,
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
        "stages": {s.id: _stage_status(s, arts) for s in STAGES if s.per_match},
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
                out.append({"name": p.name, "size": p.stat().st_size,
                            "match_id": by_path.get(str(p.resolve()))})
    return out
