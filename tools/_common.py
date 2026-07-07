"""Shared helpers for the match-report pipeline (workflows/generate_match_report.md).

Not a CLI tool. Owns the artifact layout so every stage agrees on paths:

    .tmp/<match_id>/meta.json            (ingest_video.py)
    .tmp/<match_id>/tracks.parquet       (run_tracking.py)
    .tmp/<match_id>/team_map.json        (assign_teams.py)
    .tmp/<match_id>/player_map.json      (assign_identities.py)
    .tmp/<match_id>/possession.json      (compute_possession.py)
    .tmp/<match_id>/positions.json       (compute_positions.py)
    .tmp/<match_id>/physical.json        (compute_physical.py)
    .tmp/<match_id>/figures/*.png        (compute_positions.py)
    config/<match_id>_calibration.json   (calibrate_pitch.py)
    config/<match_id>_roster.json        (label_roster.py)
    outputs/stats/<match_id>_stats.csv   (export_stats.py)
    outputs/reports/<match_id>_report.pdf (build_report.py)
    outputs/videos/<match_id>_tracking.mp4 (render_tracking_video.py)
"""
import hashlib
import json
from pathlib import Path

import numpy as np

PITCH_LENGTH = 105.0  # meters, FIFA standard default
PITCH_WIDTH = 68.0

TMP_ROOT = Path(".tmp")
CONFIG_DIR = Path("config")
OUTPUTS_DIR = Path("outputs")


def compute_match_id(video_path):
    """Video file stem + first 8 chars of the file's SHA-1 (per the workflow,
    so the same file always maps to the same ID)."""
    video_path = Path(video_path)
    sha1 = hashlib.sha1()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha1.update(chunk)
    return f"{video_path.stem}_{sha1.hexdigest()[:8]}"


def match_dir(match_id, tmp_root=TMP_ROOT, create=False):
    d = Path(tmp_root) / match_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def load_meta(match_id, tmp_root=TMP_ROOT):
    path = match_dir(match_id, tmp_root) / "meta.json"
    if not path.exists():
        raise SystemExit(f"meta.json not found at {path} — run tools/ingest_video.py first")
    return read_json(path)


def load_tracks(match_id, tmp_root=TMP_ROOT):
    import pandas as pd

    path = match_dir(match_id, tmp_root) / "tracks.parquet"
    if not path.exists():
        raise SystemExit(f"tracks.parquet not found at {path} — run tools/run_tracking.py first")
    return pd.read_parquet(path)


def load_team_map(match_id, tmp_root=TMP_ROOT):
    path = match_dir(match_id, tmp_root) / "team_map.json"
    if not path.exists():
        raise SystemExit(f"team_map.json not found at {path} — run tools/assign_teams.py first")
    raw = read_json(path)
    return {int(k): v for k, v in raw.items()}


def calibration_path(match_id, config_dir=CONFIG_DIR):
    return Path(config_dir) / f"{match_id}_calibration.json"


def roster_path(match_id, config_dir=CONFIG_DIR):
    return Path(config_dir) / f"{match_id}_roster.json"


def load_roster(match_id, config_dir=CONFIG_DIR, path=None):
    """Roster dict from label_roster.py, or None if none exists yet."""
    path = Path(path) if path else roster_path(match_id, config_dir)
    if not path.exists():
        return None
    return read_json(path)


def load_player_map(match_id, tmp_root=TMP_ROOT, path=None):
    """player_map.json from assign_identities.py, or None if none exists.
    Returns the dict with `track_to_player` keys coerced to int."""
    path = Path(path) if path else match_dir(match_id, tmp_root) / "player_map.json"
    if not path.exists():
        return None
    pm = read_json(path)
    pm["track_to_player"] = {int(k): v for k, v in pm.get("track_to_player", {}).items()}
    return pm


def ignored_track_ids(pm):
    """Track_ids curated out by the user / pitch-boundary filter (empty set if
    no player_map yet). These must be excluded from ALL stats, not just left
    unidentified — see workflows/curate_and_identify_players.md."""
    if not pm:
        return set()
    return {int(i["track_id"]) for i in pm.get("ignored_tracks", [])}


def resolve_track_teams(track_ids, team_map, pm):
    """track_id -> team in ROSTER letter space ('team_a'/'team_b'/'unknown'/
    'ignored'). Callers drop 'ignored' rows before computing anything.

    Without a player_map this is just team_map (v1 fallback, unchanged).
    With one: an identified track takes its player's roster team; the rest
    take team_map, flipped when assign_identities recorded that the jersey
    clusters were named opposite to the user's a/b labels — so team totals
    and per-player rows land in the same team."""
    if not pm:
        return {int(t): team_map.get(int(t), "unknown") for t in track_ids}
    ignored = ignored_track_ids(pm)
    t2p = pm.get("track_to_player", {})
    player_team = {k: p["team"] for k, p in pm.get("players", {}).items()}
    flip = pm.get("team_map_alignment") == "flipped"
    swap = {"team_a": "team_b", "team_b": "team_a"}
    out = {}
    for t in track_ids:
        t = int(t)
        if t in ignored:
            out[t] = "ignored"
        elif t in t2p:
            out[t] = player_team.get(t2p[t], "unknown")
        else:
            team = team_map.get(t, "unknown")
            out[t] = swap.get(team, team) if flip else team
    return out


def resolve_click_to_track(frame_rows, xy, max_px=60.0):
    """Map an image click to a player box on one frame of tracks data.

    frame_rows: DataFrame slice (one frame, player class) with x1..y2 and
    foot_x/foot_y. Prefers the smallest box containing the click (players
    overlap; the smaller box is the nearer/more specific one), else the
    nearest foot point within max_px. Returns the row (pandas Series) or None.
    """
    x, y = float(xy[0]), float(xy[1])
    inside = frame_rows[(frame_rows["x1"] <= x) & (frame_rows["x2"] >= x)
                        & (frame_rows["y1"] <= y) & (frame_rows["y2"] >= y)]
    if len(inside):
        area = (inside["x2"] - inside["x1"]) * (inside["y2"] - inside["y1"])
        return inside.loc[area.idxmin()]
    if frame_rows.empty:
        return None
    d = np.hypot(frame_rows["foot_x"] - x, frame_rows["foot_y"] - y)
    if float(d.min()) > max_px:
        return None
    return frame_rows.loc[d.idxmin()]


def load_calibration(match_id, config_dir=CONFIG_DIR, path=None):
    """Return the calibration dict, or None if no calibration file exists.
    `path` overrides the default location (for reusing another match's
    calibration when the camera hasn't moved).

    Modes:
    - "static": one homography for the whole video ("homography" key).
    - "keyframes": manual keyframes only — not usable yet; the user must run
      tools/estimate_camera_motion.py to produce per-frame homographies.
    - "dynamic": per-frame homographies loaded from the npz referenced by
      "frame_homographies" (arrays land in "frame_index"/"frame_H")."""
    path = Path(path) if path else calibration_path(match_id, config_dir)
    if not path.exists():
        return None
    calib = read_json(path)
    mode = calib.get("mode", "static")
    if mode == "static":
        calib["homography"] = np.array(calib["homography"], dtype=np.float64)
        return calib
    if mode == "keyframes":
        raise SystemExit(f"{path} has keyframes but no per-frame homographies yet — "
                         "run tools/estimate_camera_motion.py first")
    if mode == "dynamic":
        npz_path = Path(calib["frame_homographies"])
        if not npz_path.exists():
            raise SystemExit(f"{npz_path} (referenced by {path}) not found — "
                             ".tmp may have been cleared; re-run tools/estimate_camera_motion.py")
        data = np.load(npz_path)
        calib["frame_index"] = data["frames"]
        calib["frame_H"] = data["H"]
        return calib
    raise SystemExit(f"{path}: unknown calibration mode '{mode}'")


def pitch_dims(calib):
    """(length, width) in meters — the calibration's own dims when it has
    them (calibrate_pitch.py stores the user-entered size), else FIFA default."""
    if calib is None:
        return PITCH_LENGTH, PITCH_WIDTH
    return calib.get("pitch_length", PITCH_LENGTH), calib.get("pitch_width", PITCH_WIDTH)


def project_points(homography, xy):
    """Project an (N, 2) array of image pixels to pitch meters via the
    calibration homography."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(xy), 1))
    h = (np.hstack([xy, ones]) @ np.asarray(homography).T)
    return h[:, :2] / h[:, 2:3]


def apply_calibration(df, calib, cols=("foot_x", "foot_y"), out_cols=("pitch_x", "pitch_y")):
    """Add projected pitch-coordinate columns to a tracks DataFrame.
    Returns (df, unit) where unit is 'm' if calibrated else 'px' (in which
    case out_cols are just copies of the pixel columns).

    Static calibrations use one homography for every row; dynamic ones
    (moving camera) look up each row's frame in the per-frame homography
    table, snapping to the nearest available frame."""
    df = df.copy()
    if calib is None:
        df[out_cols[0]] = df[cols[0]]
        df[out_cols[1]] = df[cols[1]]
        return df, "px"
    xy = df[list(cols)].to_numpy(dtype=np.float64)
    if "frame_H" in calib:
        frame_index, frame_H = calib["frame_index"], calib["frame_H"]
        pos = np.searchsorted(frame_index, df["frame"].to_numpy())
        pos = np.clip(pos, 0, len(frame_index) - 1)
        proj = np.empty_like(xy)
        for p in np.unique(pos):
            sel = pos == p
            proj[sel] = project_points(frame_H[p], xy[sel])
    else:
        proj = project_points(calib["homography"], xy)
    df[out_cols[0]] = proj[:, 0]
    df[out_cols[1]] = proj[:, 1]
    return df, "m"
