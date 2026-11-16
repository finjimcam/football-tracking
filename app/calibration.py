"""Calibration clips (issue #46): the analyst names the players in three 5-second
clips per half; everything else about identities runs automatically.

The clips come from tools/calibration_clips.py. A name is an identity anchor
(tools/identity_anchors.py) in the same config/<id>_identity_anchors.json the
Lines tab's Identify mode writes, tagged with "clip": <clip id>, so
track_identities.py uses both. Each box shows the tracker's current guess;
"accept" turns the guesses on one frame into anchors in one go, so a clip is
"fix the wrong ones, accept the rest".
"""
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from _common import load_meta, load_player_map, match_config  # noqa: E402
from calibration_clips import PLAYER_KITS, clips_for_match  # noqa: E402
from identity_anchors import anchors_path, iou, load_anchors, save_anchors  # noqa: E402

from app.lines import ANCHOR_LOCK, team_lineup  # noqa: E402

TEAMS = ("team_a", "team_b")
KIT_TEAM = {"team_a": "team_a", "gk_a": "team_a", "team_b": "team_b", "gk_b": "team_b"}
# The jobs a "Run tracking" starts, in order; each starts when the one before succeeds.
RUN_CHAIN = ("b_identities", "b_ball", "b_possession", "b_physical", "positions", "b_verify")


class CalibrationError(ValueError):
    pass


class MatchCalibration:
    def __init__(self, root, match_id):
        self.root, self.match_id = Path(root), match_id
        self.tmp_dir, self.config_dir = self.root / ".tmp", self.root / "config"
        self.tracks_path = self.tmp_dir / match_id / "tracks.parquet"
        if not self.tracks_path.exists():
            raise CalibrationError("Needs tracks.parquet: run the pipeline down to 'Track identities' first")
        self.meta = load_meta(match_id, self.tmp_dir)
        self.fps = float(self.meta["fps"])
        self.cfg = match_config(match_id, self.tmp_dir, self.config_dir)
        self.anchor_path = anchors_path(match_id, self.config_dir)
        t = pd.read_parquet(self.tracks_path, columns=["frame", "class", "kit"])
        self.clips = clips_for_match(match_id, self.tmp_dir, self.config_dir, tracks=t)
        self.by_id = {c["id"]: c for c in self.clips}

    # ---- helpers -------------------------------------------------------
    def _clip(self, clip_id):
        c = self.by_id.get(clip_id)
        if c is None:
            raise CalibrationError(f"No calibration clip '{clip_id}'")
        return c

    def _rows(self, lo, hi):
        want = ["frame", "track_id", "class", "x1", "y1", "x2", "y2", "kit", "tracklet"]
        have = set(pq.ParquetFile(self.tracks_path).schema_arrow.names)
        rows = pd.read_parquet(self.tracks_path, filters=[("frame", ">=", int(lo)), ("frame", "<=", int(hi))],
                               columns=[c for c in want if c in have])
        for c in ("kit", "tracklet"):
            if c not in rows.columns:
                rows[c] = None if c == "kit" else -1
        rows = rows[rows["class"] == "player"]
        return rows[rows["kit"].isin(PLAYER_KITS) | (rows["track_id"] < 10000)]

    def _guesses(self):
        """{track_id: (team, number)} from the tracker's player_map."""
        pm = load_player_map(self.match_id, self.tmp_dir) or {"players": {}}
        out = {}
        for v in pm.get("players", {}).values():
            for t in v.get("track_ids", []):
                out[int(t)] = (v.get("team"), v.get("number"))
        return out

    def _team_of(self, tid, kit, guesses):
        g = guesses.get(int(tid))
        if g and g[0] in TEAMS:
            return g[0]
        return KIT_TEAM.get(kit)

    def _clip_anchors(self, clip):
        """Anchors that fall inside a clip (made here or anywhere else)."""
        return [a for a in load_anchors(self.anchor_path) if clip["start"] <= int(a["frame"]) <= clip["end"]]

    def _progress(self, clip, anchors=None):
        anchors = self._clip_anchors(clip) if anchors is None else anchors
        named = {(a["team"], int(a["number"])) for a in anchors if a.get("number") is not None}
        return {t: sorted(n for tm, n in named if tm == t) for t in TEAMS}

    def lineups(self):
        names = {"team_a": self.cfg.get("team_a_name", "Team A"), "team_b": self.cfg.get("team_b_name", "Team B")}
        colors = self.cfg.get("colors") or {}
        return {t: {"name": names[t], "color": colors.get(t) or ("#6CABDD" if t == "team_a" else "#F0F0F0"),
                    "players": team_lineup(self.config_dir, self.match_id, t)} for t in TEAMS}

    # ---- API -----------------------------------------------------------
    def summary(self):
        anchors = load_anchors(self.anchor_path)
        clips = []
        for c in self.clips:
            mine = [a for a in anchors if c["start"] <= int(a["frame"]) <= c["end"]]
            prog = self._progress(c, mine)
            clips.append({**c, "named": {t: len(v) for t, v in prog.items()}})
        stamp = self.tracks_path.stat().st_mtime
        return {"match_id": self.match_id, "fps": self.fps, "width": self.meta["width"],
                "height": self.meta["height"], "clips": clips, "lineups": self.lineups(),
                "anchors_total": len(anchors),
                "pending": sum(1 for a in anchors if a.get("created", 0) > stamp),
                "run_chain": list(RUN_CHAIN)}

    def clip(self, clip_id):
        """Every frame of the clip: [track_id, x1, y1, x2, y2, team, guess number],
        plus the anchors on it and which players are named so far."""
        c = self._clip(clip_id)
        guesses = self._guesses()
        frames = {}
        for r in self._rows(c["start"], c["end"]).itertuples():
            team = self._team_of(r.track_id, r.kit, guesses)
            g = guesses.get(int(r.track_id))
            frames.setdefault(int(r.frame), []).append(
                [int(r.track_id), round(r.x1, 1), round(r.y1, 1), round(r.x2, 1), round(r.y2, 1), team,
                 g[1] if g and g[0] == team else None])
        anchors = self._clip_anchors(c)
        return {**c, "frames": frames, "anchors": anchors, "named": self._progress(c, anchors)}

    def _make(self, clip, frame, box, team, number):
        person = next((p for p in team_lineup(self.config_dir, self.match_id, team) if p["number"] == int(number)), None)
        if person is None:
            raise CalibrationError(f"No. {number} isn't in the {team} lineup")
        rows = self._rows(frame, frame)
        best, score = None, 0.29
        for r in rows.itertuples():
            v = iou([r.x1, r.y1, r.x2, r.y2], box)
            if v > score:
                best, score = r, v
        if best is None:
            raise CalibrationError("No detected player there on this frame")
        return {"id": uuid.uuid4().hex[:10], "team": team, "number": person["number"], "name": person["name"],
                "frame": int(frame), "box": [round(float(v), 1) for v in (best.x1, best.y1, best.x2, best.y2)],
                "track_id": int(best.track_id),
                "tracklet": int(best.tracklet) if pd.notna(best.tracklet) and best.tracklet >= 0 else None,
                "clip": clip["id"], "source": "calibration", "created": time.time()}

    def name(self, clip_id, items):
        """Save names: items = [{frame, box, team, number}]. A player named twice
        in the same clip keeps the newest; a box named twice keeps the newest."""
        c = self._clip(clip_id)
        if not items:
            raise CalibrationError("Nothing to name")
        new = []
        for it in items:
            f = int(it["frame"])
            if not c["start"] <= f <= c["end"]:
                raise CalibrationError("That frame isn't in this clip")
            if it.get("team") not in TEAMS:
                raise CalibrationError("team must be team_a or team_b")
            box = [float(v) for v in it["box"]]
            new.append(self._make(c, f, box, it["team"], it["number"]))
        with ANCHOR_LOCK:
            anchors = load_anchors(self.anchor_path)
            for a in new:
                anchors = [b for b in anchors if not (
                    c["start"] <= int(b["frame"]) <= c["end"] and (
                        (b.get("team") == a["team"] and int(b.get("number", -1)) == a["number"])
                        or (int(b["frame"]) == a["frame"] and iou(b.get("box") or [0, 0, 0, 0], a["box"]) > 0.7)))]
                anchors.append(a)
            save_anchors(self.anchor_path, anchors)
        anchors = self._clip_anchors(c)
        return {"anchors": anchors, "named": self._progress(c, anchors)}

    def unname(self, clip_id, anchor_id):
        c = self._clip(clip_id)
        with ANCHOR_LOCK:
            anchors = load_anchors(self.anchor_path)
            keep = [a for a in anchors if a.get("id") != anchor_id]
            if len(keep) == len(anchors):
                raise CalibrationError("No such name")
            save_anchors(self.anchor_path, keep)
        anchors = self._clip_anchors(c)
        return {"anchors": anchors, "named": self._progress(c, anchors)}


class CalibrationCache:
    """One MatchCalibration per match, rebuilt when the tracks change."""

    def __init__(self, root):
        self.root = Path(root)
        self.items = {}

    def get(self, match_id):
        p = self.root / ".tmp" / match_id / "tracks.parquet"
        stamp = p.stat().st_mtime if p.exists() else 0
        hit = self.items.get(match_id)
        if hit and hit[0] == stamp:
            return hit[1]
        mc = MatchCalibration(self.root, match_id)
        self.items = {match_id: (stamp, mc)}
        return mc
