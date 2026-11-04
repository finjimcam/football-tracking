"""Players tab (issue #16): per-player full-match paths, heatmaps and stats.

Wraps tools/player_paths.py. A match's tracks are loaded once and kept until
tracks.parquet or player_map.json changes. That happens when "Apply to
tracking" in the Lines tab re-runs track_identities.py with the identity
anchors, so the Players tab always shows the corrected identities.
"""
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from _common import (clock_label, load_calibration, load_meta, load_player_map, match_config,  # noqa: E402
                     pitch_dims)
from player_paths import attacking_x, heatmap, path, player_index, player_rows, runs, stats  # noqa: E402


class PlayersError(ValueError):
    pass


class MatchPlayers:
    def __init__(self, root, match_id):
        self.root, self.match_id = Path(root), match_id
        tmp, cfgdir = self.root / ".tmp", self.root / "config"
        tp = tmp / match_id / "tracks.parquet"
        if not tp.exists():
            raise PlayersError("No tracks yet: run the tracking stages first.")
        self.meta = load_meta(match_id, tmp)
        self.cfg = match_config(match_id, tmp, cfgdir)
        self.fps = float(self.meta.get("fps") or 25.0)
        self.L, self.W = pitch_dims(load_calibration(match_id, cfgdir))
        pm = load_player_map(match_id, tmp)
        self.players = player_index(pm)
        if not self.players:
            raise PlayersError("No identified players yet: run \"Track identities\" first.")
        self.team_names = (pm or {}).get("team_names") or {}
        t = pd.read_parquet(tp, columns=["frame", "track_id", "class", "pitch_x", "pitch_y"])
        t = t[t["class"] == "player"].drop(columns="class")
        ids = {tid for p in self.players for tid in p["track_ids"]}
        self.tracks = t[t["track_id"].isin(ids)].reset_index(drop=True)
        win = self.cfg.get("analysis_window") or {}
        self.start = int(win.get("start_frame", self.tracks["frame"].min()))
        self.end = int(win.get("end_frame", self.tracks["frame"].max()))
        self.attacks = attacking_x(self.tracks, self.players, self.L)
        self.by_id = {p["id"]: p for p in self.players}
        # one row per frame per player, built once: every request just slices these
        self.rows = {p["id"]: player_rows(self.tracks, p["track_ids"]) for p in self.players}
        self.tracks = None
        self._summary = None

    def _window(self, start, end):
        s = self.start if start is None else max(self.start, int(start))
        e = self.end if end is None else min(self.end, int(end))
        if e <= s:
            raise PlayersError("Empty time window.")
        return s, e

    def _rows(self, pid, s=None, e=None):
        r = self.rows[pid]
        if s is None:
            return r
        f = r["frame"].to_numpy()
        return r.iloc[np.searchsorted(f, s, "left"):np.searchsorted(f, e, "right")]

    def summary(self):
        if self._summary is None:
            rows = []
            for p in self.players:
                r = runs(self._rows(p["id"]), self.fps)
                st = stats(r, self.fps, self.end - self.start + 1)
                rows.append({k: p[k] for k in ("id", "team", "number", "name", "role", "manual", "anchors")}
                            | {"distance_m": st["distance_m"], "coverage": st["coverage"],
                               "top_speed": st["top_speed"], "sprints": st["sprints"], "avg_pos": st["avg_pos"]})
            ko = (self.cfg.get("clock_sync") or {}).get("kickoff_frame")
            self._summary = {
                "match_id": self.match_id, "fps": self.fps, "pitch": [self.L, self.W],
                "start": self.start, "end": self.end, "kickoff_frame": ko,
                "start_clock": clock_label(self.start, self.cfg, self.fps),
                "end_clock": clock_label(self.end, self.cfg, self.fps),
                "teams": {t: self.team_names.get(t, t) for t in ("team_a", "team_b")},
                "attacks_x": self.attacks, "players": rows,
            }
        return self._summary

    def player(self, pid, start=None, end=None, cell=2.5, hz=2.0):
        p = self.by_id.get(pid)
        if p is None:
            raise PlayersError(f"Unknown player {pid}")
        s, e = self._window(start, end)
        r = runs(self._rows(pid, s, e), self.fps)
        team = []
        for q in self.players:  # team shape for the same window: everyone's average position
            if q["team"] != p["team"]:
                continue
            g = self._rows(q["id"], s, e)
            if len(g):
                team.append({"id": q["id"], "number": q["number"],
                             "pos": [round(float(g["x"].mean()), 1), round(float(g["y"].mean()), 1)]})
        return {
            "player": {k: v for k, v in p.items() if k != "track_ids"} | {"tracks": len(p["track_ids"])},
            "start": s, "end": e,
            "start_clock": clock_label(s, self.cfg, self.fps), "end_clock": clock_label(e, self.cfg, self.fps),
            "stats": stats(r, self.fps, e - s + 1),
            "heatmap": heatmap(r, self.fps, self.L, self.W, cell),
            "path": path(r, self.fps, hz),
            "team_shape": team,
            "attacks_x": self.attacks[p["team"]],
        }


class PlayersCache:
    """MatchPlayers per match, rebuilt when the tracking artifacts change."""

    def __init__(self, root, size=2):
        self.root, self.size = Path(root), size
        self.items = OrderedDict()
        self.lock = threading.Lock()

    def _stamp(self, match_id):
        paths = [self.root / ".tmp" / match_id / n for n in ("tracks.parquet", "player_map.json")]
        paths.append(self.root / "config" / f"{match_id}_match.json")
        return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)

    def get(self, match_id):
        stamp = self._stamp(match_id)
        with self.lock:
            hit = self.items.get(match_id)
            if hit and hit[0] == stamp:
                self.items.move_to_end(match_id)
                return hit[1]
        mp = MatchPlayers(self.root, match_id)  # reads tracks: outside the lock
        with self.lock:
            self.items[match_id] = (stamp, mp)
            self.items.move_to_end(match_id)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return mp
