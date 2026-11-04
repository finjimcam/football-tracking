"""Web line picker (issue #32): the browser version of tools/select_shape_players.py.

The analyst picks a team's back-line and midfield-line players once per phase.
Each pick is a chain: a track id, handed on when the tracker fragments, with
gaps filled in pitch space. All of that logic stays in tools/cut_phase_clips.py
and tools/select_shape_players.py, and this module only wraps it. That keeps
the browser, the CLI picker and `cut_phase_clips.py --shape-selection` drawing
exactly the same chains.

Picks are written to the same
config/<id>_<team>_<phase>_shape_selection.json the CLI picker uses, so either
tool can resume from the other's work.
"""
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import pandas as pd

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from _common import clock_label, load_calibration, load_meta, match_config  # noqa: E402
from cut_phase_clips import (build_chains, entry_slots, homography_lookup, index_rows,  # noqa: E402
                             load_selection, load_team_rows, phases, save_selection, selection_path)
from select_shape_players import auto_slots, pick_keyframe, relink_target  # noqa: E402

PHASE = "out_of_possession"
TEAMS = ("team_a", "team_b")
# Same defaults as cut_phase_clips.py / select_shape_players.py: phases are
# keyed by start frame, so these must match or saved picks won't line up.
MERGE_S, MIN_S, PRE_S, POST_S = 2.0, 3.0, 1.5, 1.0


class LinesError(ValueError):
    pass


class MatchLines:
    """Everything the picker needs for one (match, team), loaded once."""

    def __init__(self, root, match_id, team):
        if team not in TEAMS:
            raise LinesError("team must be team_a or team_b")
        self.root, self.match_id, self.team = Path(root), match_id, team
        self.tmp_dir, self.config_dir = self.root / ".tmp", self.root / "config"
        mdir = self.tmp_dir / match_id
        tl_path = mdir / "possession_timeline.parquet"
        if not (mdir / "tracks.parquet").exists() or not tl_path.exists():
            raise LinesError("Needs tracks.parquet and possession_timeline.parquet: run the pipeline "
                             "down to 'Compute possession' first")
        self.meta = load_meta(match_id, self.tmp_dir)
        self.fps = float(self.meta["fps"])
        self.cfg = match_config(match_id, self.tmp_dir, self.config_dir)
        timeline = pd.read_parquet(tl_path)
        lo, hi = int(timeline["frame"].min()), int(timeline["frame"].max())
        self.segs = [tuple(int(v) for v in sg) for sg in
                     phases(timeline, team, PHASE, self.fps, MERGE_S, MIN_S, PRE_S, POST_S, lo, hi)]
        self.by_start = {sg[0]: (k, sg) for k, sg in enumerate(self.segs)}
        self.rows_by_frame, self.own_goal_x = load_team_rows(mdir, self.tmp_dir, match_id, team, self.segs)
        self.rows_idx = index_rows(self.rows_by_frame)
        self.H_at = homography_lookup(load_calibration(match_id, self.config_dir))
        self.sel_path = selection_path(match_id, team, PHASE, self.config_dir)
        self.lock = threading.Lock()

    # ---- helpers ------------------------------------------------------
    def _seg(self, start):
        if start not in self.by_start:
            raise LinesError(f"No phase starts at frame {start}")
        return self.by_start[start]

    def _chains(self, slots, s, e):
        return build_chains(self.rows_idx, slots, s, e, self.fps, self.H_at)

    @staticmethod
    def _clean_slots(slots):
        out = []
        for sl in slots or []:
            if sl.get("line") not in ("back", "mid"):
                raise LinesError("slot line must be back or mid")
            anchors = sorted([int(f), int(t)] for f, t in sl.get("anchors", []))
            if not anchors:
                raise LinesError("every slot needs an anchor")
            out.append({"line": sl["line"], "anchors": anchors})
        return out

    def _chains_json(self, chains, s, e):
        """Per chain: line, and per frame [tid, foot_x, foot_y, pitch_y, status]
        (status s=seen, l=link (hand-off), f=filled). Missing frame = lost."""
        out = []
        for ch in chains:
            pts = {}
            for f, (tid, _px, py, fx, fy, st) in ch["pts"].items():
                if s <= f <= e:
                    pts[f] = [int(tid), round(fx, 1), round(fy, 1), round(py, 2), st[0]]
            out.append({"line": ch["line"], "pts": pts})
        return out

    def _status(self, entry):
        if entry is None:
            return "todo"
        return "picked" if entry_slots(entry) else "skipped"

    # ---- API ----------------------------------------------------------
    def summary(self):
        sel = load_selection(self.sel_path)
        out = []
        for k, (s, e, c0, c1) in enumerate(self.segs):
            entry = sel.get(s)
            slots = entry_slots(entry) if entry else []
            out.append({
                "n": k + 1, "start": s, "end": e, "core_start": c0, "core_end": c1,
                "clock": clock_label(c0, self.cfg, self.fps) or "",
                "secs": round((c1 - c0 + 1) / self.fps, 1),
                "status": self._status(entry),
                "n_back": sum(1 for sl in slots if sl["line"] == "back"),
                "n_mid": sum(1 for sl in slots if sl["line"] == "mid"),
            })
        names = {"team_a": self.cfg.get("team_a_name", "Team A"), "team_b": self.cfg.get("team_b_name", "Team B")}
        colors = self.cfg.get("colors") or {}
        return {"match_id": self.match_id, "team": self.team, "team_name": names[self.team],
                "opponent_name": names["team_b" if self.team == "team_a" else "team_a"],
                "color": colors.get(self.team) or ("#6CABDD" if self.team == "team_a" else "#F0F0F0"),
                "fps": self.fps, "width": self.meta["width"], "height": self.meta["height"],
                "calibrated": self.H_at is not None, "selection_file": self.sel_path.name, "phases": out}

    def phase(self, start):
        k, (s, e, c0, c1) = self._seg(start)
        entry = load_selection(self.sel_path).get(s)
        if entry and entry.get("keyframe") is not None:
            kf = int(entry["keyframe"])
        else:
            kf = int(pick_keyframe(self.rows_by_frame, c0, c1, self.own_goal_x))
        if entry is not None:
            slots, source = entry_slots(entry), "saved"
        else:
            rows = self.rows_by_frame.get(kf)
            slots, source = (auto_slots(rows, kf, self.own_goal_x) if rows is not None else []), "auto"
        rows = {}
        for f in range(s, e + 1):
            g = self.rows_by_frame.get(f)
            if g is not None:
                rows[f] = [[int(r.track_id), round(r.x1), round(r.y1), round(r.x2), round(r.y2),
                            round(r.foot_x, 1), round(r.foot_y, 1)] for r in g.itertuples()]
        return {"n": k + 1, "start": s, "end": e, "core_start": c0, "core_end": c1, "keyframe": kf,
                "status": self._status(entry), "source": source, "slots": slots,
                "chains": self._chains_json(self._chains(slots, s, e), s, e),
                "clocks": {f: clock_label(f, self.cfg, self.fps) for f in range(s, e + 1, int(self.fps))},
                "rows": rows}

    def edit(self, start, slots, action, frame=None, tid=None):
        """Apply one picker action to a draft (nothing saved). Actions:
        cycle (left-click: none -> back -> mid -> none), relink (right-click),
        auto (reset to the automatic guess at `frame`), clear, chains (just
        recompute). Returns the new slots + chains + a message."""
        _, (s, e, _c0, _c1) = self._seg(start)
        slots = self._clean_slots(slots)
        msg = ""
        if action in ("cycle", "relink"):
            if frame is None or tid is None or not s <= int(frame) <= e:
                raise LinesError("frame (inside the phase) and track id are required")
            frame, tid = int(frame), int(tid)
            cur = self.rows_idx.get(frame, {})
            if tid not in cur:
                raise LinesError("That player isn't detected on this frame")
            chains = self._chains(slots, s, e)
            owner = next((i for i, ch in enumerate(chains)
                          if ch["pts"].get(frame) and ch["pts"][frame][0] == tid
                          and ch["pts"][frame][5] != "filled"), None)
            if action == "cycle":
                if owner is None:
                    slots.append({"line": "back", "anchors": [[frame, tid]]})
                elif slots[owner]["line"] == "back":
                    slots[owner]["line"] = "mid"
                else:
                    slots.pop(owner)
            else:
                if owner is not None:
                    msg = "Already on a chain: re-link a player whose chain is lost or amber here"
                else:
                    r = cur[tid]
                    tgt = relink_target(chains, frame, float(r.pitch_x), float(r.pitch_y))
                    if tgt is None:
                        msg = "Every chain is detected on this frame: nothing to re-link"
                    else:
                        slots[tgt]["anchors"] = sorted(slots[tgt]["anchors"] + [[frame, tid]])
                        msg = f"Re-linked chain {tgt + 1} ({'back' if slots[tgt]['line'] == 'back' else 'midfield'})"
        elif action == "auto":
            f = int(frame) if frame is not None else s
            rows = self.rows_by_frame.get(f)
            slots = auto_slots(rows, f, self.own_goal_x) if rows is not None else []
            msg = "Reset to the automatic guess" if slots else "No automatic guess on this frame (too few players)"
        elif action == "clear":
            slots = []
        elif action != "chains":
            raise LinesError(f"Unknown action '{action}'")
        return {"slots": slots, "chains": self._chains_json(self._chains(slots, s, e), s, e), "msg": msg}

    def save(self, start, slots, keyframe=None, skip=False):
        _, (s, e, c0, c1) = self._seg(start)
        slots = [] if skip else self._clean_slots(slots)
        kf = int(keyframe) if keyframe is not None else int(pick_keyframe(self.rows_by_frame, c0, c1, self.own_goal_x))
        with self.lock:
            sel = load_selection(self.sel_path)
            sel[s] = {"keyframe": kf, "slots": slots}
            save_selection(self.sel_path, sel)
        return {"status": self._status(sel[s]), "start": s}


class LinesCache:
    """MatchLines per (match, team), rebuilt when the tracking artifacts change."""

    def __init__(self, root, size=2):
        self.root, self.size = Path(root), size
        self.items = OrderedDict()
        self.lock = threading.Lock()

    def _stamp(self, match_id):
        paths = [self.root / ".tmp" / match_id / n for n in
                 ("tracks.parquet", "possession_timeline.parquet", "player_map.json")]
        paths.append(self.root / "config" / f"{match_id}_calibration.json")
        return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)

    def get(self, match_id, team):
        key = (match_id, team)
        stamp = self._stamp(match_id)
        with self.lock:
            hit = self.items.get(key)
            if hit and hit[0] == stamp:
                self.items.move_to_end(key)
                return hit[1]
        ml = MatchLines(self.root, match_id, team)  # slow (reads tracks): outside the lock
        with self.lock:
            self.items[key] = (stamp, ml)
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return ml


class FrameSource:
    """JPEG frames from a match video. One VideoCapture per video, reading
    sequentially when the next frame is asked for (no seek), plus a small LRU
    of encoded frames for scrubbing back and forth."""

    def __init__(self, cache_frames=240):
        self.caps = {}
        self.cache = OrderedDict()
        self.cache_frames = cache_frames
        self.lock = threading.Lock()

    def jpeg(self, video, frame, width=None, quality=80):
        key = (video, frame, width)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            cap, pos = self.caps.get(video, (None, -1))
            if cap is None:
                cap = cv2.VideoCapture(str(video))
                if not cap.isOpened():
                    raise LinesError(f"Can't open {video}")
            if frame != pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            self.caps[video] = (cap, frame + 1 if ok else -1)
            if not ok:
                raise LinesError(f"Can't read frame {frame}")
            if width and width < img.shape[1]:
                img = cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            data = buf.tobytes()
            self.cache[key] = data
            while len(self.cache) > self.cache_frames:
                self.cache.popitem(last=False)
            return data
