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
import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import json  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

from _common import clock_label, load_calibration, load_meta, load_player_map, match_config  # noqa: E402
from identity_anchors import anchors_path, iou, load_anchors, save_anchors  # noqa: E402
from library import LibraryError, library_path, match_lineup  # noqa: E402
from library import load as load_library  # noqa: E402
from cut_phase_clips import (build_chains, entry_slots, export_path, export_range,  # noqa: E402
                             homography_lookup, index_rows, load_export, load_selection, load_team_rows,
                             norm_anchors, phases, save_export, save_selection, selection_path)
from select_shape_players import auto_slots, pick_keyframe, relink_target  # noqa: E402

PHASE = "out_of_possession"
TEAMS = ("team_a", "team_b")
# Same defaults as cut_phase_clips.py / select_shape_players.py: phases are
# keyed by start frame, so these must match or saved picks won't line up.
MERGE_S, MIN_S, PRE_S, POST_S = 2.0, 3.0, 1.5, 1.0
ANCHOR_LOCK = threading.Lock()  # one anchors file per match, shared by both teams' pickers


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
        self.exp_path = export_path(match_id, team, PHASE, self.config_dir)
        self.anchor_path = anchors_path(match_id, self.config_dir)
        self.tracks_path = mdir / "tracks.parquet"
        self.names = self._names(load_player_map(match_id, self.tmp_dir) or {})
        self.lock = threading.Lock()
        self._span_cache = {}

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
            anchors = norm_anchors(sl.get("anchors", []))
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
    def _export_json(self, seg, entry):
        keep, a, b = export_range(seg, entry)
        return {"keep": keep, "in": a, "out": b, "trimmed": (a, b) != (seg[0], seg[1]),
                "clip_s": round((b - a + 1) / self.fps, 1)}

    # ---- identify (issue #16) ------------------------------------------
    @property
    def lineup(self):
        """Read on every use (a small file), so library edits show without reloading the tracks."""
        return self._lineup()

    def _lineup(self):
        # the library (issue #16) knows the whole squad, so subs can be identified too
        try:
            lib = load_library(library_path(self.config_dir))
            lu = match_lineup(lib, self.match_id, self.team)
        except (OSError, LibraryError):
            lu = []
        if lu:
            return lu
        try:
            ref = json.loads((self.config_dir / f"{self.match_id}_reference.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        lu = (ref.get("lineups") or {}).get(self.team) or {}
        out = []
        for li, line in enumerate(lu.get("lines") or []):
            for person in line:
                if isinstance(person, dict) and person.get("number") is not None:
                    out.append({"number": int(person["number"]), "name": person.get("name"), "gk": li == 0})
        for person in lu.get("subs") or lu.get("substitutes") or []:
            if isinstance(person, dict) and person.get("number") is not None:
                out.append({"number": int(person["number"]), "name": person.get("name"), "gk": False, "sub": True})
        return out

    def _names(self, pm):
        out = {}
        for pl in pm.get("players", {}).values():
            if pl.get("team") != self.team:
                continue
            for t in pl.get("track_ids", []):
                out[int(t)] = {"number": pl.get("number"), "name": pl.get("name"),
                               "manual": str(pl.get("name_source", "")).startswith("manual")}
        return out

    def _spans(self, tracklets):
        """First and last frame of each tracklet (cached; tracklet ids are global)."""
        want = [int(t) for t in tracklets if t is not None and int(t) not in self._span_cache]
        if want:
            try:
                d = pd.read_parquet(self.tracks_path, filters=[("tracklet", "in", want)], columns=["frame", "tracklet"])
                for t, g in d.groupby("tracklet"):
                    self._span_cache[int(t)] = [int(g["frame"].min()), int(g["frame"].max())]
            except (OSError, ValueError, KeyError):
                pass
            for t in want:
                self._span_cache.setdefault(t, None)
        return self._span_cache

    def _anchors_json(self):
        stamp = self.tracks_path.stat().st_mtime if self.tracks_path.exists() else 0
        mine = [a for a in load_anchors(self.anchor_path) if a.get("team") == self.team]
        spans = self._spans(a.get("tracklet") for a in mine)
        for a in mine:
            a["pending"] = a.get("created", 0) > stamp
            # until "Apply to tracking" runs, the picker shows the anchored number on
            # this stretch of the track (the anchored tracklet, or just the frame)
            t = a.get("tracklet")
            a["span"] = (spans.get(int(t)) if t is not None else None) or [int(a["frame"]), int(a["frame"])]
            a["phase"] = next((sg[0] for sg in self.segs if sg[0] <= a["frame"] <= sg[1]), None)
            a["clock"] = clock_label(int(a["frame"]), self.cfg, self.fps) or ""
        return sorted(mine, key=lambda a: a["frame"])

    def identify(self, frame, box, number):
        """Anchor "the detected player in `box` at `frame` is No. `number`".
        The box can be a clicked detection's own box or a rough box dragged
        round a player: the detection whose feet are inside it (best overlap
        first) is the one anchored."""
        person = next((p for p in self.lineup if p["number"] == int(number)), None)
        if person is None:
            raise LinesError(f"No. {number} isn't in the {self.team} lineup (config/<id>_reference.json)")
        if not box or len(box) != 4:
            raise LinesError("box must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(v) for v in box)
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        rows = pd.read_parquet(self.tracks_path, filters=[("frame", "==", int(frame))],
                               columns=["frame", "track_id", "class", "x1", "y1", "x2", "y2", "foot_x", "foot_y",
                                        "tracklet"])
        rows = rows[rows["class"] == "player"]
        best, score = None, 0.0
        for r in rows.itertuples():
            v = iou([r.x1, r.y1, r.x2, r.y2], box)
            inside = box[0] <= r.foot_x <= box[2] and box[1] <= r.foot_y <= box[3]
            sc = v + (1.0 if inside else 0.0)
            if (inside or v >= 0.3) and sc > score:
                best, score = r, sc
        if best is None:
            raise LinesError("No detected player in that box at this frame. Try a frame where the player "
                             "is visible and has a box (the detector missed them here)")
        anchor = {"id": uuid.uuid4().hex[:10], "team": self.team, "number": person["number"],
                  "name": person["name"], "frame": int(frame),
                  "box": [round(float(v), 1) for v in (best.x1, best.y1, best.x2, best.y2)],
                  "track_id": int(best.track_id),
                  "tracklet": int(best.tracklet) if pd.notna(best.tracklet) and best.tracklet >= 0 else None,
                  "created": time.time()}
        with ANCHOR_LOCK:
            anchors = load_anchors(self.anchor_path)
            # one anchor per detection: re-identifying the same box replaces it
            anchors = [a for a in anchors if not (int(a["frame"]) == anchor["frame"]
                                                  and iou(a.get("box") or [0, 0, 0, 0], anchor["box"]) > 0.7)]
            anchors.append(anchor)
            save_anchors(self.anchor_path, anchors)
        return {"anchor": anchor, "anchors": self._anchors_json()}

    def unidentify(self, anchor_id):
        with ANCHOR_LOCK:
            anchors = load_anchors(self.anchor_path)
            keep = [a for a in anchors if a.get("id") != anchor_id]
            if len(keep) == len(anchors):
                raise LinesError("No such anchor")
            save_anchors(self.anchor_path, keep)
        return {"anchors": self._anchors_json()}

    def summary(self):
        sel = load_selection(self.sel_path)
        exp = load_export(self.exp_path)
        out = []
        for k, (s, e, c0, c1) in enumerate(self.segs):
            entry = sel.get(s)
            slots = entry_slots(entry) if entry else []
            out.append({
                "export": self._export_json((s, e), exp.get(s)),
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
                "calibrated": self.H_at is not None, "selection_file": self.sel_path.name,
                "export_file": self.exp_path.name, "phases": out,
                "lineup": self.lineup, "names": {str(k): v for k, v in self.names.items()},
                "anchors": self._anchors_json(), "anchors_file": self.anchor_path.name}

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

    def swap_moment(self, t_old, t_new, frame, s, look_s=6.0):
        """The frame the tracker most likely swapped ids t_old <-> t_new: their
        closest approach (pitch distance) in the `look_s` before `frame`, while
        both are detected. None if they're never both seen in that window."""
        best, bd = None, 1e9
        for f in range(frame, max(s, frame - int(look_s * self.fps)) - 1, -1):
            cur = self.rows_idx.get(f)
            if not cur or t_old not in cur or t_new not in cur:
                continue
            a, b = cur[t_old], cur[t_new]
            d = float(np.hypot(a.pitch_x - b.pitch_x, a.pitch_y - b.pitch_y))
            if d < bd - 1e-6:  # strict: ties keep the later frame
                best, bd = f, d
        return best, bd

    def _first_seen(self, tid, frm, to):
        for f in range(frm, to + 1):
            if tid in self.rows_idx.get(f, {}):
                return f
        return None

    def _recut(self, anchors, cut, frame, s, t_before, new_anchor):
        """Anchors with `new_anchor` (a cut) added and the anchors between the
        cut and the click dropped (they pointed at the wrong player). If that
        leaves nothing before the cut, pin the pre-swap track just before it
        so the chain still follows the right player up to the collision."""
        kept = [a for a in anchors if not (cut <= a[0] <= frame)]
        if t_before is not None and not any(a[0] < cut for a in kept):
            for f in range(cut - 1, max(s, cut - int(4 * self.fps)) - 1, -1):
                if t_before in self.rows_idx.get(f, {}):
                    kept.append([f, t_before])
                    break
        return norm_anchors(kept + [new_anchor])

    def _swap(self, slots, s, e, chain, frame, tid):
        """Put chain `chain` onto track `tid` from the swap moment on; if `tid`
        was on another picked chain, that chain takes this chain's old track
        from the same moment (the two identities swap back)."""
        if not 0 <= chain < len(slots):
            raise LinesError("Pick the chain to fix first")
        chains = self._chains(slots, s, e)
        mine = chains[chain]["pts"].get(frame)
        if mine and mine[0] == tid and mine[5] != "filled":
            return slots, "That chain is already on this player here"
        t_old = int(mine[0]) if mine else None
        partner = next((i for i, ch in enumerate(chains) if i != chain and ch["pts"].get(frame)
                        and ch["pts"][frame][0] == tid and ch["pts"][frame][5] != "filled"), None)
        cut, dist = (None, None)
        if t_old is not None and t_old != tid:
            cut, dist = self.swap_moment(t_old, tid, frame, s)
        if cut is None:
            cut = frame
        start_new = self._first_seen(tid, cut, frame)
        slots[chain]["anchors"] = self._recut(slots[chain]["anchors"], cut, frame, s, t_old, [start_new, tid, 1])
        line = "back" if slots[chain]["line"] == "back" else "midfield"
        when = f"{(frame - cut) / self.fps:.1f} s earlier" if cut < frame else "here"
        msg = f"Moved the {line} chain onto this player from the swap point ({when}"
        msg += f", {dist:.1f} m apart)" if dist is not None else ")"
        if partner is not None and t_old is not None:
            p_new = self._first_seen(t_old, cut, frame)
            if p_new is not None:
                slots[partner]["anchors"] = self._recut(slots[partner]["anchors"], cut, frame, s, tid,
                                                        [p_new, t_old, 1])
                msg += "; swapped the other picked chain back too"
        return slots, msg

    def edit(self, start, slots, action, frame=None, tid=None, chain=None):
        """Apply one picker action to a draft (nothing saved). Actions:
        cycle (left-click: none -> back -> mid -> none), relink (right-click),
        swap (fix an identity swap: chain `chain` belongs on `tid`), auto
        (reset to the automatic guess at `frame`), clear, chains (just
        recompute). Returns the new slots + chains + a message."""
        _, (s, e, _c0, _c1) = self._seg(start)
        slots = self._clean_slots(slots)
        msg = ""
        if action == "swap":
            if frame is None or tid is None or chain is None or not s <= int(frame) <= e:
                raise LinesError("frame (inside the phase), chain and track id are required")
            if int(tid) not in self.rows_idx.get(int(frame), {}):
                raise LinesError("That player isn't detected on this frame")
            slots, msg = self._swap(slots, s, e, int(chain), int(frame), int(tid))
        elif action in ("cycle", "relink"):
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


    def set_export(self, start, keep=None, trim_in=None, trim_out=None, reset_trim=False):
        """Issue #37: keep/drop one phase for the clip export and/or trim its
        start/end (frames inside the phase). Saved straight away to
        config/<id>_<team>_<phase>_export.json, which cut_phase_clips.py
        --export-selection reads."""
        _, (s, e, _c0, _c1) = self._seg(start)
        for v in (trim_in, trim_out):
            if v is not None and not s <= int(v) <= e:
                raise LinesError("Trim points must be inside the phase")
        with self.lock:
            exp = load_export(self.exp_path)
            cur = dict(exp.get(s) or {"keep": True, "in": None, "out": None})
            if keep is not None:
                cur["keep"] = bool(keep)
            if reset_trim:
                cur["in"] = cur["out"] = None
            if trim_in is not None:
                cur["in"] = None if int(trim_in) == s else int(trim_in)
            if trim_out is not None:
                cur["out"] = None if int(trim_out) == e else int(trim_out)
            a = cur["in"] if cur["in"] is not None else s
            b = cur["out"] if cur["out"] is not None else e
            if b - a < self.fps:  # at least a second
                raise LinesError("The clip's end must be at least 1 s after its start")
            exp[s] = cur
            save_export(self.exp_path, exp)
        return {"start": s, "export": self._export_json((s, e), cur)}

    def set_export_all(self, keep):
        """Keep or drop every phase (trims are kept)."""
        with self.lock:
            exp = load_export(self.exp_path)
            for s, *_ in self.segs:
                cur = dict(exp.get(s) or {"in": None, "out": None})
                cur["keep"] = bool(keep)
                exp[s] = cur
            save_export(self.exp_path, exp)
        return {"kept": len(self.segs) if keep else 0}


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
