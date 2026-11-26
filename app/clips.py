"""Clips panel of the Lines tab (issue #58): set pieces, shots & key chances and
passes into the final third, found by tools/detect_clips.py.

Reads .tmp/<id>/clips.json and keeps the keep / drop / trim choices in
config/<id>_clips_export.json, which tools/export_clips.py reads:

    {"clips": {"<clip id>": {"keep": bool, "in": frame|null, "out": frame|null}}}

Clips are kept unless dropped, like the phase export.
"""
import json
import sys
import threading
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import clip_marks as CM  # noqa: E402
from _common import load_calibration, load_meta, match_config  # noqa: E402
from cut_phase_clips import homography_lookup  # noqa: E402
from select_shape_players import relink_target  # noqa: E402
from export_clips import (GROUP_TITLES, KIND_TITLES, export_path, export_range, load_clips,  # noqa: E402
                          load_export, people, save_export, zip_path)

TEAMS = ("team_a", "team_b")
PREVIEW_CAP_MB = 400  # clip preview videos kept in .tmp/<id>/clip_previews (oldest go first)

_H_CACHE = {}  # match_id -> (calibration mtime, frame -> H)
_PREVIEW_LOCKS = {}
_PREVIEW_GUARD = threading.Lock()


def _homographies(root, match_id):
    cfg_dir = Path(root) / "config"
    p = cfg_dir / f"{match_id}_calibration.json"
    mt = p.stat().st_mtime if p.exists() else None
    hit = _H_CACHE.get(match_id)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        H_at = homography_lookup(load_calibration(match_id, cfg_dir))
    except SystemExit:
        H_at = None
    _H_CACHE[match_id] = (mt, H_at)
    return H_at


class ClipsError(Exception):
    pass


class Clips:
    lock = threading.Lock()  # one writer at a time for the export / marks files

    def __init__(self, root, match_id):
        self.root, self.match_id = Path(root), match_id
        self.tmp_dir, self.config_dir = self.root / ".tmp", self.root / "config"
        self.path = self.tmp_dir / match_id / "clips.json"
        self.exp_path = export_path(match_id, self.config_dir)
        self.marks_path = CM.marks_path(match_id, self.config_dir)

    def _data(self):
        if not self.path.exists():
            raise ClipsError("No clips yet: run 'Detect clips' on the match page (it needs Compute possession first)")
        try:
            return load_clips(self.path)
        except (OSError, ValueError) as e:
            raise ClipsError(f"Can't read {self.path.name}: {e}") from e

    @staticmethod
    def _export_json(clip, entry):
        keep, a, b = export_range(clip, entry)
        return {"keep": keep, "in": a, "out": b, "trimmed": a != clip["start"] or b != clip["end"]}

    def summary(self, team):
        if team not in TEAMS:
            raise ClipsError("team must be team_a or team_b")
        d = self._data()
        exp = load_export(self.exp_path)
        groups = []
        for g, kinds in d["groups"].items():
            subs = []
            for k in kinds:
                items = [dict(c, export=self._export_json(c, exp.get(c["id"])))
                         for c in d["clips"] if c["kind"] == k and c.get("team") in (team, None)]
                if items or k != "penalty":  # penalties only when there are some
                    subs.append({"kind": k, "title": KIND_TITLES.get(k, k), "clips": items})
            groups.append({"group": g, "title": GROUP_TITLES.get(g, g), "kinds": subs})
        marks = CM.load_marks(self.marks_path)
        for g in groups:
            for k in g["kinds"]:
                for c in k["clips"]:
                    c["n_marks"] = len(marks.get(c["id"], []))
        try:
            meta = load_meta(self.match_id, self.root / ".tmp")
        except (OSError, ValueError, SystemExit):
            meta = {}
        return {"match_id": self.match_id, "team": team, "team_name": d["teams"].get(team, team),
                "fps": d["fps"], "generated": d.get("generated"), "groups": groups,
                "width": meta.get("width"), "height": meta.get("height"),
                "zip_exists": zip_path(self.match_id, team, outputs_dir=self.root / "outputs").exists(),
                "export_file": self.exp_path.name}

    def _clip(self, clip_id):
        for c in self._data()["clips"]:
            if c["id"] == clip_id:
                return c
        raise ClipsError(f"No clip {clip_id}")

    def set_export(self, clip_id, keep=None, trim_in=None, trim_out=None, reset_trim=False):
        c = self._clip(clip_id)
        with self.lock:
            exp = load_export(self.exp_path)
            e = dict(exp.get(clip_id) or {"keep": True, "in": None, "out": None})
            if keep is not None:
                e["keep"] = bool(keep)
            if reset_trim:
                e["in"] = e["out"] = None
            if trim_in is not None:
                e["in"] = min(max(int(trim_in), c["start"]), c["end"])
            if trim_out is not None:
                e["out"] = min(max(int(trim_out), c["start"]), c["end"])
            exp[clip_id] = e
            save_export(self.exp_path, exp)
        return self._export_json(c, e)

    def set_export_many(self, team, keep, group=None, kind=None):
        d = self._data()
        n = 0
        with self.lock:
            exp = load_export(self.exp_path)
            for c in d["clips"]:
                if c.get("team") not in (team, None) or (group and c["group"] != group) or (kind and c["kind"] != kind):
                    continue
                e = dict(exp.get(c["id"]) or {"keep": True, "in": None, "out": None})
                e["keep"] = bool(keep)
                exp[c["id"]] = e
                n += 1
            save_export(self.exp_path, exp)
        return {"changed": n}


    # ---- one clip: players, marks (lines / highlights), preview video ----------------
    def _people(self):
        return people(self.match_id, self.tmp_dir)

    def _fps(self):
        return float(load_meta(self.match_id, self.tmp_dir)["fps"])

    def _chains_json(self, chain_list, slots, names):
        labels = CM.slot_labels(slots, names)
        out = []
        for i, ch in enumerate(chain_list):
            pts = {f: [int(tid), round(fx, 1), round(fy, 1), round(py, 2), st[0]]
                   for f, (tid, _px, py, fx, fy, st) in ch["pts"].items()}
            out.append({"line": ch["line"], "label": labels.get(i), "pts": pts})
        return out

    def clip(self, clip_id):
        """The clip, every detected player on each of its frames, and its marks as chains."""
        c = self._clip(clip_id)
        fps = self._fps()
        names, team_of = self._people()
        rows_by_frame = CM.clip_rows(self.match_id, self.tmp_dir, self.config_dir, c["start"], c["end"], fps)
        slots = CM.load_marks(self.marks_path).get(clip_id, [])
        chain_list = CM.chains(rows_by_frame, slots, c["start"], c["end"], fps, _homographies(self.root, self.match_id))
        rows = {f: [[int(r.track_id), round(r.x1), round(r.y1), round(r.x2), round(r.y2),
                     round(r.foot_x, 1), round(r.foot_y, 1)] for r in g.itertuples()]
                for f, g in rows_by_frame.items()}
        tids = {int(t) for g in rows_by_frame.values() for t in g["track_id"]}
        cfg = match_config(self.match_id, self.tmp_dir, self.config_dir)
        return {"clip": c, "rows": rows, "slots": slots, "chains": self._chains_json(chain_list, slots, names),
                "names": {str(t): names[t] for t in tids if t in names},
                "teams": {str(t): team_of[t] for t in tids if t in team_of},
                "colors": cfg.get("colors") or {}}

    def edit(self, clip_id, action, frame=None, tid=None, slots=None):
        """Change a clip's marks and save them. Actions: line (tap a player: none -> back ->
        midfield -> none), highlight (tap: on / off), relink (put a lost line or highlight back on
        this player), clear, set (replace with `slots`)."""
        c = self._clip(clip_id)
        fps = self._fps()
        names, _ = self._people()
        rows_by_frame = CM.clip_rows(self.match_id, self.tmp_dir, self.config_dir, c["start"], c["end"], fps)
        H_at = _homographies(self.root, self.match_id)
        with self.lock:
            marks = CM.load_marks(self.marks_path)
            cur = [dict(sl) for sl in marks.get(clip_id, [])]
            msg = ""
            if action == "set":
                try:
                    cur = CM.clean_slots(slots)
                except ValueError as e:
                    raise ClipsError(str(e)) from None
            elif action == "clear":
                cur = []
            elif action in ("line", "highlight", "relink"):
                if frame is None or tid is None or not c["start"] <= int(frame) <= c["end"]:
                    raise ClipsError("frame (inside the clip) and track id are required")
                frame, tid = int(frame), int(tid)
                here = rows_by_frame.get(frame)
                if here is None or tid not in set(here["track_id"].astype(int)):
                    raise ClipsError("That player isn't detected on this frame")
                chain_list = CM.chains(rows_by_frame, cur, c["start"], c["end"], fps, H_at)
                owner = next((i for i, ch in enumerate(chain_list) if ch["pts"].get(frame)
                              and ch["pts"][frame][0] == tid and ch["pts"][frame][5] != "filled"), None)
                if action == "highlight":
                    if owner is not None and cur[owner]["line"] == "hl":
                        cur.pop(owner)
                    elif owner is not None:
                        cur[owner]["line"] = "hl"
                    else:
                        cur.append({"line": "hl", "anchors": [[frame, tid]]})
                elif action == "line":
                    if owner is None:
                        cur.append({"line": "back", "anchors": [[frame, tid]]})
                    elif cur[owner]["line"] == "back":
                        cur[owner]["line"] = "mid"
                    else:
                        cur.pop(owner)
                else:
                    if owner is not None:
                        msg = "Already marked: re-link a player whose mark is lost or amber here"
                    else:
                        r = here[here["track_id"].astype(int) == tid].iloc[0]
                        tgt = relink_target(chain_list, frame, float(r.pitch_x), float(r.pitch_y))
                        if tgt is None:
                            msg = "Every mark is on a player here: nothing to re-link"
                        else:
                            cur[tgt]["anchors"] = sorted(cur[tgt]["anchors"] + [[frame, tid]])
                            msg = "Re-linked"
            else:
                raise ClipsError(f"Unknown action '{action}'")
            marks[clip_id] = cur
            CM.save_marks(self.marks_path, marks)
        chain_list = CM.chains(rows_by_frame, cur, c["start"], c["end"], fps, H_at)
        return {"slots": cur, "chains": self._chains_json(chain_list, cur, names), "msg": msg}

    def preview(self, clip_id):
        """Path of the clip's preview MP4, made on first use (a few seconds)."""
        c = self._clip(clip_id)
        meta = load_meta(self.match_id, self.tmp_dir)
        pdir = self.tmp_dir / self.match_id / "clip_previews"
        out = pdir / f"{c['start']}_{c['end']}.mp4"
        with _PREVIEW_GUARD:
            lk = _PREVIEW_LOCKS.setdefault(str(out), threading.Lock())
        with lk:
            if not out.exists():
                CM.make_preview(meta["video"], c["start"], c["end"], float(meta["fps"]), out)
                _trim_previews(pdir, keep=out)
        return out

    def prefetch(self, clip_ids):
        """Make the next clips' previews in the background, one at a time."""
        def run():
            for cid in clip_ids:
                try:
                    self.preview(cid)
                except Exception:  # noqa: BLE001 - best effort
                    pass
        threading.Thread(target=run, daemon=True, name="clip-prefetch").start()


def _trim_previews(pdir, keep=None, cap_mb=PREVIEW_CAP_MB):
    files = sorted((p for p in Path(pdir).glob("*.mp4")), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= cap_mb * 1e6:
            break
        if p != keep:
            total -= p.stat().st_size
            p.unlink(missing_ok=True)


def clips_file_json(path):
    """For tests / debugging: the raw clips.json."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
