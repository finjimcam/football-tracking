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

from _common import load_meta  # noqa: E402
from export_clips import (GROUP_TITLES, KIND_TITLES, export_path, export_range, load_clips,  # noqa: E402
                          load_export, save_export, zip_path)

TEAMS = ("team_a", "team_b")


class ClipsError(Exception):
    pass


class Clips:
    def __init__(self, root, match_id):
        self.root, self.match_id = Path(root), match_id
        self.path = self.root / ".tmp" / match_id / "clips.json"
        self.exp_path = export_path(match_id, self.root / "config")
        self.lock = threading.Lock()

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


def clips_file_json(path):
    """For tests / debugging: the raw clips.json."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
