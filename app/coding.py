"""Code window (issue #3): configurable tagging buttons and the instances they make.

A *template* is a button layout, saved as config/code_windows/<name>.json:

    {"name": "Default",
     "buttons": [{"id": "pass", "label": "Pass", "key": "q", "color": "#4C8DF6",
                  "lead_s": 3, "lag_s": 5}, ...],
     "label_groups": [{"id": "team", "name": "Team", "source": "teams"},
                      {"id": "player", "name": "Player", "source": "lineup"},
                      {"id": "outcome", "name": "Outcome",
                       "labels": [{"label": "Success", "key": "1"}, ...]}]}

Pressing a button (click or its hotkey) makes an *instance* at the video time
t, covering [t - lead_s, t + lag_s] clamped to the video. Label groups attach
descriptors to an instance, one label per group. A group's labels are listed
in the template, or come from the match: "teams" gives the two team names and
"lineup" the players of config/<id>_reference.json (lineups).

A match's instances live in config/<id>_instances.json. They are hand-made
work, so they sit in config/ with the other hand-checked inputs, not in .tmp.
Each instance keeps its code's name and colour, so editing or switching the
template later never changes what was already tagged.
"""
import csv
import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,59}$")
COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
# keys the code window uses itself (transport, delete), so buttons can't take them
RESERVED_KEYS = {" ", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Delete", "Backspace",
                 "Escape", "Enter", "Tab", ",", ".", "<", ">", "[", "]"}
MAX_LEAD_LAG_S = 120
SOURCES = ("teams", "lineup")

DEFAULT_TEMPLATE = {
    "name": "Default",
    "buttons": [
        {"id": "pass", "label": "Pass", "key": "q", "color": "#4C8DF6", "lead_s": 3, "lag_s": 4},
        {"id": "shot", "label": "Shot", "key": "w", "color": "#EF5B5B", "lead_s": 8, "lag_s": 5},
        {"id": "cross", "label": "Cross", "key": "e", "color": "#7FB0FF", "lead_s": 6, "lag_s": 5},
        {"id": "foul", "label": "Foul", "key": "r", "color": "#F08C2E", "lead_s": 5, "lag_s": 5},
        {"id": "corner", "label": "Corner", "key": "t", "color": "#B98CF2", "lead_s": 2, "lag_s": 12},
        {"id": "free_kick", "label": "Free kick", "key": "y", "color": "#D07AD6", "lead_s": 2, "lag_s": 10},
        {"id": "press", "label": "Press", "key": "a", "color": "#FFD84D", "lead_s": 4, "lag_s": 6},
        {"id": "turnover", "label": "Turnover", "key": "s", "color": "#E8A33A", "lead_s": 5, "lag_s": 6},
        {"id": "recovery", "label": "Ball recovery", "key": "d", "color": "#5DBB63", "lead_s": 5, "lag_s": 6},
        {"id": "transition", "label": "Transition", "key": "f", "color": "#3FB8AF", "lead_s": 3, "lag_s": 10},
        {"id": "goal", "label": "Goal", "key": "g", "color": "#FFFFFF", "lead_s": 15, "lag_s": 10},
        {"id": "save", "label": "GK save", "key": "h", "color": "#9AA1AB", "lead_s": 6, "lag_s": 4},
    ],
    "label_groups": [
        {"id": "team", "name": "Team", "source": "teams"},
        {"id": "player", "name": "Player", "source": "lineup"},
        {"id": "outcome", "name": "Outcome",
         "labels": [{"label": "Success", "key": "1"}, {"label": "Fail", "key": "2"}]},
        {"id": "zone", "name": "Zone",
         "labels": [{"label": "Defensive third", "key": "3"}, {"label": "Middle third", "key": "4"},
                    {"label": "Attacking third", "key": "5"}]},
    ],
}


class CodingError(ValueError):
    """Bad template or instance edit (shown to the user as a 400)."""


def _slug(text):
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:40] or "code"


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(v, what):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise CodingError(f"{what} must be a number") from None
    if not 0 <= x <= MAX_LEAD_LAG_S:
        raise CodingError(f"{what} must be between 0 and {MAX_LEAD_LAG_S} s")
    return round(x, 2)


def _key(v, used, owner):
    if v in (None, ""):
        return None
    if not isinstance(v, str) or len(v) != 1 or not v.isprintable():
        raise CodingError(f"{owner}: the hotkey must be a single character")
    if v in RESERVED_KEYS:
        raise CodingError(f"{owner}: '{v}' is used by the player controls")
    k = v.lower() if v.isalpha() else v
    if k in used:
        raise CodingError(f"{owner}: hotkey '{v}' is already used by {used[k]}")
    used[k] = owner
    return k


def validate_template(data):
    """Clean a template from the browser; raises CodingError on anything off."""
    if not isinstance(data, dict):
        raise CodingError("Template must be an object")
    name = str(data.get("name") or "").strip()
    if not NAME_RE.match(name):
        raise CodingError("Template name: letters, digits, space, _ . - (up to 60)")
    buttons, ids, used = [], set(), {}
    for b in data.get("buttons") or []:
        label = str((b or {}).get("label") or "").strip()[:40]
        if not label:
            raise CodingError("Every button needs a label")
        bid = _slug(b.get("id") or label)
        while bid in ids:
            bid += "_2"
        ids.add(bid)
        color = b.get("color") or "#4C8DF6"
        if not COLOR_RE.match(color):
            raise CodingError(f"{label}: colour must look like #RRGGBB")
        buttons.append({"id": bid, "label": label, "key": _key(b.get("key"), used, label),
                        "color": color.upper(), "lead_s": _num(b.get("lead_s", 5), f"{label} lead"),
                        "lag_s": _num(b.get("lag_s", 5), f"{label} lag")})
    if not buttons:
        raise CodingError("A template needs at least one button")
    if len(buttons) > 60:
        raise CodingError("At most 60 buttons")
    groups, gids = [], set()
    for g in data.get("label_groups") or []:
        gname = str((g or {}).get("name") or "").strip()[:30]
        if not gname:
            raise CodingError("Every label group needs a name")
        gid = _slug(g.get("id") or gname)
        if gid in gids:
            raise CodingError(f"Two label groups are called '{gname}'")
        gids.add(gid)
        src = g.get("source") or None
        if src is not None and src not in SOURCES:
            raise CodingError(f"{gname}: source must be one of {', '.join(SOURCES)}")
        out = {"id": gid, "name": gname}
        if src:
            out["source"] = src
        else:
            labels, seen = [], set()
            for lab in g.get("labels") or []:
                lab = {"label": lab} if isinstance(lab, str) else (lab or {})
                text = str(lab.get("label") or "").strip()[:40]
                if not text or text in seen:
                    continue
                seen.add(text)
                labels.append({"label": text, "key": _key(lab.get("key"), used, f"{gname}: {text}")})
            if not labels:
                raise CodingError(f"{gname}: add at least one label (or pick a source)")
            out["labels"] = labels
        groups.append(out)
    return {"name": name, "buttons": buttons, "label_groups": groups}


class CodeStore:
    """Templates in config/code_windows/, instances in config/<id>_instances.json."""

    def __init__(self, root):
        self.root = Path(root)
        self.cfg = self.root / "config"
        self.tdir = self.cfg / "code_windows"
        self._lock = threading.Lock()

    # ---------- templates ----------
    def _tpath(self, name):
        if not NAME_RE.match(name or ""):
            raise CodingError("Bad template name")
        return self.tdir / f"{_slug(name)}.json"

    def list_templates(self):
        out = {}
        if self.tdir.is_dir():
            for p in sorted(self.tdir.glob("*.json")):
                d = _read_json(p)
                if isinstance(d, dict) and d.get("name"):
                    out[d["name"]] = {"name": d["name"], "buttons": len(d.get("buttons") or []), "saved": True}
        if DEFAULT_TEMPLATE["name"] not in out:
            out[DEFAULT_TEMPLATE["name"]] = {"name": DEFAULT_TEMPLATE["name"],
                                             "buttons": len(DEFAULT_TEMPLATE["buttons"]), "saved": False}
        return sorted(out.values(), key=lambda t: (t["name"] != DEFAULT_TEMPLATE["name"], t["name"].lower()))

    def get_template(self, name):
        d = _read_json(self._tpath(name))
        if isinstance(d, dict) and d.get("buttons"):
            return d
        if name == DEFAULT_TEMPLATE["name"]:
            return json.loads(json.dumps(DEFAULT_TEMPLATE))
        raise KeyError(name)

    def save_template(self, data):
        t = validate_template(data)
        with self._lock:
            _write_json_atomic(self._tpath(t["name"]), t)
        return t

    def delete_template(self, name):
        p = self._tpath(name)
        if not p.exists():
            raise KeyError(name)
        p.unlink()

    # ---------- match labels ----------
    def match_labels(self, match_id, team_names):
        """Labels for "teams"/"lineup" groups: [{label, team}] per source."""
        teams = [{"label": team_names.get("team_a") or "Team A", "team": "team_a"},
                 {"label": team_names.get("team_b") or "Team B", "team": "team_b"}]
        players = []
        ref = _read_json(self.cfg / f"{match_id}_reference.json") or {}
        for team in ("team_a", "team_b"):
            lu = (ref.get("lineups") or {}).get(team) or {}
            entries = [p for line in lu.get("lines") or [] for p in line]
            entries += lu.get("subs") or lu.get("substitutes") or []
            for p in entries:
                if not isinstance(p, dict) or not p.get("name"):
                    continue
                num = p.get("number")
                players.append({"label": f"{num} {p['name']}" if num is not None else p["name"],
                                "team": team})
        return {"teams": teams, "lineup": players}

    # ---------- instances ----------
    def _ipath(self, match_id):
        return self.cfg / f"{match_id}_instances.json"

    def _load(self, match_id):
        d = _read_json(self._ipath(match_id))
        if not isinstance(d, dict) or not isinstance(d.get("instances"), list):
            d = {"instances": []}
        return d

    def instances(self, match_id):
        return self._load(match_id)

    def _save(self, match_id, d):
        d["instances"].sort(key=lambda i: (i["t"], i.get("created", 0)))
        d["updated"] = time.time()
        _write_json_atomic(self._ipath(match_id), d)

    def add(self, match_id, duration_s, code, color, t, lead_s, lag_s, labels=None, template=None):
        code = str(code or "").strip()[:40]
        if not code:
            raise CodingError("Instance needs a code")
        if color and not COLOR_RE.match(color):
            raise CodingError("Bad colour")
        t = float(t)
        if not 0 <= t <= (duration_s or float("inf")):
            raise CodingError("Time is outside the video")
        lead, lag = _num(lead_s, "Lead"), _num(lag_s, "Lag")
        end_max = duration_s if duration_s else t + lag
        inst = {"id": uuid.uuid4().hex[:10], "code": code, "color": (color or "#4C8DF6").upper(),
                "t": round(t, 3), "start": round(max(0.0, t - lead), 3), "end": round(min(end_max, t + lag), 3),
                "labels": _clean_labels(labels), "created": time.time()}
        with self._lock:
            d = self._load(match_id)
            if template:
                d["template"] = str(template)[:60]
            d["instances"].append(inst)
            self._save(match_id, d)
        return inst

    def update(self, match_id, iid, duration_s, labels=None, set_label=None, start=None, end=None):
        """labels replaces all; set_label={group, label|None} sets or clears one
        group; start/end trim the clip (start <= t <= end is not required)."""
        with self._lock:
            d = self._load(match_id)
            inst = next((i for i in d["instances"] if i["id"] == iid), None)
            if inst is None:
                raise KeyError(iid)
            if labels is not None:
                inst["labels"] = _clean_labels(labels)
            if set_label is not None:
                group = str(set_label.get("group") or "").strip()[:30]
                if not group:
                    raise CodingError("set_label needs a group")
                value = set_label.get("label")
                if value in (None, ""):
                    inst["labels"].pop(group, None)
                else:
                    inst["labels"][group] = str(value)[:60]
            s = inst["start"] if start is None else float(start)
            e = inst["end"] if end is None else float(end)
            if not 0 <= s < e or (duration_s and e > duration_s + 1e-6):
                raise CodingError("Clip must start before it ends and stay inside the video")
            inst["start"], inst["end"] = round(s, 3), round(e, 3)
            self._save(match_id, d)
            return inst

    def delete(self, match_id, iid):
        with self._lock:
            d = self._load(match_id)
            n = len(d["instances"])
            d["instances"] = [i for i in d["instances"] if i["id"] != iid]
            if len(d["instances"]) == n:
                raise KeyError(iid)
            self._save(match_id, d)

    def csv_text(self, match_id, clock=None):
        """Instances as CSV, one column per label group (spreadsheet / #15 export)."""
        inst = self._load(match_id)["instances"]
        groups = []
        for i in inst:
            for g in i["labels"]:
                if g not in groups:
                    groups.append(g)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["code", "time_s", "match_clock", "start_s", "end_s", "duration_s"] + groups)
        for i in inst:
            w.writerow([i["code"], i["t"], clock(i["t"]) if clock else "", i["start"], i["end"],
                        round(i["end"] - i["start"], 3)] + [i["labels"].get(g, "") for g in groups])
        return buf.getvalue()


def _clean_labels(labels):
    if not labels:
        return {}
    if not isinstance(labels, dict):
        raise CodingError("labels must be {group: label}")
    return {str(k).strip()[:30]: str(v)[:60] for k, v in labels.items()
            if str(k).strip() and v not in (None, "")}
