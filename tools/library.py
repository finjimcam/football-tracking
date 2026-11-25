"""Library of teams, players, competitions, seasons and matches (issue #16).

Everything lives in one hand-editable file, config/library.json:

  teams         {id: {name, short, color, kit}}
  players       {id: {name, positions: [..], notes}}
  competitions  {id: {name}}
  seasons       {id: {competition, name}}               e.g. "2021-22"
  squads        {season_id: {team_id: [{player, number, positions?}]}}
  matches       {match_id: {season, round, date, venue, result,
                            team_a, team_b, home,
                            lineups: {team_a: [{player, number, position, starter}]},
                            video, data: [files]}}

A player's shirt number belongs to a squad (team + season), not to the
player, so a transfer or a new number next season doesn't rewrite history.
A match links a video (match_id is the ingested video's id) to a season, two
teams and the data files that go with it.

`import-match` fills the library from a match that already has
config/<id>_match.json and config/<id>_reference.json. Running it again
updates entries in place and doesn't duplicate them.

Usage:
    python tools/library.py import-match --match-id <id>
    python tools/library.py show
"""
import argparse
import json
import os
import re
import tempfile
import threading
import unicodedata
from pathlib import Path

KINDS = ("teams", "players", "competitions", "seasons")
FIELDS = {
    "teams": ("name", "short", "color", "kit"),
    "players": ("name", "positions", "notes"),
    "competitions": ("name",),
    "seasons": ("competition", "name"),
}
MATCH_FIELDS = ("season", "round", "date", "venue", "result", "team_a", "team_b", "home", "lineups", "video", "data")
POSITIONS = ("GK", "DEF", "MID", "FWD")
LOCK = threading.Lock()  # one writer at a time inside the app


class LibraryError(ValueError):
    pass


def library_path(config_dir="config"):
    return Path(config_dir) / "library.json"


def empty():
    return {"version": 1, "teams": {}, "players": {}, "competitions": {}, "seasons": {}, "squads": {}, "matches": {}}


def load(path):
    path = Path(path)
    if not path.exists():
        return empty()
    try:
        lib = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise LibraryError(f"{path} is not valid JSON: {e}") from None
    base = empty()
    base.update({k: v for k, v in lib.items() if k in base})
    return base


def save(path, lib):
    validate(lib)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".library.", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(lib, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(tmp, 0o644)  # mkstemp makes it owner-only; keep it readable like the other config files
    os.replace(tmp, path)


def slug(text):
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "x"


def new_id(existing, name):
    base = slug(name)
    i, out = 2, base
    while out in existing:
        out, i = f"{base}-{i}", i + 1
    return out


def _clean(kind, data):
    out = {}
    for k in FIELDS[kind]:
        if k not in data or data[k] is None:
            continue
        v = data[k]
        if k == "positions":
            v = [p for p in (v if isinstance(v, list) else [v]) if p]
            bad = [p for p in v if p not in POSITIONS]
            if bad:
                raise LibraryError(f"Unknown position {bad[0]!r}: use {', '.join(POSITIONS)}")
        elif isinstance(v, str):
            v = v.strip()
        out[k] = v
    if kind != "players" and not out.get("name"):
        raise LibraryError(f"A {kind[:-1]} needs a name")
    if kind == "players" and not out.get("name"):
        raise LibraryError("A player needs a name")
    if kind == "teams" and out.get("color") and not re.fullmatch(r"#[0-9A-Fa-f]{6}", out["color"]):
        raise LibraryError("Team colour must look like #6CABDD")
    return out


def validate(lib):
    for sid, s in lib["seasons"].items():
        if s.get("competition") not in lib["competitions"]:
            raise LibraryError(f"Season {sid} points at unknown competition {s.get('competition')!r}")
    for sid, teams in lib["squads"].items():
        if sid not in lib["seasons"]:
            raise LibraryError(f"Squad for unknown season {sid!r}")
        for tid, entries in teams.items():
            if tid not in lib["teams"]:
                raise LibraryError(f"Squad for unknown team {tid!r}")
            seen = {}
            for e in entries:
                if e.get("player") not in lib["players"]:
                    raise LibraryError(f"Squad {tid} {sid}: unknown player {e.get('player')!r}")
                n = e.get("number")
                if n is not None:
                    if not (isinstance(n, int) and 1 <= n <= 99):
                        raise LibraryError(f"Shirt numbers are 1-99 (got {n!r})")
                    if n in seen:
                        name = lib["players"][e["player"]]["name"]
                        raise LibraryError(f"No. {n} is used twice in {lib['teams'][tid]['name']} "
                                           f"{lib['seasons'][sid]['name']} ({seen[n]} and {name})")
                    seen[n] = lib["players"][e["player"]]["name"]
    for mid, m in lib["matches"].items():
        if m.get("season") and m["season"] not in lib["seasons"]:
            raise LibraryError(f"Match {mid}: unknown season {m['season']!r}")
        for side in ("team_a", "team_b"):
            if m.get(side) and m[side] not in lib["teams"]:
                raise LibraryError(f"Match {mid}: unknown team {m[side]!r}")
        if m.get("team_a") and m.get("team_a") == m.get("team_b"):
            raise LibraryError(f"Match {mid}: a team can't play itself")
        if m.get("home") not in (None, "team_a", "team_b"):
            raise LibraryError("home must be team_a or team_b")
        for side, lu in (m.get("lineups") or {}).items():
            for e in lu:
                if e.get("player") not in lib["players"]:
                    raise LibraryError(f"Match {mid} {side} lineup: unknown player {e.get('player')!r}")


def upsert(lib, kind, data, item_id=None):
    """Create (item_id None) or update an entry; returns its id."""
    if kind not in KINDS:
        raise LibraryError(f"Unknown kind {kind!r}")
    clean = _clean(kind, data)
    if kind == "seasons" and clean.get("competition") not in lib["competitions"]:
        raise LibraryError("Pick the competition this season belongs to")
    if item_id is None:
        name = clean["name"] if kind != "seasons" else f"{clean['competition']}-{clean['name']}"
        item_id = new_id(lib[kind], name)
        lib[kind][item_id] = clean
    else:
        if item_id not in lib[kind]:
            raise LibraryError(f"No {kind[:-1]} {item_id!r}")
        lib[kind][item_id] = {**lib[kind][item_id], **clean}
    return item_id


def references(lib, kind, item_id):
    """Human-readable places that use this entry."""
    out = []
    if kind == "competitions":
        out += [f"season {s['name']}" for s in lib["seasons"].values() if s.get("competition") == item_id]
    if kind == "seasons":
        if lib["squads"].get(item_id):
            out.append("its squads")
        out += [f"match {mid}" for mid, m in lib["matches"].items() if m.get("season") == item_id]
    if kind == "teams":
        out += [f"squad {sid}" for sid, t in lib["squads"].items() if t.get(item_id)]
        out += [f"match {mid}" for mid, m in lib["matches"].items() if item_id in (m.get("team_a"), m.get("team_b"))]
    if kind == "players":
        out += [f"squad {tid} {sid}" for sid, ts in lib["squads"].items() for tid, es in ts.items()
                if any(e.get("player") == item_id for e in es)]
        out += [f"match {mid}" for mid, m in lib["matches"].items()
                if any(e.get("player") == item_id for lu in (m.get("lineups") or {}).values() for e in lu)]
    return out


def delete(lib, kind, item_id):
    if item_id not in lib.get(kind, {}):
        raise LibraryError(f"No {kind[:-1]} {item_id!r}")
    used = references(lib, kind, item_id)
    if used:
        raise LibraryError(f"Still used by {', '.join(used[:3])}{' and more' if len(used) > 3 else ''}: remove those first")
    del lib[kind][item_id]


def find_player(lib, name, team=None):
    """A player id by name (case/accent-insensitive), preferring one already in `team`'s squads."""
    key = slug(name)
    hits = [pid for pid, p in lib["players"].items() if slug(p.get("name", "")) == key]
    if team and len(hits) > 1:
        mine = {e["player"] for ts in lib["squads"].values() for e in ts.get(team, [])}
        hits = [h for h in hits if h in mine] or hits
    return hits[0] if hits else None


def set_squad(lib, season, team, entries):
    """Replace a squad. Entries: {player} or {name} (new players are created), number, positions."""
    if season not in lib["seasons"]:
        raise LibraryError(f"No season {season!r}")
    if team not in lib["teams"]:
        raise LibraryError(f"No team {team!r}")
    out = []
    for e in entries:
        pid = e.get("player")
        if pid and pid not in lib["players"]:
            raise LibraryError(f"No player {pid!r}")
        name = (e.get("name") or "").strip()
        if not pid:
            if not name:
                continue  # an empty row
            pid = find_player(lib, name, team) or upsert(lib, "players", {"name": name, "positions": e.get("positions") or []})
        elif name and name != lib["players"][pid]["name"]:
            lib["players"][pid]["name"] = name
        if e.get("positions") is not None:
            lib["players"][pid]["positions"] = _clean("players", {"name": "x", "positions": e["positions"]})["positions"]
        n = e.get("number")
        n = int(n) if n not in (None, "") else None
        out.append({"player": pid, "number": n})
    lib["squads"].setdefault(season, {})[team] = out
    validate(lib)
    return out


def link_match(lib, match_id, data):
    m = dict(lib["matches"].get(match_id, {}))
    for k in MATCH_FIELDS:
        if k in data:
            v = data[k]
            if v in ("", None):
                m.pop(k, None)
            else:
                m[k] = v.strip() if isinstance(v, str) else v
    lib["matches"][match_id] = m
    validate(lib)
    return m


def match_lineup(lib, match_id, side):
    """[{number, name, gk, sub}] for the Identify picker: the match lineup, then the rest of the squad as subs."""
    m = lib["matches"].get(match_id) or {}
    team, season = m.get(side), m.get("season")
    out, seen = [], set()
    for e in (m.get("lineups") or {}).get(side, []):
        p = lib["players"].get(e["player"], {})
        out.append({"number": e.get("number"), "name": p.get("name"), "gk": e.get("position") == "GK",
                    "sub": not e.get("starter", True), "player": e["player"]})
        seen.add(e["player"])
    for e in lib["squads"].get(season, {}).get(team, []):
        if e["player"] in seen or e.get("number") is None:
            continue
        p = lib["players"].get(e["player"], {})
        out.append({"number": e["number"], "name": p.get("name"), "gk": "GK" in p.get("positions", []),
                    "sub": True, "player": e["player"]})
    return out


COMP_RE = re.compile(r"^\s*(.*?)\s+(\d{4}(?:[-/]\d{2,4})?)\s*(?:,\s*(.*))?$")


def split_competition(text):
    """'Premier League 2021-22, Gameweek 26' -> ('Premier League', '2021-22', 'Gameweek 26')."""
    m = COMP_RE.match(text or "")
    if not m:
        return (text or "").strip() or None, None, None
    return m.group(1).strip(), m.group(2).replace("/", "-"), (m.group(3) or "").strip() or None


def _line_position(i, n_lines):
    if i == 0:
        return "GK"
    if i == 1:
        return "DEF"
    return "FWD" if i == n_lines - 1 else "MID"


def import_match(lib, match_id, cfg, reference=None, video=None, data_files=()):
    """Fill the library from a match's config (+ reference lineups). Returns a summary dict."""
    comp, season_name, rnd = split_competition(cfg.get("competition"))
    added = {"teams": 0, "players": 0, "competitions": 0, "seasons": 0}
    season = None
    if comp:
        cid = next((k for k, c in lib["competitions"].items() if slug(c["name"]) == slug(comp)), None)
        if cid is None:
            cid = upsert(lib, "competitions", {"name": comp})
            added["competitions"] += 1
        if season_name:
            season = next((k for k, s in lib["seasons"].items()
                           if s["competition"] == cid and s["name"] == season_name), None)
            if season is None:
                season = upsert(lib, "seasons", {"competition": cid, "name": season_name})
                added["seasons"] += 1
    teams = cfg.get("teams") or {}
    colors = cfg.get("colors") or {}
    side_team = {}
    for side in ("team_a", "team_b"):
        t = teams.get(side) or {}
        name = t.get("name") or cfg.get(f"{side}_name")
        if not name:
            continue
        data = {"name": name, "short": t.get("short"), "kit": t.get("kit"), "color": colors.get(side)}
        tid = next((k for k, v in lib["teams"].items() if slug(v["name"]) == slug(name)), None)
        if tid is None:
            tid = upsert(lib, "teams", data)
            added["teams"] += 1
        else:  # fill gaps, never overwrite hand edits
            for k, v in _clean("teams", data).items():
                lib["teams"][tid].setdefault(k, v)
        side_team[side] = tid
    lineups = {}
    for side, tid in side_team.items():
        lu = ((reference or {}).get("lineups") or {}).get(side) or {}
        lines = lu.get("lines") or []
        rows = [(p, _line_position(i, len(lines)), True) for i, line in enumerate(lines) for p in line]
        rows += [(p, None, False) for p in (lu.get("subs") or lu.get("substitutes") or [])]
        entries = []
        squad = (lib["squads"].get(season, {}).get(tid, []) if season else [])
        for p, pos, starter in rows:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            pid = find_player(lib, p["name"], tid)
            if pid is None:
                pid = upsert(lib, "players", {"name": p["name"], "positions": [pos] if pos else []})
                added["players"] += 1
            elif pos and not lib["players"][pid].get("positions"):
                lib["players"][pid]["positions"] = [pos]
            num = p.get("number")
            entries.append({"player": pid, "number": num, "position": pos, "starter": starter})
            if season and not any(e["player"] == pid for e in squad):
                clash = num is not None and any(e.get("number") == num for e in squad)
                squad.append({"player": pid, "number": None if clash else num})
        if season:
            lib["squads"].setdefault(season, {})[tid] = squad
        if entries:
            lineups[side] = entries
    m = lib["matches"].get(match_id, {})
    result = cfg.get("result") or ""
    home = None
    if side_team and result:
        a = cfg.get("team_a_name") or ""
        b = cfg.get("team_b_name") or ""
        if a and result.startswith(a):
            home = "team_a"
        elif b and result.startswith(b):
            home = "team_b"
    new = {"season": season, "round": rnd, "date": cfg.get("date"), "venue": cfg.get("venue"), "result": cfg.get("result"),
           "team_a": side_team.get("team_a"), "team_b": side_team.get("team_b"), "home": home,
           "lineups": lineups or None, "video": video, "data": list(data_files) or None}
    lib["matches"][match_id] = {**{k: v for k, v in new.items() if v is not None}, **{k: v for k, v in m.items() if k not in new or new[k] is None}}
    validate(lib)
    return {"added": added, "season": season, "teams": side_team, "lineup_players": sum(len(v) for v in lineups.values())}


def match_data_files(match_id, config_dir="config"):
    names = [f"{match_id}_match.json", f"{match_id}_reference.json", f"{match_id}_calibration.json",
             f"{match_id}_identity_anchors.json", f"{match_id}_instances.json"]
    return [f"config/{n}" for n in names if (Path(config_dir) / n).exists()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    im = sub.add_parser("import-match", help="add a match's teams, lineups and season to the library")
    im.add_argument("--match-id", required=True)
    sh = sub.add_parser("show", help="print the library")
    for p in (im, sh):
        p.add_argument("--config-dir", default="config")
        p.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()
    path = library_path(a.config_dir)
    lib = load(path)
    if a.cmd == "show":
        for sid, s in lib["seasons"].items():
            print(f"{lib['competitions'][s['competition']]['name']} {s['name']}")
            for tid, es in lib["squads"].get(sid, {}).items():
                print(f"  {lib['teams'][tid]['name']}: " + ", ".join(
                    f"{e['number'] or '-'} {lib['players'][e['player']]['name']}" for e in es))
        for mid, m in lib["matches"].items():
            print(f"match {mid}: {m.get('date', '')} {m.get('result', '')}")
        return
    from _common import load_meta, match_config, read_json, reference_path
    cfg = match_config(a.match_id, a.tmp_dir, a.config_dir)
    if not cfg:
        raise SystemExit(f"No config/{a.match_id}_match.json")
    ref_p = reference_path(a.match_id, a.config_dir)
    ref = read_json(ref_p) if ref_p.exists() else None
    try:
        video = load_meta(a.match_id, a.tmp_dir).get("video")
    except (OSError, ValueError):
        video = None
    s = import_match(lib, a.match_id, cfg, ref, video, match_data_files(a.match_id, a.config_dir))
    save(path, lib)
    print(f"Imported {a.match_id}: added {s['added']}, {s['lineup_players']} lineup players -> {path}")


if __name__ == "__main__":
    main()
