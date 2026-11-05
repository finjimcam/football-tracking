"""Library tab (issue #16): wraps tools/library.py for the API.

Every write takes the library lock, loads config/library.json, applies one
change, checks the whole library and saves it atomically. A rejected change
leaves the file as it was.
"""
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import library as lib_tool  # noqa: E402
from library import LOCK, LibraryError, library_path, load, save  # noqa: E402,F401


def _path(root):
    return library_path(Path(root) / "config")


def _write(root, fn):
    with LOCK:
        lib = load(_path(root))
        out = fn(lib)
        save(_path(root), lib)
        return out


def summary(root, matches=()):
    lib = load(_path(root))
    linked = set(lib["matches"])
    return {
        "library": lib,
        "positions": list(lib_tool.POSITIONS),
        "videos": [{"match_id": m["match_id"], "title": f"{m['team_a_name']} v {m['team_b_name']}",
                    "video_name": m.get("video_name"), "kickoff": m.get("kickoff"),
                    "has_config": bool(m["artifacts"].get("match_config", {}).get("exists")),
                    "linked": m["match_id"] in linked} for m in matches],
    }


def upsert(root, kind, data, item_id=None):
    item_id = _write(root, lambda lib: lib_tool.upsert(lib, kind, data, item_id))
    return {"id": item_id, **summary(root)}


def delete(root, kind, item_id):
    _write(root, lambda lib: lib_tool.delete(lib, kind, item_id))
    return summary(root)


def set_squad(root, season, team, entries):
    _write(root, lambda lib: lib_tool.set_squad(lib, season, team, entries))
    return summary(root)


def link_match(root, match_id, data):
    _write(root, lambda lib: lib_tool.link_match(lib, match_id, data))
    return summary(root)


def import_match(root, match_id):
    sys_root = Path(root)
    from _common import match_config, read_json, reference_path

    cfg = match_config(match_id, sys_root / ".tmp", sys_root / "config")
    if not cfg:
        raise LibraryError("This match has no config/<id>_match.json to import from")
    ref_p = reference_path(match_id, sys_root / "config")
    ref = read_json(ref_p) if ref_p.exists() else None
    meta_p = sys_root / ".tmp" / match_id / "meta.json"
    video = read_json(meta_p).get("video") if meta_p.exists() else None
    files = lib_tool.match_data_files(match_id, sys_root / "config")
    res = _write(root, lambda lib: lib_tool.import_match(lib, match_id, cfg, ref, video, files))
    return {"imported": res, **summary(root)}
