"""Manual identity anchors (issue #16): "the player in this box, at this frame, is No. 17".

The analyst makes anchors in the Stopframe Lines tab (Identify mode): click a
tracked player, or drag a box round one the tracker left unidentified, and
pick a name from the lineup. They're stored in
config/<match_id>_identity_anchors.json:

    {"anchors": [{"id": "...", "team": "team_a", "number": 17, "name": "De Bruyne",
                  "frame": 33150, "box": [x1, y1, x2, y2], "tracklet": 1234,
                  "track_id": 7, "created": 1730000000.0}]}

tools/track_identities.py applies them every time it runs:
  1. resolve: each anchor finds its detection again, as the box with the best
     IoU at that frame (falling back to the stored tracklet id), so anchors
     survive a tracklet rebuild;
  2. names: each identity chain is named after the player most of its
     anchors point at, overriding the role-inferred guess. The chain that had
     that name takes the displaced name, so names stay one per player;
  3. moves: an anchored tracklet that isn't in its player's chain (still
     unidentified, or linked into someone else's chain) is moved into it, and
     any other tracklet of that chain overlapping it in time is dropped from
     the chain (it can't be the same player twice at once).
Every stage after it (possession, physical, positions, verification, report)
then uses the corrected identities for the whole match.
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

TEAM_BASE = {"team_a": 0, "team_b": 100}  # GK = base + 1, outfield base + 2..11


def anchors_path(match_id, config_dir="config"):
    return Path(config_dir) / f"{match_id}_identity_anchors.json"


def load_anchors(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [a for a in d.get("anchors", []) if isinstance(a, dict)]


def save_anchors(path, anchors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"anchors": anchors}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_box(rows, box, min_iou=0.3):
    """Index of the row (x1..y2 columns) best matching box, or None."""
    best, bi = None, min_iou
    for i, r in enumerate(rows[["x1", "y1", "x2", "y2"]].to_numpy()):
        v = iou(r, box)
        if v >= bi:
            best, bi = i, v
    return best


def resolve(anchors, p, known_tracklets=()):
    """Attach the current tracklet id to each anchor: [(anchor, tracklet|None)].
    p: person detections with frame, tracklet, x1..y2."""
    out = []
    by_frame = {f: g for f, g in p[p["frame"].isin({int(a["frame"]) for a in anchors})].groupby("frame")}
    known = set(int(t) for t in known_tracklets)
    for a in anchors:
        g = by_frame.get(int(a["frame"]))
        tl = None
        if g is not None and a.get("box"):
            i = best_box(g, a["box"])
            if i is not None:
                v = int(g["tracklet"].iloc[i])
                tl = v if v >= 0 else None
        if tl is None and a.get("tracklet") is not None and int(a["tracklet"]) in known:
            tl = int(a["tracklet"])
        out.append((a, tl))
    return out


def apply(tl, names, resolved):
    """Apply resolved anchors to the identity table and the name table in place.

    tl: one row per tracklet with tracklet, identity, start, end (+ link).
    names: {identity: {"name", "number", ...}} as from infer_names.
    Returns a report dict."""
    ident = dict(zip(tl["tracklet"].astype(int), tl["identity"].astype(int)))
    span = dict(zip(tl["tracklet"].astype(int), zip(tl["start"].astype(int), tl["end"].astype(int))))
    rep = {"anchors": len(resolved), "unresolved": 0, "named_chains": 0, "moved": 0, "evicted": 0,
           "unknown_player": 0, "conflicts": []}
    good = []
    for a, t in resolved:
        if t is None or t not in ident:
            rep["unresolved"] += 1
            continue
        if a.get("team") not in TEAM_BASE or a.get("number") is None:
            rep["unknown_player"] += 1
            continue
        good.append((a, t))

    # 2. chain names by majority vote of the anchors inside each chain
    for team, base in TEAM_BASE.items():
        chains = range(base + 1, base + 12)
        num_of = {c: (names.get(c) or {}).get("number") for c in chains}
        votes = Counter((ident[t], int(a["number"])) for a, t in good
                        if a["team"] == team and ident[t] in chains)
        taken_c, taken_n = set(), set()
        for (c, n), k in votes.most_common():
            if c in taken_c or n in taken_n:
                if c not in taken_c or n not in taken_n:
                    rep["conflicts"].append({"team": team, "chain": c, "number": n, "anchors": k})
                continue
            taken_c.add(c)
            taken_n.add(n)
            old = next((x for x in chains if num_of.get(x) == n), None)
            if old is not None and old != c:
                a_info, b_info = dict(names.get(c) or {}), dict(names.get(old) or {})
                for key in ("name", "number"):
                    a_info[key], b_info[key] = b_info.get(key), a_info.get(key)
                b_info["name_source"] = b_info.get("name_source") or "displaced by a manual anchor"
                names[c], names[old] = a_info, b_info
                num_of[c], num_of[old] = n, b_info.get("number")
            elif old is None:  # a player the role inference didn't place (e.g. a substitute)
                info = dict(names.get(c) or {})
                person = next((a for a, t in good if a["team"] == team and int(a["number"]) == n), {})
                info.update(name=person.get("name"), number=n)
                names[c] = info
                num_of[c] = n
            names[c]["name_source"] = "manual (Lines tab anchors)"
            names[c]["anchors"] = k
            rep["named_chains"] += 1

    # 3. move anchored tracklets into their player's chain
    for team, base in TEAM_BASE.items():
        chains = range(base + 1, base + 12)
        chain_of_num = {(names.get(c) or {}).get("number"): c for c in chains}
        mine = [(a, t) for a, t in good if a["team"] == team]
        anchored = {}
        for a, t in mine:
            anchored.setdefault(int(a["number"]), set()).add(t)
        for n, ts in anchored.items():
            c = chain_of_num.get(n)
            if c is None:
                rep["unknown_player"] += len(ts)
                continue
            for t in ts:
                if ident[t] == c:
                    continue
                s, e = span[t]
                for u, cu in ident.items():
                    if cu == c and u not in ts and span[u][0] <= e and span[u][1] >= s:
                        ident[u] = -1
                        rep["evicted"] += 1
                ident[t] = c
                rep["moved"] += 1
    new = tl["tracklet"].astype(int).map(ident)
    changed = new.to_numpy() != tl["identity"].to_numpy()
    tl["identity"] = new.to_numpy()
    if "link" in tl.columns:
        tl.loc[changed & (new.to_numpy() >= 0), "link"] = "anchor"
        tl.loc[changed & (new.to_numpy() < 0), "link"] = "evicted_by_anchor"
    if "team" in tl.columns:
        tl.loc[changed & (new.to_numpy() >= 0), "team"] = np.where(new[changed & (new.to_numpy() >= 0)] >= 100,
                                                                    "team_b", "team_a")
    return rep
