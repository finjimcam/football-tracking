"""Hand fixes to the tracking (issue #42, the Stopframe Tracking tab).

The tracker's output (.tmp/<id>/tracks.parquet) is never rewritten. Instead the
fixes live in config/<match_id>_track_edits.json, and every tool that reads
tracks through _common.read_tracks()/load_tracks() gets them applied on top.
Undoing a fix is deleting it from the list.

Edit kinds (all frames are video frames; ids are tracks.parquet track_ids):
  swap        {frame, a, b[, to]}       ids a and b trade places from `frame` on
                                        (to the end, or to `to`): the fix for an
                                        identity swap at a collision
  assign      {frame, tid, to[, tracklet]}
                                        the box `tid` is player `to` from `frame`
                                        to the end of that tracklet; whatever `to`
                                        had on those frames becomes `tid` (they
                                        trade), so a player is never doubled
  box         {frame, tid, x1, y1, x2, y2, pitch_x, pitch_y}
                                        a hand-drawn box for `tid`. Frames between
                                        it and the player's nearest box (hand or
                                        tracked, up to INTERP_S away, with nothing
                                        in between) are interpolated; with no
                                        neighbour it's held HOLD_S each way
  split       {frame, tid[, tracklet]}  from `frame` to the end of that tracklet
                                        the box isn't `tid` any more (new id)
  delete      {frame, tid[, to]}        drop tid's box on `frame` (..`to`)
  not_player  {frame, tid[, tracklet]}  that whole tracklet is a referee / fan /
                                        coach: new id, kit "official"
  ok          {problem}                 a flagged problem you checked: no change

Edits apply in list order. Rows added by hand get source="hand" (the box you
drew) or "hand_fill" (interpolated); tracked rows are source="auto".

The file records a fingerprint of the tracks it was made against. If
track_identities.py is re-run the ids can move, so a mismatched file is NOT
applied (a warning is printed) until the fixes are checked again.

    python tools/track_edits.py --match-id <id>          # list the fixes + status
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("swap", "assign", "box", "split", "delete", "not_player", "ok")
NEW_ID_BASE = 900000  # ids for split / not-player pieces: NEW_ID_BASE + edit id
INTERP_S = 5.0
HOLD_S = 0.5
VERSION = 1


class EditError(ValueError):
    pass


def edits_path(match_id, config_dir="config"):
    return Path(config_dir) / f"{match_id}_track_edits.json"


def empty(match_id=None):
    return {"version": VERSION, "match_id": match_id, "base": None, "edits": []}


def load_edits(path):
    path = Path(path)
    if not path.exists():
        return empty()
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("edits", [])
    data.setdefault("base", None)
    return data


def save_edits(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def fingerprint(tracks):
    """Order-independent summary of the PLAYER rows (ball re-tracking doesn't
    change it; a re-run of track_identities.py usually does)."""
    p = tracks[tracks["class"] == "player"]
    return {"rows": int(len(p)), "id_sum": int(p["track_id"].astype("int64").sum()),
            "frame_sum": int(p["frame"].astype("int64").sum())}


def _int(e, k, required=True):
    v = e.get(k)
    if v is None:
        if required:
            raise EditError(f"'{k}' is required")
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        raise EditError(f"'{k}' must be a whole number") from None


def _num(e, k):
    try:
        v = float(e[k])
    except (KeyError, TypeError, ValueError):
        raise EditError(f"'{k}' must be a number") from None
    if not np.isfinite(v):
        raise EditError(f"'{k}' must be a number")
    return v


def clean_edit(e):
    """Validated copy of one edit (only known keys)."""
    if not isinstance(e, dict):
        raise EditError("An edit must be an object")
    kind = e.get("kind")
    if kind not in KINDS:
        raise EditError(f"Unknown edit kind '{kind}'")
    out = {"kind": kind}
    if kind == "ok":
        p = e.get("problem")
        if not isinstance(p, str) or not p or len(p) > 80:
            raise EditError("'problem' is required")
        out["problem"] = p
        if e.get("frame") is not None:
            out["frame"] = _int(e, "frame")
        return out
    out["frame"] = _int(e, "frame")
    if kind == "swap":
        out["a"], out["b"] = _int(e, "a"), _int(e, "b")
        if out["a"] == out["b"]:
            raise EditError("Pick two different players to swap")
        if e.get("to") is not None:
            out["to"] = _int(e, "to")
    elif kind == "assign":
        out["tid"], out["to"] = _int(e, "tid"), _int(e, "to")
        if out["tid"] == out["to"]:
            raise EditError("That box is already this player")
    else:
        out["tid"] = _int(e, "tid")
    if kind in ("assign", "split", "not_player") and e.get("tracklet") is not None:
        out["tracklet"] = _int(e, "tracklet")
    if kind == "delete" and e.get("to") is not None:
        out["to"] = _int(e, "to")
    if kind == "box":
        for k in ("x1", "y1", "x2", "y2", "pitch_x", "pitch_y"):
            out[k] = round(_num(e, k), 3)
        if out["x2"] - out["x1"] < 2 or out["y2"] - out["y1"] < 2:
            raise EditError("The box is too small")
    for k in ("problem", "note"):
        if isinstance(e.get(k), str) and e[k]:
            out[k] = e[k][:200]
    return out


# --------------------------------------------------------------------------
# applying
# --------------------------------------------------------------------------
def _players(df):
    return (df["class"] == "player").to_numpy()


def _tracklet_mask(df, tid, frame, tracklet):
    """Rows of `tid` in the tracklet it's on at `frame` (or `tracklet` if
    given). Without a tracklet column: all of tid's rows."""
    m = _players(df) & (df["track_id"].to_numpy() == tid)
    if "tracklet" not in df.columns:
        return m
    tl = df["tracklet"].to_numpy()
    if tracklet is None:
        here = m & (df["frame"].to_numpy() == frame)
        if not here.any():
            return m & False
        tracklet = tl[here][0]
    if tracklet is None or (isinstance(tracklet, float) and np.isnan(tracklet)):
        return m & np.isnan(tl.astype(float))
    return m & (tl == tracklet)


def _swap(df, e):
    fr = df["frame"].to_numpy()
    m = _players(df) & (fr >= e["frame"])
    if e.get("to") is not None:
        m &= fr <= e["to"]
    ids = df["track_id"].to_numpy()
    ma, mb = m & (ids == e["a"]), m & (ids == e["b"])
    df.loc[ma, "track_id"] = e["b"]
    df.loc[mb, "track_id"] = e["a"]
    return df


def _assign(df, e):
    fr = df["frame"].to_numpy()
    src = _tracklet_mask(df, e["tid"], e["frame"], e.get("tracklet")) & (fr >= e["frame"])
    if not src.any():
        return df
    lo, hi = fr[src].min(), fr[src].max()
    dst = _players(df) & (df["track_id"].to_numpy() == e["to"]) & (fr >= lo) & (fr <= hi) & ~src
    df.loc[src, "track_id"] = e["to"]
    df.loc[dst, "track_id"] = e["tid"]
    return df


def _split(df, e, new_id):
    fr = df["frame"].to_numpy()
    m = _tracklet_mask(df, e["tid"], e["frame"], e.get("tracklet")) & (fr >= e["frame"])
    df.loc[m, "track_id"] = new_id
    return df


def _not_player(df, e, new_id):
    m = _tracklet_mask(df, e["tid"], e["frame"], e.get("tracklet"))
    df.loc[m, "track_id"] = new_id
    if "kit" in df.columns:
        df.loc[m, "kit"] = "official"
    return df


def _delete(df, e):
    fr = df["frame"].to_numpy()
    to = e.get("to") if e.get("to") is not None else e["frame"]
    m = _players(df) & (df["track_id"].to_numpy() == e["tid"]) & (fr >= e["frame"]) & (fr <= to)
    return df[~m]


def _box_row(df, e, fps):
    tid = e["tid"]
    ids = df["track_id"].to_numpy()
    mine = _players(df) & (ids == tid)
    kit = None
    if "kit" in df.columns and mine.any():
        kit = df.loc[mine, "kit"].mode().iloc[0]
    x1, y1, x2, y2 = e["x1"], e["y1"], e["x2"], e["y2"]
    row = {"frame": e["frame"], "t_s": e["frame"] / fps, "track_id": tid, "class": "player",
           "x1": x1, "y1": y1, "x2": x2, "y2": y2, "foot_x": (x1 + x2) / 2, "foot_y": y2,
           "conf": 1.0, "pitch_x": e["pitch_x"], "pitch_y": e["pitch_y"], "tracklet": np.nan,
           "kit": kit, "source": "hand"}
    # the hand box replaces whatever tid had on that frame
    drop = mine & (df["frame"].to_numpy() == e["frame"])
    return df[~drop], row


INTERP_COLS = ("x1", "y1", "x2", "y2", "pitch_x", "pitch_y")


def _fill_hand(df, fps):
    """Interpolate between each hand box and tid's nearest boxes (hand or
    tracked) when nothing lies between them and they're <= INTERP_S apart;
    hold a hand box HOLD_S on a side with no such neighbour."""
    hand = df[df["source"] == "hand"]
    if hand.empty:
        return df
    max_gap, hold = int(round(INTERP_S * fps)), int(round(HOLD_S * fps))
    new = []
    for tid, hrows in hand.groupby("track_id"):
        mine = df[_players(df) & (df["track_id"].to_numpy() == tid)].sort_values("frame")
        frames = mine["frame"].to_numpy()
        vals = mine[list(INTERP_COLS)].to_numpy(float)
        kit = hrows["kit"].iloc[0] if "kit" in hrows.columns else None
        made = set()
        for hf in hrows["frame"].to_numpy():
            i = int(np.searchsorted(frames, hf))
            for side in (-1, 1):
                j = i + side
                if 0 <= j < len(frames) and abs(frames[j] - hf) <= max_gap:
                    a, b = sorted((i, j))
                    fa, fb = frames[a], frames[b]
                    for f in range(fa + 1, fb):
                        if f in made:
                            continue
                        w = (f - fa) / (fb - fa)
                        new.append((f, tid, *(vals[a] * (1 - w) + vals[b] * w), kit))
                        made.add(f)
                else:
                    rng = range(hf - hold, hf) if side < 0 else range(hf + 1, hf + hold + 1)
                    for f in rng:
                        if f in made:
                            continue
                        new.append((f, tid, *vals[i], kit))
                        made.add(f)
    if not new:
        return df
    add = pd.DataFrame(new, columns=["frame", "track_id", *INTERP_COLS, "kit"])
    add["frame"] = add["frame"].astype("int64")
    add["t_s"] = add["frame"] / fps
    add["class"] = "player"
    add["foot_x"] = (add["x1"] + add["x2"]) / 2
    add["foot_y"] = add["y2"]
    add["conf"] = 0.5
    add["tracklet"] = np.nan
    add["source"] = "hand_fill"
    # a filled frame never overrides a real box of that player
    mine = df[_players(df) & df["track_id"].isin(add["track_id"].unique()).to_numpy()]
    have = set(zip(mine["frame"].tolist(), mine["track_id"].tolist()))
    keep = [(f, t) not in have for f, t in zip(add["frame"].tolist(), add["track_id"].tolist())]
    add = add[keep]
    return _concat(df, add)


def _concat(df, add):
    add = add[[c for c in add.columns if c in df.columns]]
    out = pd.concat([df, add], ignore_index=True)
    for c in ("frame", "track_id"):
        out[c] = out[c].astype("int64")
    return out


def apply_edits(tracks, edits, fps=25.0):
    """tracks (any subset of tracks.parquet columns that includes frame,
    track_id and class) with `edits` applied in order. Returns a new frame,
    sorted by frame, with a `source` column."""
    df = tracks.copy()
    if "source" not in df.columns:
        df["source"] = "auto"
    for c in ("frame", "track_id"):
        df[c] = df[c].astype("int64")
    for i, raw in enumerate(edits):
        e = clean_edit(raw)
        eid = int(raw.get("id", i + 1))
        k = e["kind"]
        if k == "swap":
            df = _swap(df, e)
        elif k == "assign":
            df = _assign(df, e)
        elif k == "split":
            df = _split(df, e, NEW_ID_BASE + eid)
        elif k == "not_player":
            df = _not_player(df, e, NEW_ID_BASE + eid)
        elif k == "delete":
            df = _delete(df, e)
        elif k == "box":
            df, row = _box_row(df, e, fps)
            df = _concat(df, pd.DataFrame([row]))
    df = _fill_hand(df, fps)
    return df.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def apply_to(tracks, match_id, config_dir="config", fps=25.0, quiet=False):
    """tracks with this match's saved edits applied (unchanged when there are
    none, or when they were made against different tracks)."""
    path = edits_path(match_id, config_dir)
    if not path.exists():
        return tracks
    data = load_edits(path)
    edits = [e for e in data["edits"] if e.get("kind") != "ok"]
    if not edits:
        return tracks
    base = data.get("base")
    if base and base != fingerprint(tracks):
        if not quiet:
            print(f"WARNING: {path} was made against different tracks (tracking re-run?): "
                  "its fixes are NOT applied. Re-check them in the Tracking tab.", file=sys.stderr)
        return tracks
    return apply_edits(tracks, edits, fps)


def add_edit(data, edit, base=None):
    """Append a validated edit (new id + timestamp) to `data` in place."""
    e = clean_edit(edit)
    e["id"] = max((int(x.get("id", 0)) for x in data["edits"]), default=0) + 1
    e["at"] = int(time.time())
    if data.get("base") is None and base is not None:
        data["base"] = base
    data["edits"].append(e)
    return e


def remove_edit(data, edit_id):
    n = len(data["edits"])
    data["edits"] = [e for e in data["edits"] if int(e.get("id", -1)) != int(edit_id)]
    if len(data["edits"]) == n:
        raise EditError(f"No fix with id {edit_id}")
    if not data["edits"]:
        data["base"] = None
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()
    path = edits_path(a.match_id, a.config_dir)
    data = load_edits(path)
    print(f"{path}: {len(data['edits'])} fix(es)")
    for e in data["edits"]:
        rest = {k: v for k, v in e.items() if k not in ("id", "kind", "at")}
        print(f"  #{e.get('id')} {e['kind']:<10} {json.dumps(rest)}")
    tp = Path(a.tmp_dir) / a.match_id / "tracks.parquet"
    if tp.exists() and data["edits"]:
        cur = fingerprint(pd.read_parquet(tp, columns=["frame", "track_id", "class"]))
        ok = data.get("base") in (None, cur)
        print("matches the current tracks: applied" if ok else
              "made against DIFFERENT tracks (tracking re-run?): not applied until re-checked")


if __name__ == "__main__":
    main()
