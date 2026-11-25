"""Whole-match identities: track each half on its own, then combine them,
with substitutions.

track_identities.py assumes the same 11 v 11 for its whole window, and reads
each team's attacking direction from where its keeper stands, so a match
with more than one period (config/<id>_match.json "periods") is tracked one
period at a time and the halves are joined here.

    python tools/track_periods.py --match-id <id>                  # merge, identities, ball, combine
    python tools/track_periods.py --match-id <id> --stages combine # re-join only

Workspaces: .tmp/<id>/periods/h1, h2, ... (in "periods" order). Each is a
tmp root for the tools - h<n>/<id>/meta.json with the period's window - plus
h<n>/config/ (match, reference, calibration, anchors and calibration clips,
copied from --config-dir with the window set to the period).

Stages:
  merge       Fold a period's own detection / calibration runs into the
              match's: h<n>/<id>/det_*.parquet (detect_frames.py --tmp-dir
              <workspace> --start-frame/--end-frame, renamed det_<start>)
              replace those frames of .tmp/<id>/detections.parquet, and
              h<n>/<id>/frame_homographies.npz (calibrate_auto.py --tmp-dir
              <workspace> --calibration h<n>/config/...) replaces those frames
              of the match's npz.
  identities  track_identities.py per period, on the period's slice of the
              homographies (that slice is its window).
  ball        track_ball.py per period.
  combine     The live .tmp/<id>/ outputs over the whole match:
              - each later period's identities take the first period's
                player whose place in the team shape (median offset from the
                team centroid, facing the opponent's goal) is closest
                (one-to-one per team, before that team's first substitution),
                unless the analyst named them in that period;
              - "substitutions" in config/<id>_reference.json move the
                identity of the player going off to a new one for the
                player coming on, from the substitution's "frame" or, without
                one, the middle of its "minute" (via the period's
                kickoff_frame / clock_start_s);
              - unidentified fragments and tracklets get match-unique ids;
              - the Tracking tab's hand fixes are rebased onto the result.

Substitution ids: team_a 12, 13, ...; team_b 112, 113, ... (identities stay
below 1000; unlinked fragments are 10000 + tracklet).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_meta, match_config, match_dir, read_json, reference_path, write_json

TOOLS = Path(__file__).resolve().parent
TEAMS = ("team_a", "team_b")
GK = {"team_a": 1, "team_b": 101}
SUB_BASE = {"team_a": 12, "team_b": 112}
CONFIG_FILES = ("match", "reference", "calibration", "identity_anchors", "calibration_clips")


# --------------------------------------------------------------------------
# periods and the match clock
# --------------------------------------------------------------------------
def periods_of(cfg):
    """[(name, start_frame, end_frame, info)] in time order."""
    per = cfg.get("periods") or {}
    out = [(k, int(v["start_frame"]), int(v["end_frame"]), v) for k, v in per.items()
           if v.get("start_frame") is not None and v.get("end_frame") is not None]
    if not out:
        w = cfg.get("analysis_window") or {}
        out = [("match", int(w["start_frame"]), int(w["end_frame"]), {})]
    return sorted(out, key=lambda p: p[1])


def parse_minute(m):
    """'62', 62, '45+2' -> minutes as a float (45+2 -> 47)."""
    if isinstance(m, (int, float)):
        return float(m)
    s = str(m).replace("'", "").strip()
    if "+" in s:
        a, b = s.split("+", 1)
        return float(a) + float(b)
    return float(s)


def minute_to_frame(minute, cfg, fps):
    """Frame at the middle of match minute `minute` (the 62nd minute is
    61:00-62:00 on the clock -> 61:30). A period may give kickoff_frame and
    clock_start_s (e.g. 2700 for the second half); otherwise clock_sync's
    kickoff_frame with the clock running from 0."""
    t = (parse_minute(minute) - 0.5) * 60.0
    sync = cfg.get("clock_sync") or {}
    best = (int(sync.get("kickoff_frame", 0)), 0.0)
    for _, _, _, info in periods_of(cfg):
        if info.get("kickoff_frame") is None:
            continue
        c0 = float(info.get("clock_start_s", 0.0))
        if t >= c0 and c0 >= best[1]:
            best = (int(info["kickoff_frame"]), c0)
    k0, c0 = best
    return int(round(k0 + (t - c0) * fps))


def substitutions(ref, cfg, fps):
    """The reference file's substitutions with a frame each, in time order."""
    out = []
    for s in ref.get("substitutions") or []:
        f = s.get("frame")
        if f is None:
            f = minute_to_frame(s["minute"], cfg, fps)
        out.append({**s, "frame": int(f)})
    return sorted(out, key=lambda s: s["frame"])


# --------------------------------------------------------------------------
# workspaces
# --------------------------------------------------------------------------
def workspace(match_id, tmp_dir, k):
    return match_dir(match_id, tmp_dir) / "periods" / f"h{k + 1}"


def write_period_config(match_id, config_dir, ws, start, end, npz_path):
    cdir = ws / "config"
    cdir.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        src = Path(config_dir) / f"{match_id}_{name}.json"
        if not src.exists():
            continue
        data = read_json(src)
        if name == "match":
            data["analysis_window"] = {"start_frame": start, "end_frame": end}
        if name == "calibration":
            data["mode"] = "dynamic"
            data["frame_homographies"] = str(Path(npz_path).resolve())
        write_json(cdir / src.name, data)
    return cdir


def write_period_meta(match_id, tmp_dir, ws, start, end):
    meta = load_meta(match_id, tmp_dir)
    meta.setdefault("config", {})["analysis_window"] = {"start_frame": start, "end_frame": end}
    d = ws / match_id
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "meta.json", meta)
    return d


def npz_slice(npz, start, end):
    f = npz["frames"]
    m = (f >= start) & (f <= end)
    return {k: npz[k][m] if npz[k].shape[:1] == f.shape else npz[k] for k in npz.files}


def npz_merge(base, part):
    """base with part's frames replaced/added (dicts of arrays keyed like the npz)."""
    if base is None:
        return dict(part)
    f0, f1 = base["frames"], part["frames"]
    keep = ~np.isin(f0, f1)
    out = {}
    for k in base:
        if k in part and base[k].shape[:1] == f0.shape:
            out[k] = np.concatenate([base[k][keep], part[k]])
    order = np.argsort(out["frames"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def stage_merge(a, cfg, mdir):
    det_path, npz_path = mdir / "detections.parquet", mdir / "frame_homographies.npz"
    for k, (name, start, end, _) in enumerate(periods_of(cfg)):
        pd_dir = workspace(a.match_id, a.tmp_dir, k) / a.match_id
        parts = sorted(pd_dir.glob("det_*.parquet"))
        if parts:
            new = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            lo, hi = int(new["frame"].min()), int(new["frame"].max())
            old = pd.read_parquet(det_path) if det_path.exists() else new.iloc[:0]
            old = old[(old["frame"] < lo) | (old["frame"] > hi)]
            det = pd.concat([old, new], ignore_index=True).sort_values(["frame", "class"]).reset_index(drop=True)
            det.to_parquet(det_path, index=False)
            print(f"  {name}: {len(new)} detections for frames {lo}-{hi} merged into {det_path.name}")
        pz = pd_dir / "frame_homographies.npz"
        if pz.exists() and (not npz_path.exists() or pz.stat().st_mtime > npz_path.stat().st_mtime):
            part = dict(np.load(pz))
            base = dict(np.load(npz_path)) if npz_path.exists() else None
            merged = npz_merge(base, part)
            np.savez_compressed(npz_path, **merged)
            print(f"  {name}: homographies for frames {int(part['frames'][0])}-{int(part['frames'][-1])} "
                  f"merged into {npz_path.name}")
            cal_p = Path(a.config_dir) / f"{a.match_id}_calibration.json"
            pcal_p = workspace(a.match_id, a.tmp_dir, k) / "config" / cal_p.name
            if cal_p.exists() and pcal_p.exists():
                cal, pcal = read_json(cal_p), read_json(pcal_p)
                cal.setdefault("motion_summary_periods", {})[name] = pcal.get("motion_summary")
                write_json(cal_p, cal)


def run(cmd):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([sys.executable, "-u", *map(str, cmd)])
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(map(str, cmd))}")


def stage_period(a, cfg, mdir, tool, extra=()):
    npz = dict(np.load(mdir / "frame_homographies.npz"))
    for k, (name, start, end, _) in enumerate(periods_of(cfg)):
        ws = workspace(a.match_id, a.tmp_dir, k)
        pdir = write_period_meta(a.match_id, a.tmp_dir, ws, start, end)
        sl = npz_slice(npz, start, end)
        if not len(sl["frames"]):
            raise SystemExit(f"{name}: no homographies for frames {start}-{end} - calibrate that period first")
        if tool == "track_identities.py":
            np.savez_compressed(pdir / "period_homographies.npz", **sl)
            det = pdir / "detections.parquet"
            if det.is_symlink() or not det.exists():
                det.unlink(missing_ok=True)
                det.symlink_to((mdir / "detections.parquet").resolve())
        cdir = write_period_config(a.match_id, a.config_dir, ws, start, end, pdir / "period_homographies.npz")
        print(f"[{a.match_id}] {name}: frames {start}-{end} ({len(sl['frames'])} calibrated)", flush=True)
        run([TOOLS / tool, "--match-id", a.match_id, "--tmp-dir", ws, "--config-dir", cdir, *extra])


# --------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------
def shape_offsets(tracks, ids, gk_id, until=None):
    """{track_id: (depth, width)}: median offset from the team's centroid,
    facing the opponent's goal (depth > 0 = further forward; width > 0 = the
    team's right), from observed rows before frame `until`."""
    t = tracks[tracks["track_id"].isin(list(ids)) & tracks["pitch_x"].notna()]
    if "status" in t.columns:
        t = t[t["status"].fillna("detected") == "detected"]
    if until is not None:
        t = t[t["frame"] < until]
    if t.empty:
        return {}
    gk = tracks[(tracks["track_id"] == gk_id) & tracks["pitch_x"].notna()]
    direction = 1.0 if (gk["pitch_x"].median() < 52.5 if len(gk) else True) else -1.0
    g = t.groupby("frame")
    cx, cy, cnt = g["pitch_x"].transform("mean"), g["pitch_y"].transform("mean"), g["pitch_x"].transform("size")
    rel = t.assign(dx=(t["pitch_x"] - cx) * direction, dy=(t["pitch_y"] - cy) * direction)[cnt >= 5]
    med = rel.groupby("track_id")[["dx", "dy"]].median()
    return {int(i): (float(r.dx), float(r.dy)) for i, r in med.iterrows()}


def match_by_shape(ref_pos, new_pos):
    """One-to-one {new id: ref id} minimising total distance in the shape."""
    from scipy.optimize import linear_sum_assignment

    r_ids, n_ids = list(ref_pos), list(new_pos)
    if not r_ids or not n_ids:
        return {}
    C = np.array([[np.hypot(new_pos[n][0] - ref_pos[r][0], new_pos[n][1] - ref_pos[r][1]) for r in r_ids]
                  for n in n_ids])
    ri, ci = linear_sum_assignment(C)
    return {n_ids[i]: r_ids[j] for i, j in zip(ri, ci)}


def named_ids(pm):
    """track ids the analyst named in this period (not role-inferred)."""
    out = {}
    for key, pl in pm["players"].items():
        src = str(pl.get("name_source") or "")
        if pl.get("anchors") or "calibration names" in src or "manual" in src:
            out[int(pl["track_ids"][0])] = (pl["team"], pl.get("number"))
    return out


def identity_map(first_pm, first_tracks, pm, tracks, team_first_sub):
    """{period id: first-period id} for one later period's 22 identities."""
    by_num = {(pl["team"], pl.get("number")): int(pl["track_ids"][0]) for pl in first_pm["players"].values()}
    mapping, how = {}, {}
    named = named_ids(pm)
    for team in TEAMS:
        gk = GK[team]
        mapping[gk] = gk
        how[gk] = "goalkeeper"
        ids = [int(pl["track_ids"][0]) for pl in pm["players"].values() if pl["team"] == team]
        out_ids = [i for i in ids if i != gk]
        taken = set()
        for i in out_ids:
            if i in named and named[i] in by_num:
                mapping[i] = by_num[named[i]]
                how[i] = "named in this half"
                taken.add(mapping[i])
        ref_ids = [int(pl["track_ids"][0]) for pl in first_pm["players"].values()
                   if pl["team"] == team and int(pl["track_ids"][0]) != gk and int(pl["track_ids"][0]) not in taken]
        free = [i for i in out_ids if i not in mapping]
        ref_pos = shape_offsets(first_tracks, ref_ids, gk)
        new_pos = shape_offsets(tracks, free, gk, until=team_first_sub.get(team))
        m = match_by_shape({r: ref_pos[r] for r in ref_ids if r in ref_pos},
                           {n: new_pos[n] for n in free if n in new_pos})
        for n, r in m.items():
            mapping[n] = r
            how[n] = "team-shape match to the first half"
        left_ref = [r for r in ref_ids if r not in set(mapping.values())]
        for n in [i for i in free if i not in mapping]:  # no shape (never seen): leftovers in order
            if left_ref:
                mapping[n] = left_ref.pop(0)
                how[n] = "leftover (no shape in this half)"
    return mapping, how


def remap_period(tracks, links, gaps, mapping, offset):
    """Period ids -> match ids: identities by `mapping`, fragments and
    tracklets shifted by `offset`."""
    def ids(s):
        s = s.astype("int64")
        out = s.map(mapping)
        frag = s >= 10000
        out = out.where(out.notna(), np.where(frag, s + offset, s))
        return out.astype("int64")

    tracks = tracks.copy()
    tracks["track_id"] = ids(tracks["track_id"])
    if "tracklet" in tracks.columns:
        t = tracks["tracklet"]
        tracks["tracklet"] = np.where(t.notna() & (t >= 0), t + offset, t)
    if links is not None:
        links = links.copy()
        links["tracklet"] = links["tracklet"] + offset
        links["identity"] = np.where(links["identity"] >= 0, links["identity"].map(mapping).fillna(-1),
                                     links["identity"]).astype(int)
    if gaps is not None and len(gaps):
        gaps = gaps.copy()
        gaps["track_id"] = ids(gaps["track_id"])
        for col in ("partner",):
            if col in gaps.columns:
                p = gaps[col]
                gaps[col] = np.where(p.notna() & (p >= 10000), p + offset, p.map(mapping).fillna(p))
    return tracks, links, gaps


def apply_substitutions(tracks, gaps, players, t2p, subs):
    """Split the outgoing player's identity at each substitution. Mutates
    players / t2p; returns (tracks, gaps, report)."""
    by_num = {(pl["team"], pl.get("number")): key for key, pl in players.items()}
    next_id = dict(SUB_BASE)
    used = {int(t) for t in t2p}
    report = []
    for s in subs:
        team = s["team"]
        off_key = by_num.get((team, s["off"]["number"] if isinstance(s["off"], dict) else s["off"]))
        if off_key is None:
            report.append({**s, "status": "player going off not found"})
            continue
        off_tid = int(players[off_key]["track_ids"][-1])
        while next_id[team] in used:
            next_id[team] += 1
        new_tid = next_id[team]
        used.add(new_tid)
        f = int(s["frame"])
        m = (tracks["track_id"] == off_tid) & (tracks["frame"] >= f)
        tracks.loc[m, "track_id"] = new_tid
        if gaps is not None and len(gaps):
            gm = (gaps["track_id"] == off_tid) & (gaps["lo"] >= f)
            gaps.loc[gm, "track_id"] = new_tid
        on = s["on"]
        letter = "a" if team == "team_a" else "b"
        key = f"{letter}{on['number']}"
        if key in players:
            key = f"{letter}id{new_tid}"
        rows = tracks[(tracks["track_id"] == new_tid)]
        players[key] = {
            "team": team, "number": on["number"], "name": on.get("name"), "role": players[off_key].get("role"),
            "name_source": f"substitution for #{players[off_key].get('number')} at frame {f}"
                           + (f" (minute {s['minute']}, {s.get('source', 'reference')})" if "minute" in s else ""),
            "anchors": 0, "track_ids": [new_tid], "substitute": True,
            "on": {"frame": f, "minute": s.get("minute"), "for": off_key},
            "first_frame": int(rows["frame"].min()) if len(rows) else None,
            "last_frame": int(rows["frame"].max()) if len(rows) else None,
        }
        players[off_key]["off"] = {"frame": f, "minute": s.get("minute"), "for": key}
        t2p[str(new_tid)] = key
        report.append({**s, "status": "applied", "off_track_id": off_tid, "on_track_id": new_tid,
                       "rows_moved": int(m.sum())})
    return tracks, gaps, report


def stage_combine(a, cfg, mdir):
    meta = load_meta(a.match_id, a.tmp_dir)
    fps = float(meta["fps"])
    ref = read_json(reference_path(a.match_id, a.config_dir))
    pers = periods_of(cfg)
    outs = []
    for k, (name, start, end, _) in enumerate(pers):
        pdir = workspace(a.match_id, a.tmp_dir, k) / a.match_id
        if not (pdir / "player_map.json").exists():
            raise SystemExit(f"{name}: no player_map.json in {pdir} - run the identities stage first")
        outs.append({
            "name": name, "start": start, "end": end,
            "tracks": pd.read_parquet(pdir / "tracks.parquet"),
            "pm": read_json(pdir / "player_map.json"),
            "links": pd.read_parquet(pdir / "identity_links.parquet") if (pdir / "identity_links.parquet").exists()
            else None,
            "gaps": pd.read_parquet(pdir / "gaps.parquet") if (pdir / "gaps.parquet").exists() else None,
            "ball": pd.read_parquet(pdir / "ball_track.parquet") if (pdir / "ball_track.parquet").exists() else None,
            "team_map": read_json(pdir / "team_map.json"),
        })
    subs = substitutions(ref, cfg, fps)
    first = outs[0]
    players = json.loads(json.dumps(first["pm"]["players"]))
    t2p = dict(first["pm"]["track_to_player"])
    for pl in players.values():
        pl["periods"] = {first["name"]: {k: pl.get(k) for k in ("name_source", "detections", "coverage",
                                                                 "tracklets", "fragile_links")}}
    offset = 0
    T, Lk, Gp, B = [], [], [], []
    unident, ignored = [], []
    det_ident = det_all = 0
    period_report = {}
    for k, o in enumerate(outs):
        if k == 0:
            mapping, how = {int(t): int(t) for t in t2p}, {}
        else:
            first_sub = {}
            for s in subs:
                if o["start"] <= s["frame"] <= o["end"]:
                    first_sub.setdefault(s["team"], s["frame"])
            mapping, how = identity_map(first["pm"], first["tracks"], o["pm"], o["tracks"], first_sub)
        tr, lk, gp = remap_period(o["tracks"], o["links"], o["gaps"], mapping, offset)
        if "period" not in tr.columns:
            tr["period"] = o["name"]
        T.append(tr)
        if lk is not None:
            Lk.append(lk.assign(period=o["name"]))
        if gp is not None:
            Gp.append(gp.assign(period=o["name"]))
        if o["ball"] is not None:
            B.append(o["ball"].assign(period=o["name"]))
        frag = lambda t: int(t) + offset if int(t) >= 10000 else mapping.get(int(t), int(t))  # noqa: E731
        unident += [frag(t) for t in o["pm"].get("unidentified_tracks", [])]
        ignored += [{**i, "track_id": frag(i["track_id"]), "period": o["name"]} for i in o["pm"].get("ignored_tracks", [])]
        n = sum(pl.get("detections", 0) for pl in o["pm"]["players"].values())
        f = o["pm"].get("identified_detection_fraction") or 0
        if f:
            det_ident += n
            det_all += n / f
        if k:
            inv = {}
            for key, pl in o["pm"]["players"].items():
                tid = int(pl["track_ids"][0])
                to = mapping.get(tid)
                pkey = t2p.get(str(to))
                if pkey is None:
                    continue
                inv[pkey] = {"name_source": how.get(tid), "tracker_name": pl.get("name"),
                             "detections": pl.get("detections"), "coverage": pl.get("coverage"),
                             "tracklets": pl.get("tracklets"), "fragile_links": pl.get("fragile_links")}
            for pkey, info in inv.items():
                players[pkey].setdefault("periods", {})[o["name"]] = info
            period_report[o["name"]] = {"identity_map": {str(a_): int(b_) for a_, b_ in mapping.items()},
                                        "how": {str(a_): v for a_, v in how.items()}}
        tl_max = int(o["tracks"]["tracklet"].max()) if "tracklet" in o["tracks"] and o["tracks"]["tracklet"].notna().any() else 0
        offset += tl_max + 1

    tracks = pd.concat(T, ignore_index=True)
    gaps = pd.concat(Gp, ignore_index=True) if Gp else None
    tracks, gaps, sub_report = apply_substitutions(tracks, gaps, players, t2p, subs)
    tracks = tracks.sort_values(["frame", "class", "track_id"]).reset_index(drop=True)

    # per-player totals over the match
    win = sum(o["end"] - o["start"] + 1 for o in outs)
    obs = tracks[tracks["class"] == "player"]
    if "status" in obs.columns:
        obs = obs[obs["status"].fillna("detected") == "detected"]
    for key, pl in players.items():
        rows = obs[obs["track_id"].isin(pl["track_ids"])]
        pl["detections"] = int(len(rows))
        pl["first_frame"] = int(rows["frame"].min()) if len(rows) else None
        pl["last_frame"] = int(rows["frame"].max()) if len(rows) else None
        on_f = (pl.get("on") or {}).get("frame")
        off_f = (pl.get("off") or {}).get("frame")
        span = sum(max(0, min(o["end"], (off_f or o["end"] + 1) - 1) - max(o["start"], on_f or o["start"]) + 1)
                   for o in outs)
        pl["coverage"] = round(rows["frame"].nunique() / max(span, 1), 3)
        pl["minutes_window"] = round(span / fps / 60, 1)
        if pl.get("substitute"):
            pl["tracklets"] = None
            pl["fragile_links"] = None
        else:
            pl["tracklets"] = sum((pl.get("periods") or {}).get(o["name"], {}).get("tracklets") or 0 for o in outs)
            pl["fragile_links"] = sum((pl.get("periods") or {}).get(o["name"], {}).get("fragile_links") or 0
                                      for o in outs)

    pm = dict(first["pm"])
    pm.update({
        "method": "tools/track_periods.py: tools/track_identities.py per period, joined by team shape; "
                  "substitutions from the reference file",
        "window": {"start_frame": outs[0]["start"], "end_frame": outs[-1]["end"]},
        "periods": {o["name"]: {"start_frame": o["start"], "end_frame": o["end"],
                                "identified_detection_fraction": o["pm"].get("identified_detection_fraction"),
                                "link_stability": o["pm"].get("link_stability"),
                                **period_report.get(o["name"], {})} for o in outs},
        "track_to_player": t2p, "players": players,
        "unidentified_tracks": sorted(set(unident)), "ignored_tracks": ignored,
        "identified_detection_fraction": round(det_ident / det_all, 3) if det_all else None,
        "substitutions": sub_report,
        "frames_in_window": win,
    })
    team_map = {}
    for t in tracks.loc[tracks["class"] == "player", "track_id"].unique():
        t = int(t)
        team_map[str(t)] = ("team_a" if 1 <= t < 100 else "team_b" if 101 <= t < 1000 else "unknown")
    tracks.to_parquet(mdir / "tracks.parquet", index=False)
    write_json(mdir / "player_map.json", pm)
    write_json(mdir / "team_map.json", team_map)
    if Lk:
        pd.concat(Lk, ignore_index=True).to_parquet(mdir / "identity_links.parquet", index=False)
    if gaps is not None:
        gaps.to_parquet(mdir / "gaps.parquet", index=False)
    if B:
        pd.concat(B, ignore_index=True).to_parquet(mdir / "ball_track.parquet", index=False)
    print(f"Wrote whole-match tracks.parquet ({len(tracks)} rows), player_map.json, team_map.json to {mdir}")
    for o in outs[1:]:
        r = period_report[o["name"]]
        print(f"  {o['name']}: " + ", ".join(f"{v} {sum(1 for x in r['how'].values() if x == v)}"
                                             for v in sorted(set(r["how"].values()))))
    for s in sub_report:
        print(f"  sub {s['team']} min {s.get('minute')} frame {s['frame']}: #{s['off']['number'] if isinstance(s['off'], dict) else s['off']}"
              f" -> #{s['on']['number']} {s['on'].get('name')}: {s['status']}"
              + (f" ({s['rows_moved']} rows)" if "rows_moved" in s else ""))
    try:
        from track_identities import rebase_track_edits
        rebase_track_edits(a.match_id, a.config_dir, tracks, pm)
    except Exception as exc:  # noqa: BLE001 - the tracks are written; fixes can be redone
        print(f"WARNING: could not rebase the Tracking tab's fixes: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--stages", default="merge,identities,ball,combine")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--tmp-dir", default=".tmp")
    ap.add_argument("--identities-args", default="", help="Extra args for track_identities.py (quoted)")
    a = ap.parse_args()
    cfg = match_config(a.match_id, a.tmp_dir, a.config_dir)
    mdir = match_dir(a.match_id, a.tmp_dir)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if "merge" in stages:
        stage_merge(a, cfg, mdir)
    if "identities" in stages:
        stage_period(a, cfg, mdir, "track_identities.py", a.identities_args.split())
    if "ball" in stages:
        stage_period(a, cfg, mdir, "track_ball.py")
    if "combine" in stages:
        stage_combine(a, cfg, mdir)
    print("Next: compute_possession.py --include-unidentified, compute_physical.py, compute_positions.py")


if __name__ == "__main__":
    main()
