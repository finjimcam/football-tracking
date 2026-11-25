"""Detect on-ball events from the tracking data -> .tmp/<id>/events.parquet (+ events.json).

Issue #7, first step: rule-based events from what the pipeline already has
(ball track, possession holder per frame, persistent identities), with no new
model. A later step can train a classifier on these, checked against the
same Opta events.

Control spells: runs of frames where the same player is the possession
holder (compute_possession.py: nearest player within 2.5 m of the ball),
bridging gaps of <= --bridge-frames. Spells shorter than --min-spell-frames
are dropped as flicker between a player and his marker (on Man City v Spurs
H1 that took fake tackles from ~230 to ~35 per team and the pass split from
51% to 63% City). Consecutive spells A -> B give:

  pass        A and B on the same team, ball moved >= --pass-min-m,
              B started <= --max-gap-s after A ended
  pass_lost   different teams, ball moved >= --pass-min-m (an interception
              or a misplaced pass; the event belongs to A)
  tackle      different teams, ball moved less (B took it off A: a duel)
  shot        A's team attacks a goal A was within --shot-max-m of, and in the
              next --shot-window-s the ball travels towards that goal at
              >= --shot-min-speed and ends within --shot-line-m of the goal
              line (or vanishes heading there: in the net, behind, blocked)

Which goal each team attacks comes from where its goalkeeper stands.

events.json also scores shots against the reference's timestamped Opta events
(config/<id>_reference.json first_half_events: shots and goals), matching each
Opta shot to a detected one within --match-s seconds, and compares the pass
split with Opta's full-match pass totals.

--instances also writes the events into the Code tab's instances
(config/<id>_instances.json) as auto-tagged instances, replacing any earlier
auto ones and leaving hand-tagged ones alone.

Usage:
    python tools/detect_events.py --match-id <id> [--instances]
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (clock_label, load_calibration, load_meta, load_player_map, match_config, match_dir,
                     pitch_dims, reference_path, write_json)

TEAMS = ("team_a", "team_b")
# how auto events look in the Code tab (same colours / clip windows as its Default template)
CODE_STYLE = {"pass": ("Pass", "#4C8DF6", 3, 4), "pass_lost": ("Turnover", "#E8A33A", 5, 6),
              "tackle": ("Ball recovery", "#5DBB63", 5, 6), "shot": ("Shot", "#EF5B5B", 8, 5)}


def control_spells(timeline, team_of, bridge=6, min_frames=5, merge_s=2.0, fps=25.0):
    """[(start, end, holder, team)] from possession_timeline's holder column.

    The holder is just the nearest player within 2.5 m, so it flickers between
    a player and his marker. Spells shorter than min_frames are dropped as
    flicker, then spells of one holder less than merge_s apart are joined."""
    tl = timeline[timeline["holder"] >= 0][["frame", "holder", "candidate"]].to_numpy()
    spells = []
    for f, h, cand in tl:
        team = team_of.get(int(h)) or (cand if cand in TEAMS else None)
        if spells and spells[-1][2] == h and f - spells[-1][1] <= bridge:
            spells[-1][1] = f
        else:
            spells.append([int(f), int(f), int(h), team])
    out = []
    for s in spells:
        if s[3] not in TEAMS or s[1] - s[0] + 1 < min_frames:
            continue
        if out and out[-1][2] == s[2] and s[0] - out[-1][1] <= merge_s * fps:
            out[-1][1] = s[1]
        else:
            out.append(list(s))
    return [tuple(s) for s in out]


def attacking_goal(tracks, pmap, L):
    """{team: x of the goal it attacks}, from where each keeper stands (else the team's mean x)."""
    out = {}
    pl = tracks[tracks["class"] == "player"]
    for team in TEAMS:
        gk = [int(t) for p in pmap.get("players", {}).values()
              if p.get("team") == team and p.get("role") == "GK" for t in p.get("track_ids", [])]
        xs = pl.loc[pl["track_id"].isin(gk), "pitch_x"]
        if xs.empty:
            ids = [int(t) for p in pmap.get("players", {}).values() if p.get("team") == team
                   for t in p.get("track_ids", [])]
            xs = pl.loc[pl["track_id"].isin(ids), "pitch_x"]
        own = xs.median() if len(xs) else (0 if team == "team_a" else L)
        out[team] = L if own < L / 2 else 0.0
    if out["team_a"] == out["team_b"]:  # both keepers at one end: trust team_a's
        out["team_b"] = L - out["team_a"]
    return out


def detect(spells, ball, goals, fps, L, W, pass_min_m=4.0, max_gap_s=6.0, shot_max_m=35.0,
           shot_window_s=2.5, shot_min_speed=8.0, shot_line_m=4.0, shot_post_m=16.0):
    """Events from control spells + the ball track. ball: frame-indexed DataFrame with
    pitch_x, pitch_y (detected rows only)."""
    bf = ball.index.to_numpy()
    bx, by = ball["pitch_x"].to_numpy(), ball["pitch_y"].to_numpy()

    def pos(f):
        i = np.searchsorted(bf, f)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(bf) and abs(bf[j] - f) <= 3 and (best is None or abs(bf[j] - f) < abs(bf[best] - f)):
                best = j
        return (bx[best], by[best]) if best is not None else None

    events = []
    shot_after = {}
    for k, (s, e, h, team) in enumerate(spells):
        # shot: from the end of this spell, the ball heads for the goal team attacks
        gx = goals[team]
        p0 = pos(e)
        if p0 is not None and abs(p0[0] - gx) <= shot_max_m:
            nxt = spells[k + 1][0] if k + 1 < len(spells) else e + int(shot_window_s * fps) + 1
            stop = min(e + int(shot_window_s * fps), nxt + 2)
            sel = (bf > e) & (bf <= stop)
            if sel.any():
                fx, xx, yy = bf[sel], bx[sel], by[sel]
                d0 = abs(p0[0] - gx)
                dx = np.abs(xx - gx)
                j = int(np.argmin(dx))
                travel = np.hypot(xx[j] - p0[0], yy[j] - p0[1])
                speed = travel / max((fx[j] - e) / fps, 1 / fps)
                toward = d0 - dx[j]
                line_ok = dx[j] <= shot_line_m and abs(yy[j] - W / 2) <= shot_post_m
                # vanished heading for goal: last sight close, moving fast towards it
                last = len(fx) - 1
                vanished = (fx[last] < stop - int(0.6 * fps) and dx[last] <= 12 and abs(yy[last] - W / 2) <= shot_post_m
                            and d0 - dx[last] >= 5)
                if speed >= shot_min_speed and toward >= 4 and (line_ok or vanished):
                    events.append({"frame": e, "end_frame": int(fx[j]), "type": "shot", "team": team, "player": h,
                                   "to_player": None, "x": p0[0], "y": p0[1], "end_x": xx[j], "end_y": yy[j],
                                   "speed_ms": round(float(speed), 1)})
                    shot_after[k] = True
        if k + 1 >= len(spells) or shot_after.get(k):
            continue
        s2, e2, h2, team2 = spells[k + 1]
        if h2 == h or (s2 - e) / fps > max_gap_s:
            continue
        p1, p2 = pos(e), pos(s2)
        if p1 is None or p2 is None:
            continue
        d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if team2 == team:
            if d < pass_min_m:
                continue
            kind = "pass"
        else:
            kind = "pass_lost" if d >= pass_min_m else "tackle"
        ev_team, ev_player, other = (team, h, h2) if kind != "tackle" else (team2, h2, h)
        events.append({"frame": e, "end_frame": s2, "type": kind, "team": ev_team, "player": ev_player,
                       "to_player": other, "x": p1[0], "y": p1[1], "end_x": p2[0], "end_y": p2[1],
                       "speed_ms": round(d / max((s2 - e) / fps, 1 / fps), 1)})
    ev = pd.DataFrame(events, columns=["frame", "end_frame", "type", "team", "player", "to_player", "x", "y",
                                       "end_x", "end_y", "speed_ms"])
    ev["length_m"] = np.hypot(ev["end_x"] - ev["x"], ev["end_y"] - ev["y"]).round(1)
    return ev.sort_values("frame").reset_index(drop=True)


def score_shots(ev, ref_events, kickoff, fps, window_s=5.0):
    """Recall of Opta shots/goals among detected shots (each detection matches once)."""
    opta = [e for e in ref_events if e.get("type") in ("shot", "goal")]
    shots = ev[ev["type"] == "shot"]
    free = dict(zip(shots.index, shots["frame"]))
    hits = []
    for e in opta:
        f = kickoff + fps * e["s"]
        cand = [(abs(fr - f), i) for i, fr in free.items() if abs(fr - f) <= window_s * fps]
        if cand:
            _, i = min(cand)
            del free[i]
            hits.append({"s": e["s"], "type": e["type"], "team": e["team"], "player": e.get("player"),
                         "detected_frame": int(shots.loc[i, "frame"]), "team_ok": shots.loc[i, "team"] == e["team"]})
        else:
            hits.append({"s": e["s"], "type": e["type"], "team": e["team"], "player": e.get("player"),
                         "detected_frame": None, "team_ok": False})
    n = len(opta)
    found = sum(h["detected_frame"] is not None for h in hits)
    return {"opta_shots": n, "found": found, "recall": round(found / n, 3) if n else None,
            "team_ok": sum(h["team_ok"] for h in hits), "detected": len(shots),
            "unmatched_detections": len(free), "per_shot": hits}


def write_instances(ev, path, fps, duration_s, name_of, team_names, clock=None):
    """Replace the auto instances in the Code tab's instances file with these events."""
    path = Path(path)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    keep = [i for i in d.get("instances", []) if not i.get("auto")]
    now = time.time()
    for r in ev.itertuples():
        code, color, lead, lag = CODE_STYLE[r.type]
        t = r.frame / fps
        labels = {"Team": team_names.get(r.team, r.team), "Source": "auto"}
        if name_of.get(r.player):
            labels["Player"] = name_of[r.player]
        keep.append({"id": uuid.uuid4().hex[:10], "code": code, "color": color, "t": round(t, 3),
                     "start": round(max(0.0, t - lead), 3), "end": round(min(duration_s or t + lag, t + lag), 3),
                     "labels": labels, "created": now, "auto": True})
    keep.sort(key=lambda i: (i["t"], i.get("created", 0)))
    d["instances"] = keep
    d["updated"] = now
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return len(ev), len(keep) - len(ev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--bridge-frames", type=int, default=6)
    ap.add_argument("--min-spell-frames", type=int, default=5,
                    help="Drop control spells shorter than this (holder flicker)")
    ap.add_argument("--pass-min-m", type=float, default=4.0)
    ap.add_argument("--max-gap-s", type=float, default=6.0)
    ap.add_argument("--shot-max-m", type=float, default=35.0)
    ap.add_argument("--shot-window-s", type=float, default=2.5)
    ap.add_argument("--shot-min-speed", type=float, default=8.0)
    ap.add_argument("--match-s", type=float, default=5.0, help="Opta shot matching window (s)")
    ap.add_argument("--instances", action="store_true", help="Also add the events to the Code tab as auto instances")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()

    meta = load_meta(a.match_id, a.tmp_dir)
    mdir = match_dir(a.match_id, a.tmp_dir)
    fps = meta["fps"]
    cfg = match_config(a.match_id, a.tmp_dir, a.config_dir)
    calib = load_calibration(a.match_id, a.config_dir, a.calibration)
    L, W = pitch_dims(calib)
    for need in ("possession_timeline.parquet", "ball_track.parquet", "tracks.parquet"):
        if not (mdir / need).exists():
            raise SystemExit(f"{need} not found: run track_ball.py and compute_possession.py first")
    timeline = pd.read_parquet(mdir / "possession_timeline.parquet")
    ball = pd.read_parquet(mdir / "ball_track.parquet", columns=["frame", "pitch_x", "pitch_y", "interp"])
    ball = ball[~ball["interp"].astype(bool)].drop_duplicates("frame").set_index("frame").sort_index()
    tracks = pd.read_parquet(mdir / "tracks.parquet", columns=["frame", "track_id", "class", "pitch_x"])
    pmap = load_player_map(a.match_id, a.tmp_dir) or {}
    team_of, name_of = {}, {}
    for p in pmap.get("players", {}).values():
        for t in p.get("track_ids", []):
            team_of[int(t)] = p.get("team")
            if p.get("name"):
                name_of[int(t)] = f"{p['number']} {p['name']}" if p.get("number") is not None else p["name"]

    goals = attacking_goal(tracks, pmap, L)
    spells = control_spells(timeline, team_of, a.bridge_frames, a.min_spell_frames, fps=fps)
    ev = detect(spells, ball[["pitch_x", "pitch_y"]], goals, fps, L, W, pass_min_m=a.pass_min_m,
                max_gap_s=a.max_gap_s, shot_max_m=a.shot_max_m, shot_window_s=a.shot_window_s,
                shot_min_speed=a.shot_min_speed)
    ev.insert(1, "t_s", (ev["frame"] / fps).round(2))
    ev.insert(2, "clock", [clock_label(f, cfg, fps) for f in ev["frame"]])
    ev["player_name"] = ev["player"].map(name_of)
    ev["to_player_name"] = ev["to_player"].map(name_of)
    ev.to_parquet(mdir / "events.parquet", index=False)

    counts = {t: ev[ev["team"] == t]["type"].value_counts().to_dict() for t in TEAMS}
    summary = {"match_id": a.match_id, "spells": len(spells), "attacking_goal_x": goals, "counts": counts,
               "events": len(ev)}
    names = {t: cfg.get(f"{t}_name") or t for t in TEAMS}
    print(f"[{a.match_id}] {len(spells)} control spells -> {len(ev)} events; "
          f"{names['team_a']} attack x={goals['team_a']:.0f}, {names['team_b']} x={goals['team_b']:.0f}")
    for t in TEAMS:
        c = counts[t]
        print(f"  {names[t]:<12} passes {c.get('pass', 0):>4}  lost {c.get('pass_lost', 0):>4}  "
              f"tackles {c.get('tackle', 0):>4}  shots {c.get('shot', 0):>3}")
    ref = {}
    rp = reference_path(a.match_id, a.config_dir)
    if rp.exists():
        ref = json.loads(rp.read_text(encoding="utf-8"))
    kickoff = (cfg.get("clock_sync") or {}).get("kickoff_frame")
    if ref.get("first_half_events") and kickoff is not None:
        sc = score_shots(ev, ref["first_half_events"], kickoff, fps, a.match_s)
        summary["opta_shots"] = sc
        print(f"  Opta shots+goals found: {sc['found']}/{sc['opta_shots']} (within {a.match_s:.0f} s; right team "
              f"{sc['team_ok']}), {sc['unmatched_detections']} of {sc['detected']} detected shots unmatched")
    tp = (ref.get("full_match_team_stats") or {}).get("total_passes")
    if tp:
        ours = {t: counts[t].get("pass", 0) for t in TEAMS}
        tot = sum(ours.values()) or 1
        summary["pass_share"] = {"ours": {t: round(ours[t] / tot, 3) for t in TEAMS},
                                 "opta_full_match": {t: round(tp[t] / sum(tp.values()), 3) for t in TEAMS}}
        print(f"  pass share {names['team_a']}: ours {ours['team_a'] / tot:.0%} vs Opta full match "
              f"{tp['team_a'] / sum(tp.values()):.0%} ({tp['team_a']}/{tp['team_b']} passes over 90 min)")
    write_json(mdir / "events.json", summary)
    print(f"Wrote {mdir / 'events.parquet'} and events.json")
    if a.instances:
        n, kept = write_instances(ev, Path(a.config_dir) / f"{a.match_id}_instances.json", fps,
                                  meta.get("duration_s"), name_of, names)
        print(f"Added {n} auto instances to the Code tab (kept {kept} hand-tagged ones)")


if __name__ == "__main__":
    main()
