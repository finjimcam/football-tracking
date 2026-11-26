"""Find clip-worthy moments in a tracked match -> .tmp/<id>/clips.json (issue #58).

Three groups, each clip tagged with the team it belongs to:

  set_pieces   goal kicks, throw-ins, corners, free kicks (penalties too)
  chances      shots, and spells where a team has the ball in the opponent's box
  final_third  completed passes from outside the attacking third into it

It needs no event data, only what the pipeline already made: the ball track
(track_ball.py), the possession timeline (compute_possession.py), the tracks
and the player map. So it runs on any match tracked that far.

Set pieces are restarts. A restart is found in two ways:
  - a stoppage: the identified players' median speed stays under
    --stop-speed for at least --stop-s (players stroll while the ball is out
    or a free kick is set up), then play resumes. The restart is the first
    "dead ball" (still, then kicked) near the end of the stoppage, or else
    wherever the ball is first seen as play resumes.
  - a dead ball anywhere else: the ball still for at least --still-s, then
    moving away continuously (a quick free kick).
Each restart is named by where the ball was: within 4 m of a corner flag a
corner, in or just around the goal area a goal kick, on or over a touchline a
throw-in, on the penalty spot a penalty, the centre spot a kick-off (not
listed), anywhere else a free kick.

Ball detections that sit still in the same spot again and again across the
match (a painted spot, a spare ball behind the boards) are "clutter" and are
ignored, so they don't read as dead balls.

Chances: shots come from detect_events.detect() (run per period, since the
teams change ends), and a team's spells with the ball in the opponent's
penalty area become "box entries". The two are merged when they overlap, so
a move into the box that ends in a shot is one clip.

Passes into the final third: detect_events.detect() passes (same team, ball
moved >= 4 m) that start more than --third-m from the goal the passer's team
attacks and end within it.

Published events: when config/<id>_reference.json has "match_events"
(minute-stamped shots, goals, free kicks, corners... from ESPN / BBC / FotMob),
each one is tied to a detected clip of that team inside its minute (+-30 s),
which then carries the published player and outcome; one with no match becomes
a clip of its own at the best moment in that minute. Minutes are turned into
frames with the match config's clock_sync and each period's kickoff_frame /
clock_start_s.

The ball is often lost on broadcast footage (in the air, behind players,
close-ups and replays), so some moments are missed and some are labelled
wrongly. The Clips panel in the Lines tab is where they are checked and
dropped.

Usage:
    python tools/detect_clips.py --match-id <id>
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from _common import (clock_label, load_calibration, load_meta, load_player_map, match_config, match_dir,
                     pitch_dims, reference_path, write_json)
from detect_events import TEAMS, attacking_goal, control_spells, detect

GROUPS = {
    "set_pieces": ("goal_kick", "throw_in", "corner", "free_kick", "penalty"),
    "chances": ("shot", "box_entry"),
    "final_third": ("pass",),
}
GROUP_OF = {k: g for g, kinds in GROUPS.items() for k in kinds}
# seconds of video before / after the key frame of each kind of clip
WINDOWS = {"goal_kick": (2, 8), "throw_in": (2, 6), "corner": (3, 10), "free_kick": (3, 10), "penalty": (4, 8),
           "shot": (8, 4), "box_entry": (6, 5), "pass": (4, 5)}


# --------------------------------------------------------------------------
# periods / ends
# --------------------------------------------------------------------------
def period_ranges(cfg, lo, hi):
    """[(start_frame, end_frame)] of the match's periods, in time order (else one range)."""
    per = sorted((int(p["start_frame"]), int(p["end_frame"])) for p in (cfg.get("periods") or {}).values()
                 if p.get("start_frame") is not None and p.get("end_frame") is not None)
    return per or [(lo, hi)]


def goals_by_period(tracks, pmap, L, periods):
    """[(start, end, {team: x of the goal it attacks})]: keepers' positions per period."""
    out = []
    for s, e in periods:
        t = tracks[(tracks["frame"] >= s) & (tracks["frame"] <= e)]
        out.append((s, e, attacking_goal(t, pmap, L)))
    return out


def goal_lookup(gp):
    def at(frame):
        for s, e, g in gp:
            if s <= frame <= e:
                return g
        # between periods / outside: nearest period
        return min(gp, key=lambda p: min(abs(frame - p[0]), abs(frame - p[1])))[2]
    return at


# --------------------------------------------------------------------------
# restarts (set pieces)
# --------------------------------------------------------------------------
def still_runs(f, x, y, fps, radius=0.8, min_s=0.6, max_gap=4):
    """[(first_frame, last_frame, x, y, i0, i1)]: runs of ball detections that stay within
    `radius` m of their running mean for at least min_s."""
    runs, i, n = [], 0, len(f)
    while i < n:
        j = i
        while (j + 1 < n and f[j + 1] - f[j] <= max_gap
               and np.hypot(x[j + 1] - x[i:j + 1].mean(), y[j + 1] - y[i:j + 1].mean()) < radius):
            j += 1
        if f[j] - f[i] >= min_s * fps:
            runs.append((int(f[i]), int(f[j]), float(x[i:j + 1].mean()), float(y[i:j + 1].mean()), i, j))
        i = j + 1
    return runs


def clutter_cells(runs, fps, cell=1.5, min_runs=4, min_spread_s=180):
    """Grid cells where still 'balls' turn up again and again over the match (painted spots,
    spare balls): {(cx, cy)}."""
    by = {}
    for s, _, x, y, *_ in runs:
        by.setdefault((int(np.floor(x / cell)), int(np.floor(y / cell))), []).append(s)
    return {c for c, ss in by.items() if len(ss) >= min_runs and (max(ss) - min(ss)) / fps >= min_spread_s}


def in_clutter(x, y, cells, cell=1.5):
    cx, cy = int(np.floor(x / cell)), int(np.floor(y / cell))
    return any((cx + dx, cy + dy) in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1))


def kicked(f, x, y, run, fps, move_m=3.0, within_s=1.2, first_step_m=2.5, step_m=2.5):
    """Does the ball leave a still run, continuously, by move_m within within_s?"""
    _, e, mx, my, _, j = run
    k = j + 1
    path = []
    while k < len(f) and f[k] <= e + within_s * fps:
        path.append((f[k], x[k], y[k]))
        k += 1
    if not path:
        return False
    d = [np.hypot(px - mx, py - my) for _, px, py in path]
    if d[0] > first_step_m:
        return False
    prev = (e, mx, my)
    for fr, px, py in path:  # no jumps: the tracker switching to another 'ball'
        if np.hypot(px - prev[1], py - prev[2]) > step_m * max(fr - prev[0], 1):
            return False
        prev = (fr, px, py)
    return max(d) >= move_m


def stoppages(speed, frames, fps, max_speed=1.3, min_s=3.0):
    """[(first, last)] frames where the players' median speed stays under max_speed for min_s."""
    slow = np.asarray(speed < max_speed)
    out, s = [], None
    for i, v in enumerate(slow):
        if v and s is None:
            s = i
        if s is not None and (not v or i == len(slow) - 1):
            last = i if v else i - 1
            if frames[last] - frames[s] >= min_s * fps:
                out.append((int(frames[s]), int(frames[last])))
            s = None
    return out


def player_speed(tracks, ids, fps, smooth_frames=30):
    """Median speed (m/s) of the identified players, per frame, smoothed."""
    p = tracks[tracks["track_id"].isin(list(ids)) & tracks["pitch_x"].notna()]
    p = p.sort_values(["track_id", "frame"])
    g = p.groupby("track_id")
    v = np.hypot(g["pitch_x"].diff(), g["pitch_y"].diff()) / g["frame"].diff() * fps
    p = p.assign(v=v)
    per = p[p["v"] < 12].groupby("frame")["v"].median()
    return per.rolling(smooth_frames, center=True, min_periods=5).median()


def classify(x, y, L, W):
    """Restart type from where the ball was."""
    if min(np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)) <= 4.0:
        return "corner"
    if (x <= 7.0 or x >= L - 7.0) and abs(y - W / 2) <= 11.5:
        return "goal_kick"
    if np.hypot(x - L / 2, y - W / 2) <= 6.0:
        return "kick_off"
    if min(abs(x - 11), abs(x - (L - 11))) <= 1.5 and abs(y - W / 2) <= 1.5:
        return "penalty"
    if y <= 2.5 or y >= W - 2.5:
        return "throw_in"
    if x < -1 or x > L + 1:
        return "goal_kick" if abs(y - W / 2) <= 20 else "corner"
    return "free_kick"


def restart_team(kind, x, frame, timeline_holder, team_of, goals, fps, L):
    """Who takes the restart: by the end of the pitch for corners / goal kicks, else the first
    player on the ball within 2 s."""
    if kind in ("corner", "goal_kick"):
        end_x = 0.0 if x < L / 2 else L
        attacking = [t for t in TEAMS if abs(goals[t] - end_x) < 1]
        if attacking:
            a = attacking[0]
            return a if kind == "corner" else ("team_b" if a == "team_a" else "team_a")
    h = timeline_holder
    lo = np.searchsorted(h[0], frame - int(0.5 * fps))
    hi = np.searchsorted(h[0], frame + int(2.0 * fps))
    for k in range(lo, hi):
        t = team_of.get(int(h[1][k])) or (h[2][k] if h[2][k] in TEAMS else None)
        if t:
            return t
    return None


def taker_cue(rows, goals, L, W, corner_m=3.5, touch_m=3.0):
    """Restart type from where the players stand at the end of a stoppage (the ball is often
    not seen there): an outfield player at a corner flag -> corner, one over a touchline ->
    throw-in, a keeper in his goal area with no opponent in that penalty area -> goal kick.
    rows: DataFrame of identified players over the last seconds of the stoppage, with team
    and gk columns. Returns (kind, team, x, y) or None."""
    if rows is None or rows.empty:
        return None
    of = rows[~rows["gk"]]
    for (cx, cy) in ((0, 0), (0, W), (L, 0), (L, W)):
        d = np.hypot(of["pitch_x"] - cx, of["pitch_y"] - cy)
        hit = of[d <= corner_m]
        if len(hit) >= 5:  # held there, not running past
            team = hit["team"].mode().iat[0]
            return "corner", team, float(cx), float(cy)
    over = of[((of["pitch_y"] < -0.2) & (of["pitch_y"] > -touch_m)) | ((of["pitch_y"] > W + 0.2) & (of["pitch_y"] < W + touch_m))]
    over = over[(over["pitch_x"] > 2) & (over["pitch_x"] < L - 2)]
    if len(over) >= 5:
        team = over["team"].mode().iat[0]
        return "throw_in", team, float(over["pitch_x"].median()), float(over["pitch_y"].median())
    gk = rows[rows["gk"]]
    for team, g in gk.groupby("team"):
        own_x = 0.0 if abs(goals[team] - L) < 1 else L  # the goal this keeper defends
        in_area = g[(np.abs(g["pitch_x"] - own_x) <= 6.0) & (np.abs(g["pitch_y"] - W / 2) <= 10.0)]
        if len(in_area) >= 0.6 * len(g) and len(g) >= 5:
            opp = rows[(rows["team"] != team) & (np.abs(rows["pitch_x"] - own_x) <= 16.5)
                       & (np.abs(rows["pitch_y"] - W / 2) <= 20.16)]
            if opp["track_id"].nunique() <= 1:
                return "goal_kick", team, float(own_x + (5.5 if own_x == 0 else -5.5)), float(W / 2)
    return None


def next_holder_near(holder, team_of, frame, x, y, team, fps, pos_at, within_s=3.0, near_m=20.0):
    """Is the first player on the ball after `frame` (within within_s) of `team` and within near_m
    of (x, y)? Rules out a substitute leaving over the touchline, a player waiting by a corner flag."""
    lo = np.searchsorted(holder[0], frame)
    hi = np.searchsorted(holder[0], frame + int(within_s * fps))
    for k in range(lo, hi):
        h = int(holder[1][k])
        if h < 0:
            continue
        if team_of.get(h) != team:
            return False
        p = pos_at(int(holder[0][k]), h)
        return p is None or np.hypot(p[0] - x, p[1] - y) <= near_m
    return False


def kickoff_cue(rows, goals, L, frac=0.85, min_players=12, spot_m=9.0, margin_m=3.0):
    """A kick-off: every player in his own half (the one with the goal his team defends), with
    someone on the centre spot. rows: identified players over the last seconds of a stoppage."""
    if rows is None or rows.empty:
        return False
    last = rows[rows["frame"] >= rows["frame"].max() - 15]
    if last["track_id"].nunique() < min_players:
        return False
    own = np.where(np.isclose([goals.get(t, -1) for t in last["team"]], L), last["pitch_x"] <= L / 2 + margin_m,
                   last["pitch_x"] >= L / 2 - margin_m)
    near_spot = (np.hypot(last["pitch_x"] - L / 2, last["pitch_y"] - 34) <= spot_m).any()
    return bool(own.mean() >= frac and near_spot)


def find_restarts(ball, speed, timeline, team_of, goal_at, fps, L, W, a, players=None, wide=None):
    f = ball.index.to_numpy()
    x, y = ball["pitch_x"].to_numpy(), ball["pitch_y"].to_numpy()
    runs = still_runs(f, x, y, fps, min_s=0.3)
    cells = clutter_cells(runs, fps)
    live = [r for r in runs if not in_clutter(r[2], r[3], cells)]
    dead = [r for r in live if kicked(f, x, y, r, fps)]

    stops = stoppages(speed.to_numpy(), speed.index.to_numpy(), fps, a.stop_speed, a.stop_s)
    holder = (timeline["frame"].to_numpy(), timeline["holder"].to_numpy(), timeline["candidate"].to_numpy())
    pf = players["frame"].to_numpy() if players is not None else None
    pos_idx = players.set_index(["frame", "track_id"])[["pitch_x", "pitch_y"]] if players is not None else None

    def pos_at(frame, tid):
        try:
            r = pos_idx.loc[(frame, tid)]
        except KeyError:
            return None
        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        return float(r["pitch_x"]), float(r["pitch_y"])

    found = []  # (frame, kind, team, x, y, how)
    used = set()
    for s, e in stops:
        if pf is not None:  # a kick-off (after a goal or at half time): players split by the halfway line
            i0, i1 = np.searchsorted(pf, e - int(3 * fps)), np.searchsorted(pf, e + 1)
            if kickoff_cue(players.iloc[i0:i1], goal_at(e), L):
                found.append((e, "kick_off", None, L / 2, W / 2, "stoppage, players"))
                continue
        cand = [r for r in dead if e - 2 * fps <= r[1] <= e + 3 * fps]
        if cand:
            r = cand[0]
            used.add(r[1])
            kind = classify(r[2], r[3], L, W)
            found.append((r[1], kind, None, r[2], r[3], "stoppage, dead ball"))
            continue
        if pf is not None and not (wide is not None and wide(e)):
            # nothing seen of the ball: where are the players standing? (not in a zoomed-out
            # wide view, where positions are too rough)
            i0, i1 = np.searchsorted(pf, e - int(3 * fps)), np.searchsorted(pf, e + 1)
            cue = taker_cue(players.iloc[i0:i1], goal_at(e), L, W)
            if cue and cue[0] == "throw_in" and not next_holder_near(holder, team_of, e, cue[2], cue[3], cue[1],
                                                                       fps, pos_at):
                cue = None
            if cue:
                found.append((e, cue[0], cue[1], cue[2], cue[3], "stoppage, players"))
    for r in dead:  # quick restarts that didn't stop play for long
        if r[1] not in used and r[1] - r[0] >= a.still_s * fps:
            found.append((r[1], classify(r[2], r[3], L, W), None, r[2], r[3], "dead ball"))
    found.sort(key=lambda t: t[0])

    out, last = [], -10 ** 9
    kickoffs = []
    for frame, kind, team, px, py, how in found:
        if frame - last < 4 * fps:  # one restart per stoppage
            continue
        last = frame
        if kind == "kick_off":
            kickoffs.append(int(frame))
            continue
        team = team or restart_team(kind, px, frame, holder, team_of, goal_at(frame), fps, L)
        out.append({"frame": int(frame), "kind": kind, "team": team, "x": round(px, 1), "y": round(py, 1),
                    "how": how})
    return out, stops, len(cells), kickoffs


# --------------------------------------------------------------------------
# chances / final third
# --------------------------------------------------------------------------
def in_box(x, y, goal_x, L, W):
    return abs(x - goal_x) <= 16.5 and abs(y - W / 2) <= 20.16


def box_entries(timeline, goal_at, fps, L, W, min_s=0.4, bridge_s=2.0):
    """[(first, last, team)]: a team in possession with the ball in the box it attacks."""
    tl = timeline[timeline["ball"].astype(bool) & timeline["state"].isin(TEAMS)]
    rows = []
    for fr, st, bx, by in tl[["frame", "state", "ball_x", "ball_y"]].itertuples(index=False):
        if pd.notna(bx) and in_box(bx, by, goal_at(fr)[st], L, W):
            rows.append((int(fr), st))
    out = []
    for fr, st in rows:
        if out and out[-1][2] == st and fr - out[-1][1] <= bridge_s * fps:
            out[-1][1] = fr
        else:
            out.append([fr, fr, st])
    return [tuple(o) for o in out if o[1] - o[0] >= min_s * fps]


def events_by_period(spells, ball, gp, fps, L, W):
    evs = []
    for s, e, goals in gp:
        sp = [x for x in spells if s <= x[0] <= e]
        if sp:
            evs.append(detect(sp, ball, goals, fps, L, W))
    if not evs:
        return pd.DataFrame(columns=["frame", "end_frame", "type", "team", "player", "x", "y", "end_x", "end_y"])
    return pd.concat(evs, ignore_index=True).sort_values("frame").reset_index(drop=True)


def chances(ev, entries, fps, merge_s=6.0):
    """Shots and box entries, merged when within merge_s: [{frame, kind, team, ...}]."""
    items = [{"frame": int(r.frame), "kind": "shot", "team": r.team, "player": int(r.player),
              "x": round(float(r.x), 1), "y": round(float(r.y), 1), "first": int(r.frame)}
             for r in ev[ev["type"] == "shot"].itertuples()]
    for s, e, team in entries:
        items.append({"frame": int(s), "kind": "box_entry", "team": team, "player": None, "x": None, "y": None,
                      "first": int(s)})
    items.sort(key=lambda d: d["first"])
    out = []
    for it in items:
        prev = out[-1] if out else None
        if prev and prev["team"] == it["team"] and it["first"] - prev["last"] <= merge_s * fps:
            prev["last"] = it["first"]
            if it["kind"] == "shot":  # a move that ends in a shot is a shot; key frame = the shot
                prev.update(kind="shot", frame=it["frame"], player=it["player"], x=it["x"], y=it["y"])
            continue
        out.append(dict(it, last=it["first"]))
    for o in out:
        o["lead_s"] = round((o["frame"] - o["first"]) / fps, 1)
    return out


def final_third_passes(ev, goal_at, third_m=35.0):
    out = []
    for r in ev[ev["type"] == "pass"].itertuples():
        gx = goal_at(r.frame)[r.team]
        if abs(r.x - gx) > third_m and abs(r.end_x - gx) <= third_m:
            out.append({"frame": int(r.frame), "kind": "pass", "team": r.team, "player": int(r.player),
                        "to_player": int(r.to_player) if pd.notna(r.to_player) else None,
                        "x": round(float(r.x), 1), "y": round(float(r.y), 1),
                        "end_x": round(float(r.end_x), 1), "end_y": round(float(r.end_y), 1),
                        "end_frame": int(r.end_frame)})
    return out


# --------------------------------------------------------------------------
# published, minute-stamped events (reference "match_events")
# --------------------------------------------------------------------------
PUBLISHED_KIND = {"shot": "shot", "goal": "shot", "free_kick": "free_kick", "throw_in": "throw_in",
                  "corner": "corner", "goal_kick": "goal_kick", "penalty": "penalty"}


def parse_minute(m):
    """"45'+2'" -> (45, 2); "62" -> (62, 0)."""
    txt = str(m).replace("'", "").replace("’", "").strip()
    base, _, add = txt.partition("+")
    return int(float(base)), int(float(add)) if add.strip() else 0


def minute_frames(m, cfg, fps):
    """(first, last) video frame of a published match minute: minute N is clock (N-1):00 to N:00,
    added time runs on from the end of the period. Each period's clock starts at its kick-off."""
    try:
        base, add = parse_minute(m)
    except ValueError:
        return None
    sync = cfg.get("clock_sync") or {}
    origins = [(int(sync["kickoff_frame"]), 0.0)] if sync.get("kickoff_frame") is not None else []
    for p in (cfg.get("periods") or {}).values():
        if p.get("kickoff_frame") is not None and p.get("clock_start_s") is not None:
            origins.append((int(p["kickoff_frame"]), float(p["clock_start_s"])))
    if not origins:
        return None
    origins.sort(key=lambda o: o[1])
    clock_s = (base - 1 + add) * 60.0
    # the period this minute belongs to: the last one whose clock has started by `base` (not counting added time)
    frame0, start_s = [o for o in origins if o[1] <= (base - 1) * 60.0][-1] if any(
        o[1] <= (base - 1) * 60.0 for o in origins) else origins[0]
    f0 = frame0 + (clock_s - start_s) * fps
    return int(f0), int(f0 + 60 * fps)


def best_attack_moment(timeline, team, goal_x, lo, hi):
    """Frame in [lo, hi] where `team` has the ball closest to the goal it attacks (else None)."""
    t = timeline[(timeline["frame"] >= lo) & (timeline["frame"] <= hi) & (timeline["state"] == team)
                 & timeline["ball_x"].notna()]
    if t.empty:
        return None
    return int(t.loc[(t["ball_x"] - goal_x).abs().idxmin(), "frame"])


def goal_moment(kickoffs, stops, win, fps, after_s=180.0, join_s=6.0):
    """A goal is followed by a long stoppage (celebration) and a kick-off. The first kick-off
    after the goal's minute starts (within after_s of the minute's end): walk back over the
    stoppages before it, joined when less than join_s apart, to where play stopped. That is
    about when the ball went in. None if there's no kick-off."""
    ko = [k for k in kickoffs if win[0] <= k <= win[1] + after_s * fps]
    if not ko:
        return None
    k = ko[0]
    before = sorted((s, e) for s, e in stops if e <= k + 2 * fps)
    if not before:
        return None
    start = before[-1][0]
    for s, e in reversed(before[:-1]):
        if start - e > join_s * fps:
            break
        start = s
    return int(start) if start >= win[0] - 60 * fps else None


def apply_published(items, events, cfg, fps, timeline, stops, goal_at, slack_s=30.0, kickoffs=()):
    """Tie published events to the detected clips. A detected clip of the same team (and kind,
    for set pieces) inside the event's minute (+- slack) gets the published details; an event
    with no match becomes a clip of its own, at the best moment in its minute: the team's
    nearest approach to goal for a shot, the end of a stoppage for a set piece.
    Returns (items, report)."""
    used, report = set(), {"matched": 0, "added": 0, "skipped": 0}
    for ev in events or []:
        kind = PUBLISHED_KIND.get(ev.get("type"))
        team = ev.get("team")
        win = minute_frames(ev.get("minute", ""), cfg, fps) if kind else None
        if not win or team not in TEAMS:
            report["skipped"] += 1
            continue
        lo, hi = win[0] - int(slack_s * fps), win[1] + int(slack_s * fps)
        mid = (win[0] + win[1]) // 2
        group = GROUP_OF[kind]
        ok_kinds = ("shot", "box_entry") if group == "chances" else (kind,)
        cand = [i for i, it in enumerate(items) if i not in used and lo <= it["frame"] <= hi
                and it["kind"] in ok_kinds and it.get("team") in (team, None)]
        if not cand and group == "set_pieces":  # a restart of that team, labelled otherwise
            cand = [i for i, it in enumerate(items) if i not in used and lo <= it["frame"] <= hi
                    and GROUP_OF[it["kind"]] == "set_pieces" and it.get("team") in (team, None)]
        pub = {k: ev[k] for k in ("minute", "type", "player", "outcome", "detail", "source") if ev.get(k)}
        gm = goal_moment(kickoffs, stops, win, fps) if ev.get("type") == "goal" else None
        if gm is not None:
            # play only stops after the celebration (~45-65 s on), so the goal is the scoring team's
            # last chance 20-90 s before the stop
            prior = [i for i, it in enumerate(items) if i not in used and it.get("team") in (team, None)
                     and it["kind"] in ("shot", "box_entry") and gm - 90 * fps <= it["frame"] <= gm - 20 * fps]
            how = "goal: last chance before play stopped for the celebration"
            if prior:
                i = max(prior, key=lambda j: items[j]["frame"])
                items[i].update(kind=kind, team=team, published=pub, how=how)
                used.add(i)
            else:
                frame = gm - int(45 * fps)
                items.append({"frame": frame, "kind": kind, "team": team, "first": frame, "lead_s": 0.0,
                              "how": "goal: about 45 s before play stopped for the celebration", "published": pub})
                used.add(len(items) - 1)
            report["matched"] += 1
            continue
        if cand:
            i = min(cand, key=lambda j: abs(items[j]["frame"] - mid))
            items[i].update(kind=kind, team=team, published=pub)
            used.add(i)
            report["matched"] += 1
            continue
        if group == "chances":
            frame = best_attack_moment(timeline, team, goal_at(mid)[team], win[0], win[1]) or mid
            it = {"frame": frame, "kind": kind, "team": team, "first": frame, "lead_s": 0.0}
        else:
            ends = [e for _, e in stops if win[0] - 10 * fps <= e <= win[1] + 10 * fps]
            frame = min(ends, key=lambda e: abs(e - mid)) if ends else mid
            it = {"frame": int(frame), "kind": kind, "team": team}
        it.update(how="published minute; moment estimated", published=pub)
        items.append(it)
        used.add(len(items) - 1)
        report["added"] += 1
    return items, report


# --------------------------------------------------------------------------
def clip_window(kind, frame, fps, lo, hi, first=None, lead_cap_s=8.0):
    pre, post = WINDOWS[kind]
    start = frame - int(pre * fps)
    if first is not None:  # a chance that built up in the box: start before it got there
        start = min(start, max(first - int(2 * fps), frame - int(lead_cap_s * fps) - int(2 * fps)))
    return max(lo, start), min(hi, frame + int(post * fps))


def build(items, fps, lo, hi, cfg, name_of):
    clips = []
    for it in items:
        s, e = clip_window(it["kind"], it["frame"], fps, lo, hi, it.get("first"))
        if (it.get("published") or {}).get("type") == "goal":  # the build-up, the goal, the celebration
            s, e = max(lo, min(s, it["frame"] - int(10 * fps))), min(hi, it["frame"] + int(8 * fps))
        c = {"id": f"{it['kind']}-{it['frame']}", "group": GROUP_OF[it["kind"]], "kind": it["kind"],
             "team": it.get("team"), "frame": it["frame"], "start": s, "end": e,
             "clock": clock_label(it["frame"], cfg, fps) or ""}
        for k in ("x", "y", "end_x", "end_y", "how", "player", "to_player", "end_frame", "lead_s", "published"):
            if it.get(k) is not None:
                c[k] = it[k]
        if it.get("player") is not None and name_of.get(it["player"]):
            c["player_name"] = name_of[it["player"]]
        if (it.get("published") or {}).get("player"):  # the published name beats the tracker's guess
            c["player_name"] = it["published"]["player"]
        if it.get("to_player") is not None and name_of.get(it["to_player"]):
            c["to_player_name"] = name_of[it["to_player"]]
        clips.append(c)
    clips.sort(key=lambda c: c["frame"])
    return clips


def wide_view_lookup(tracks, frac=0.65):
    """frame -> True in zoomed-out wide views (median player box under frac x the match's)."""
    h = (tracks["y2"] - tracks["y1"]).groupby(tracks["frame"]).median()
    wide = h < frac * h.median()
    wf = set(wide.index[wide.to_numpy()].tolist())
    return lambda f: any((f + d) in wf for d in (-15, 0, 15))


def identified_rows(tracks, pmap):
    """Identified players' rows (frame order) with team and a keeper flag."""
    team, gk = {}, set()
    for p in pmap.get("players", {}).values():
        for t in p.get("track_ids", []):
            team[int(t)] = p.get("team")
            if p.get("role") == "GK":
                gk.add(int(t))
    r = tracks[tracks["track_id"].isin(list(team)) & tracks["pitch_x"].notna()]
    r = r.assign(team=r["track_id"].map(team), gk=r["track_id"].isin(list(gk)))
    return r[r["team"].isin(TEAMS)].sort_values("frame").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--stop-speed", type=float, default=1.3, help="Stoppage: players' median speed under this (m/s)")
    ap.add_argument("--stop-s", type=float, default=3.0, help="... for at least this long (s)")
    ap.add_argument("--still-s", type=float, default=1.5,
                    help="A dead ball outside a stoppage must be still this long (s)")
    ap.add_argument("--third-m", type=float, default=35.0, help="Final third: within this of the goal line (m)")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--tmp-dir", default=".tmp")
    a = ap.parse_args()

    t0 = time.time()
    meta = load_meta(a.match_id, a.tmp_dir)
    mdir = match_dir(a.match_id, a.tmp_dir)
    fps = float(meta["fps"])
    cfg = match_config(a.match_id, a.tmp_dir, a.config_dir)
    L, W = pitch_dims(load_calibration(a.match_id, a.config_dir, a.calibration))
    for need in ("possession_timeline.parquet", "ball_track.parquet", "tracks.parquet"):
        if not (mdir / need).exists():
            raise SystemExit(f"{need} not found: run track_ball.py and compute_possession.py first")
    timeline = pd.read_parquet(mdir / "possession_timeline.parquet").sort_values("frame")
    ball = pd.read_parquet(mdir / "ball_track.parquet", columns=["frame", "pitch_x", "pitch_y", "interp"])
    ball = ball[~ball["interp"].astype(bool) & ball["pitch_x"].notna()].drop_duplicates("frame")
    ball = ball.set_index("frame").sort_index()[["pitch_x", "pitch_y"]]
    tracks = pd.read_parquet(mdir / "tracks.parquet", columns=["frame", "track_id", "class", "pitch_x", "pitch_y", "y1", "y2"])
    tracks = tracks[tracks["class"] == "player"]
    pmap = load_player_map(a.match_id, a.tmp_dir) or {}
    team_of, name_of = {}, {}
    for p in pmap.get("players", {}).values():
        for t in p.get("track_ids", []):
            team_of[int(t)] = p.get("team")
            if p.get("name"):
                name_of[int(t)] = f"{p['number']} {p['name']}" if p.get("number") is not None else p["name"]

    lo, hi = int(timeline["frame"].min()), int(timeline["frame"].max())
    gp = goals_by_period(tracks, pmap, L, period_ranges(cfg, lo, hi))
    goal_at = goal_lookup(gp)
    names = {t: cfg.get(f"{t}_name") or t for t in TEAMS}
    for s, e, g in gp:
        print(f"[{a.match_id}] frames {s}-{e}: {names['team_a']} attack x={g['team_a']:.0f}, "
              f"{names['team_b']} x={g['team_b']:.0f}", flush=True)

    speed = player_speed(tracks, team_of, fps)
    players = identified_rows(tracks, pmap)
    restarts, stops, n_clutter, kickoffs = find_restarts(ball, speed, timeline, team_of, goal_at, fps, L, W, a, players,
                                                 wide_view_lookup(tracks))
    print(f"  {len(stops)} stoppages, {n_clutter} clutter spot(s) ignored -> {len(restarts)} set pieces", flush=True)

    spells = control_spells(timeline, team_of, fps=fps)
    ev = events_by_period(spells, ball, gp, fps, L, W)
    ch = chances(ev, box_entries(timeline, goal_at, fps, L, W), fps)
    ft = final_third_passes(ev, goal_at, a.third_m)

    items = restarts + ch
    ref = {}
    rp = reference_path(a.match_id, a.config_dir)
    if rp.exists():
        ref = json.loads(rp.read_text(encoding="utf-8"))
    pub_report = None
    if ref.get("match_events"):
        items, pub_report = apply_published(items, ref["match_events"], cfg, fps, timeline, stops, goal_at,
                                            kickoffs=kickoffs)
        print(f"  published events ({len(ref['match_events'])}): {pub_report['matched']} matched a detected clip, "
              f"{pub_report['added']} added at an estimated moment, {pub_report['skipped']} skipped", flush=True)
    clips = build(items + ft, fps, lo, hi, cfg, name_of)
    counts = {t: {k: sum(1 for c in clips if c["team"] == t and c["kind"] == k) for k in GROUP_OF} for t in TEAMS}
    counts["unknown_team"] = sum(1 for c in clips if c["team"] not in TEAMS)
    out = {"match_id": a.match_id, "fps": fps, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "teams": names, "groups": {g: list(k) for g, k in GROUPS.items()}, "windows_s": WINDOWS,
           "params": {"stop_speed": a.stop_speed, "stop_s": a.stop_s, "still_s": a.still_s, "third_m": a.third_m},
           "counts": counts, "published": pub_report, "clips": clips}
    write_json(mdir / "clips.json", out)
    for t in TEAMS:
        c = counts[t]
        print(f"  {names[t]:<12} goal kicks {c['goal_kick']:>3}  throw-ins {c['throw_in']:>3}  corners {c['corner']:>3}  "
              f"free kicks {c['free_kick']:>3}  penalties {c['penalty']:>2}  shots {c['shot']:>3}  "
              f"box entries {c['box_entry']:>3}  final-third passes {c['pass']:>3}")
    if counts["unknown_team"]:
        print(f"  {counts['unknown_team']} clip(s) with no team (shown under both)")
    print(f"Wrote {mdir / 'clips.json'}: {len(clips)} clips ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
