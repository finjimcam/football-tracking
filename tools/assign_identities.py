"""Propagate the labeled roster (config/<match_id>_roster.json) across all
player tracks -> .tmp/<match_id>/player_map.json (track_id -> player).

The core idea: instead of treating every track_id as its own player (a 10 s
clip easily produces 40+ IDs), there are at most 11 players per team — the
roster the user labeled with label_roster.py. Each roster player is a slot;
tracks get assigned to slots and everything left over stays unidentified
(referees, spectators, detector noise).

Curation (v2): tracks the user ignored (`x` in label_roster.py) plus tracks
whose feet spend most of the video OUTSIDE the pitch (--outside-frac beyond
--pitch-margin-m, via the calibration) land in player_map.json's
"ignored_tracks" — downstream compute tools exclude them from ALL stats, so
a spectator clustered into a team color can't inflate team distance. A label
or a rescue click always overrides the auto filter. Labels flagged "manual"
(drawn boxes) resolve against the source="manual" rows retrack_manual.py
creates — run that first if the roster has manual boxes.

How assignment works:
1. Seed: each anchor click is re-resolved against the CURRENT tracks.parquet
   (clicks are stored as frame + image position, so re-running
   stitch_tracks.py never invalidates the roster). The clicked track carries
   that player's identity for its whole lifetime.
2. Propagate: repeatedly take the globally best (player, unlabeled track)
   match and assign it. A candidate match extrapolates the player's position
   from the end of their previous track (or backwards from the start of
   their next track — so a mid-video anchor also identifies the player's
   earlier tracks) and gates on:
     - time gap <= --max-gap-s,
     - predicted-position distance <= base + growth x gap (pitch meters when
       calibration exists — robust under a panning camera — else pixels
       scaled by median box height),
     - team consistency: a track that assign_teams.py confidently put on the
       other team can't take this player's slot (skipped if no team_map).
   A player can never hold two tracks at the same time, and a track can
   never split between players.

Artifact-only and cheap — no video access, safe to re-run and tune freely.
Run AFTER stitch_tracks.py (fewer, longer fragments to assign) and ideally
after assign_teams.py + calibration. Rewrites player_map.json from scratch
each run.

Sanity-check the coverage report it prints, then watch the rendered video
(tools/render_tracking_video.py): if a player's identity jumps to the wrong
body, add an anchor on a frame after the jump (label_roster.py appends) and
re-run this tool.

Usage:
    python tools/assign_identities.py --match-id <match_id>
    python tools/assign_identities.py --match-id <match_id> --max-gap-s 20
"""
import argparse
from collections import defaultdict

import numpy as np

from _common import (apply_calibration, load_calibration, load_meta, load_roster,
                     load_tracks, load_team_map, match_dir, pitch_dims,
                     resolve_click_to_track, write_json)

EDGE_WINDOW = 3   # rows to average for a track's entry/exit position
VEL_WINDOW = 5    # rows to fit entry/exit velocity over


def track_profile(grp):
    """Entry/exit state of one track (positions in pos_x/pos_y space)."""
    xy = grp[["pos_x", "pos_y"]].to_numpy()
    t = grp["t_s"].to_numpy()
    w = min(VEL_WINDOW, len(grp))
    dt_exit = t[-1] - t[-w]
    dt_entry = t[w - 1] - t[0]
    return {
        "start_frame": int(grp["frame"].iloc[0]),
        "end_frame": int(grp["frame"].iloc[-1]),
        "start_t": float(t[0]), "end_t": float(t[-1]),
        "entry": xy[:EDGE_WINDOW].mean(axis=0),
        "exit": xy[-EDGE_WINDOW:].mean(axis=0),
        "entry_vel": (xy[w - 1] - xy[0]) / dt_entry if dt_entry > 0 else np.zeros(2),
        "exit_vel": (xy[-1] - xy[-w]) / dt_exit if dt_exit > 0 else np.zeros(2),
        "n": len(grp),
    }


def capped(vel, vmax):
    speed = float(np.hypot(*vel))
    return vel * (vmax / speed) if speed > vmax else vel


def resolve_clicks(players, roster):
    """Anchor clicks -> ({track_id: label}, ignored ids, rescued ids, warnings).

    Labels flagged "manual" only resolve against source="manual" rows (the
    tracks retrack_manual.py built from drawn boxes) — never to a nearby
    automatic box, which is exactly the wrong person."""
    seeds, ignored, rescued, warnings = {}, set(), set(), []
    has_source = "source" in players.columns
    for anchor in sorted(roster["anchors"], key=lambda a: a["frame"]):
        rows = players[players["frame"] == anchor["frame"]]
        if rows.empty:
            near = players["frame"].unique()
            alt = int(min(near, key=lambda f: abs(int(f) - anchor["frame"])))
            if abs(alt - anchor["frame"]) > 10:
                warnings.append(f"anchor frame {anchor['frame']}: no tracked boxes within 10 frames — skipped")
                continue
            rows = players[players["frame"] == alt]
        for lab in anchor["labels"]:
            cand = rows
            if lab.get("manual"):
                cand = rows[rows["source"] == "manual"] if has_source else rows.iloc[0:0]
                if cand.empty:
                    warnings.append(f"anchor frame {anchor['frame']}: manual-box label {lab['key']} "
                                    "has no manual track — run tools/retrack_manual.py first")
                    continue
            row = resolve_click_to_track(cand, lab["image"])
            if row is None:
                warnings.append(f"anchor frame {anchor['frame']}: no box near click for "
                                f"{lab['key']} — label lost (track may have been dropped as a blip)")
                continue
            tid = int(row["track_id"])
            if tid in seeds and seeds[tid]["key"] != lab["key"]:
                warnings.append(f"track #{tid} labeled both {seeds[tid]['key']} and {lab['key']} "
                                f"(anchor frame {anchor['frame']}) — kept {seeds[tid]['key']}; this usually "
                                "means an identity switch inside one track: check the rendered video")
                continue
            seeds[tid] = lab
        for field, sink in (("ignores", ignored), ("rescues", rescued)):
            for click in anchor.get(field, []):
                row = resolve_click_to_track(rows, click["image"])
                if row is None:
                    warnings.append(f"anchor frame {anchor['frame']}: {field[:-1]} click resolved "
                                    "to no box — skipped")
                    continue
                sink.add(int(row["track_id"]))
    return seeds, ignored, rescued, warnings


def overlaps(a_start, a_end, b_start, b_end):
    return a_start <= b_end and b_start <= a_end


def greedy_fill(profiles, seeds, team_map, max_gap_s, base, growth, vmax):
    """Assign unlabeled tracks to roster players, globally best match first.
    Returns {track_id: player_key} covering seeds + propagated tracks."""
    assign = {tid: lab["key"] for tid, lab in seeds.items()}
    player_team = {lab["key"]: lab["team"] for lab in seeds.values()}
    intervals = defaultdict(list)  # key -> [(start_t, end_t, tid)]
    for tid, lab in seeds.items():
        intervals[lab["key"]].append((profiles[tid]["start_t"], profiles[tid]["end_t"], tid))

    def candidate(key, tid):
        p = profiles[tid]
        track_team = team_map.get(tid, "unknown")
        if track_team != "unknown" and track_team != player_team[key]:
            return None
        ivs = intervals[key]
        if any(overlaps(p["start_t"], p["end_t"], s, e) for s, e, _ in ivs):
            return None
        best = None
        prev = max((iv for iv in ivs if iv[1] <= p["start_t"]), key=lambda iv: iv[1], default=None)
        if prev is not None:
            gap = p["start_t"] - prev[1]
            if gap <= max_gap_s:
                q = profiles[prev[2]]
                predicted = q["exit"] + capped(q["exit_vel"], vmax) * gap
                score = float(np.hypot(*(predicted - p["entry"])))
                if score <= base + growth * gap:
                    best = (score, gap)
        nxt = min((iv for iv in ivs if iv[0] >= p["end_t"]), key=lambda iv: iv[0], default=None)
        if nxt is not None:
            gap = nxt[0] - p["end_t"]
            if gap <= max_gap_s:
                q = profiles[nxt[2]]
                predicted = q["entry"] - capped(q["entry_vel"], vmax) * gap
                score = float(np.hypot(*(predicted - p["exit"])))
                if score <= base + growth * gap and (best is None or score < best[0]):
                    best = (score, gap)
        return best

    n_propagated = 0
    while True:
        pool = [(cand[0], key, tid)
                for tid in profiles if tid not in assign
                for key in player_team
                if (cand := candidate(key, tid)) is not None]
        if not pool:
            break
        _, key, tid = min(pool)
        assign[tid] = key
        intervals[key].append((profiles[tid]["start_t"], profiles[tid]["end_t"], tid))
        n_propagated += 1
    return assign, n_propagated


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--roster", default=None, help="Default: config/<match_id>_roster.json")
    parser.add_argument("--max-gap-s", type=float, default=12.0,
                        help="Longest disappearance (seconds) a player identity can bridge")
    parser.add_argument("--base-m", type=float, default=3.0,
                        help="Calibrated: base match radius in meters")
    parser.add_argument("--growth-m", type=float, default=2.0,
                        help="Calibrated: radius growth per second of gap")
    parser.add_argument("--vmax-m", type=float, default=7.0,
                        help="Calibrated: cap on extrapolation speed, m/s")
    parser.add_argument("--calibration", default=None,
                        help="Calibration file (default: config/<match_id>_calibration.json)")
    parser.add_argument("--pitch-margin-m", type=float, default=2.0,
                        help="Feet may project this far outside the pitch before counting as off-pitch")
    parser.add_argument("--outside-frac", type=float, default=0.6,
                        help="Tracks off-pitch more than this fraction of their life are auto-ignored")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    roster = load_roster(args.match_id, args.config_dir, args.roster)
    if roster is None:
        raise SystemExit(f"No roster found for {args.match_id} — run tools/label_roster.py first")

    tracks = load_tracks(args.match_id, args.tmp_dir)
    calib = load_calibration(args.match_id, args.config_dir, args.calibration)
    tracks, unit = apply_calibration(tracks, calib, out_cols=("pos_x", "pos_y"))
    players = tracks[(tracks["class"] == "player") & (tracks["track_id"] >= 0)]
    if players.empty:
        raise SystemExit("No player tracks in tracks.parquet — run tools/run_tracking.py first")

    try:
        team_map = load_team_map(args.match_id, args.tmp_dir)
    except SystemExit:
        team_map = {}
        print("No team_map.json — proceeding without the team-consistency gate "
              "(run tools/assign_teams.py for better results)")

    if unit == "m":
        base, growth, vmax = args.base_m, args.growth_m, args.vmax_m
    else:
        h = float((players["y2"] - players["y1"]).median())
        base, growth, vmax = 2.0 * h, 1.5 * h, 5.0 * h
        print(f"No calibration — pixel thresholds from median box height {h:.0f}px: "
              f"base {base:.0f}px, growth {growth:.0f}px/s, vmax {vmax:.0f}px/s")

    seeds, user_ignored, rescued, warnings = resolve_clicks(players, roster)
    for w in warnings:
        print(f"WARNING: {w}")
    if not seeds:
        raise SystemExit("No anchor label could be resolved to a track — check the roster/anchors")

    # Curation: user ignores + the whole-track pitch-boundary filter. A label
    # or rescue click always wins over both.
    auto_ignored = set()
    if unit == "m":
        length, width = pitch_dims(calib)
        m = args.pitch_margin_m
        outside = (~players["pos_x"].between(-m, length + m)
                   | ~players["pos_y"].between(-m, width + m))
        frac = outside.groupby(players["track_id"]).mean()
        auto_ignored = set(frac[frac > args.outside_frac].index.astype(int))
    overridden = user_ignored & set(seeds)
    for tid in sorted(overridden):
        print(f"WARNING: track #{tid} is both labeled ({seeds[tid]['key']}) and ignored — "
              "the label wins; re-open label_roster.py if that's wrong")
    ignore_reason = {}
    for tid in user_ignored:
        ignore_reason[tid] = "user"
    for tid in auto_ignored - user_ignored - rescued:
        ignore_reason[tid] = "off_pitch"
    for tid in set(seeds) | rescued:
        ignore_reason.pop(tid, None)

    # The roster's a/b letters and assign_teams' team_a/team_b clusters are
    # named independently — align the clusters to the roster (majority vote
    # over seeded tracks) so the team-consistency gate compares like with like.
    alignment = None
    if team_map:
        alignment = "identity"
        agree = sum(1 for tid, lab in seeds.items() if team_map.get(tid) == lab["team"])
        disagree = sum(1 for tid, lab in seeds.items()
                       if team_map.get(tid, "unknown") not in ("unknown", lab["team"]))
        if disagree > agree:
            swap = {"team_a": "team_b", "team_b": "team_a"}
            team_map = {t: swap.get(v, v) for t, v in team_map.items()}
            agree, disagree = disagree, agree
            alignment = "flipped"
            print("Note: team_map.json cluster names were flipped relative to your a/b labels — "
                  "aligned them to the roster (your labeling wins).")
        if agree and disagree / max(agree + disagree, 1) > 0.3:
            print(f"WARNING: {disagree}/{agree + disagree} seeded tracks disagree with the jersey-color "
                  "clustering even after alignment — the team gate may be unreliable (similar kits?)")

    profiles = {int(tid): track_profile(grp.sort_values("frame"))
                for tid, grp in players.groupby("track_id")}
    # Ignored tracks can't take a roster slot — pull them out before matching.
    candidates = {tid: p for tid, p in profiles.items() if tid not in ignore_reason}
    assign, n_propagated = greedy_fill(candidates, seeds, team_map,
                                       args.max_gap_s, base, growth, vmax)

    rows_per_track = players.groupby("track_id").size()
    total_rows = int(rows_per_track.sum())
    labels_by_key = {}
    for lab in (l for a in roster["anchors"] for l in a["labels"]):
        labels_by_key.setdefault(lab["key"], lab)

    players_out = {}
    for key, lab in sorted(labels_by_key.items(), key=lambda kv: (kv[1]["team"], kv[1]["number"])):
        tids = sorted(t for t, k in assign.items() if k == key)
        spans = [profiles[t] for t in tids]
        n_rows = int(sum(rows_per_track[t] for t in tids))
        players_out[key] = {
            "team": lab["team"], "number": lab["number"], "track_ids": tids,
            "detections": n_rows,
            "first_frame": min(s["start_frame"] for s in spans) if spans else None,
            "last_frame": max(s["end_frame"] for s in spans) if spans else None,
            "coverage": round(n_rows / meta["frame_count"], 3),
        }

    # Two seeds with the same key on time-overlapping tracks = two bodies
    # sharing one shirt number (a mislabel or an unnoticed ID switch). The
    # propagation gate can't prevent it — seeds are trusted — so say it loudly.
    for key, p in players_out.items():
        spans = sorted((profiles[t]["start_frame"], profiles[t]["end_frame"], t)
                       for t in p["track_ids"])
        for (s1, e1, t1), (s2, e2, t2) in zip(spans, spans[1:]):
            if s2 <= e1:
                print(f"WARNING: {key} is on two tracks alive at the same time "
                      f"(#{t1} frames {s1}-{e1}, #{t2} frames {s2}-{e2}) — two bodies "
                      "share this number and its stats double-count. Watch the rendered "
                      "video, then re-open label_roster.py on an anchor and d/relabel "
                      "the wrong one.")

    unidentified = [{"track_id": t, "start_frame": profiles[t]["start_frame"],
                     "end_frame": profiles[t]["end_frame"], "detections": int(rows_per_track[t]),
                     "team": team_map.get(t, "unknown")}
                    for t in sorted(profiles) if t not in assign and t not in ignore_reason]
    ignored_out = [{"track_id": t, "reason": reason, "detections": int(rows_per_track[t])}
                   for t, reason in sorted(ignore_reason.items())]
    identified_rows = int(sum(rows_per_track[t] for t in assign))

    out_path = write_json(match_dir(args.match_id, args.tmp_dir) / "player_map.json", {
        "match_id": args.match_id,
        "unit": unit,
        "team_names": roster.get("team_names", {}),
        "team_map_alignment": alignment,
        "track_to_player": {str(t): k for t, k in sorted(assign.items())},
        "players": players_out,
        "unidentified_tracks": unidentified,
        "ignored_tracks": ignored_out,
        "identified_detection_fraction": round(identified_rows / total_rows, 3),
    })

    n_user = sum(1 for r in ignore_reason.values() if r == "user")
    print(f"\n{len(seeds)} seeded track(s), {n_propagated} propagated, "
          f"{len(unidentified)} track(s) left unidentified, "
          f"{len(ignored_out)} ignored ({n_user} by you, "
          f"{len(ignored_out) - n_user} off-pitch) — excluded from all stats")
    print(f"Identified detections: {identified_rows}/{total_rows} "
          f"({identified_rows / total_rows:.0%})")
    print(f"{'player':>8} {'tracks':>6} {'detections':>10} {'coverage':>8}")
    for key, p in players_out.items():
        print(f"{key:>8} {len(p['track_ids']):>6} {p['detections']:>10} {p['coverage']:>8.0%}")
        if not p["track_ids"]:
            print(f"         ^ {key} resolved to no track — its anchor click didn't land on a box")
    big_unknown = [u for u in unidentified if u["detections"] >= 30]
    if big_unknown:
        print(f"\n{len(big_unknown)} substantial unidentified track(s) "
              f"(gray in the rendered video): {[u['track_id'] for u in big_unknown]}")
        print("If one of these is a real player, add an anchor on a frame where they're "
              "tracked (label_roster.py --frames N) and re-run this tool.")
    print(f"\nWrote {out_path}")
    print("Next: python tools/render_tracking_video.py --match-id", args.match_id)


if __name__ == "__main__":
    main()
