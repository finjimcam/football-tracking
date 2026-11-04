"""Verify the identity tracking against itself and against the real match
-> .tmp/<id>/verification.json, outputs/verification/<id>/...

Stage 6 of workflows/track_identities.md - the "does it match?" pass. Reads
the artifacts written by track_identities.py / track_ball.py /
compute_possession.py plus config/<id>_reference.json (published Opta
events). Nothing here changes the tracking; it scores it.

Internal checks (no ground truth needed):
  1. Coverage per player: share of the window each identity is observed,
     number of tracklets, longest unobserved stretch.
  2. Hand-off plausibility: every link between two tracklets of one identity
     implies a speed (distance / gap). > 8 m/s sustained is not a footballer.
  3. Link stability: share of links that survived every perturbed re-solve
     in track_identities.py (fragile links = where a swap is plausible).
  4. Kit consistency: share of each identity's readable detections whose
     learned team colour agrees with the identity's team.
External checks (published Opta events, timestamps -> frames via the
scoreboard clock sync in the match config):
  5. Possession at events: shots/goals/offsides -> the shooting/attacking
     team must hold the ball just before; fouls won -> the fouled team has it
     just after. Scored on possession_timeline.parquet.
  6. Named players: for every event with a named player, which identity is
     nearest the ball in [t-2 s, t+1 s]? A hit if it carries that player's
     (role-inferred) name. Hits validate identity AND the name mapping; a
     miss can be either (or a ball the detector didn't see).
Review material:
  7. Contact sheet per player: crops sampled evenly through the half, in
     time order with the match clock - an identity swap shows up as a
     different-looking person (hair, skin, boots, sleeves) mid-sheet.
  8. uncertain_moments.csv: fragile and long-gap links with match clock and
     video time, i.e. exactly where to scrub the tracking video.

Usage:
    python tools/verify_tracking.py --match-id <id>
    python tools/verify_tracking.py --match-id <id> --no-sheets   # skip video reads
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from _common import (OUTPUTS_DIR, clock_label, load_meta, load_player_map, match_config, match_dir,
                     read_json, reference_path, write_json)


def name_index(pm):
    """track_id -> (player key, display name)."""
    out = {}
    for key, info in pm["players"].items():
        for t in info["track_ids"]:
            out[int(t)] = (key, info.get("name") or key)
    return out


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def role_separability(players, pm, step=5, min_visible=5):
    """How distinct each identity's place in the team shape is. Positions are
    taken relative to the centroid of the team's visible outfield identities
    (sampled every `step` frames, >= min_visible on screen); per identity:
    mean offset mu_k and spread s_k (RMS around it). Separability = mean
    pairwise distance between the mu_k / mean s_k. When identity chains mix,
    every mu_k drifts to the team mean and separability collapses - the
    symptom of swaps that a per-link check can't see."""
    out = {}
    for team in ("team_a", "team_b"):
        ids = [int(v["track_ids"][0]) for v in pm["players"].values()
               if v["team"] == team and v.get("role") != "GK"]
        rows = players[players["track_id"].isin(ids) & (players["frame"] % step == 0)]
        g = rows.groupby("frame")
        cnt = g["pitch_x"].transform("size")
        rel = rows.assign(rx=rows["pitch_x"] - g["pitch_x"].transform("mean"),
                          ry=rows["pitch_y"] - g["pitch_y"].transform("mean"))[cnt >= min_visible]
        mu = rel.groupby("track_id")[["rx", "ry"]].mean()
        dev = rel[["rx", "ry"]].to_numpy() - mu.loc[rel["track_id"]].to_numpy()
        spread = pd.Series(np.hypot(dev[:, 0], dev[:, 1]) ** 2, index=rel["track_id"].to_numpy())
        s = np.sqrt(spread.groupby(level=0).mean())
        m = mu.to_numpy()
        pair = np.hypot(m[:, None, 0] - m[None, :, 0], m[:, None, 1] - m[None, :, 1])
        mean_pair = float(pair[np.triu_indices(len(m), 1)].mean()) if len(m) > 1 else 0.0
        out[team] = {"separability": round(mean_pair / float(s.mean()), 3) if len(s) else None,
                     "mean_pairwise_m": round(mean_pair, 1), "mean_role_spread_m": round(float(s.mean()), 1),
                     "role_spread_m": {int(k): round(float(v), 1) for k, v in s.items()}}
    return out


def contact_sheets(video, tracks, pm, cfg, out_dir, fps, n_crops=24):
    """One image per team: a row per player, crops spread through the half."""
    cap = cv2.VideoCapture(str(video))
    for team in ("team_a", "team_b"):
        rows_img = []
        keys = [k for k, v in pm["players"].items() if v["team"] == team]
        for key in keys:
            tid = pm["players"][key]["track_ids"][0]
            rows = tracks[(tracks["track_id"] == tid) & (tracks["class"] == "player")].sort_values("frame")
            tiles = []
            if len(rows):
                pick = rows.iloc[np.linspace(0, len(rows) - 1, min(n_crops, len(rows))).astype(int)]
                for r in pick.itertuples():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
                    ok, f = cap.read()
                    if not ok:
                        continue
                    h = max(int(r.y2 - r.y1), 12)
                    cx = int((r.x1 + r.x2) / 2)
                    y0, y1 = max(0, int(r.y1) - h // 6), int(r.y2) + h // 6
                    x0, x1 = max(0, cx - h // 2), cx + h // 2
                    crop = cv2.resize(f[y0:y1, x0:x1], (48, 80), interpolation=cv2.INTER_CUBIC)
                    lab = clock_label(int(r.frame), cfg, fps) or str(int(r.frame))
                    cv2.putText(crop, lab, (1, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
                    tiles.append(crop)
            while len(tiles) < n_crops:
                tiles.append(np.zeros((80, 48, 3), np.uint8))
            label = np.zeros((80, 130, 3), np.uint8)
            info = pm["players"][key]
            cv2.putText(label, f"{info.get('name') or key}", (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(label, f"#{info['number']} cov {info['coverage']:.0%}", (3, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            rows_img.append(np.hstack([label] + tiles))
        sheet = np.vstack(rows_img)
        path = out_dir / f"contact_sheet_{team}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  wrote {path}")
    cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--vmax", type=float, default=8.0, help="Flag hand-offs implying more than this (m/s)")
    parser.add_argument("--long-gap-s", type=float, default=10.0, help="Links bridging longer gaps are listed")
    parser.add_argument("--ball-radius-m", type=float, default=3.0, help="Named-player check: max distance to ball")
    parser.add_argument("--no-sheets", action="store_true", help="Skip contact sheets (no video reads)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    fps = meta["fps"]
    cfg = match_config(args.match_id, args.tmp_dir, args.config_dir)
    ref = read_json(reference_path(args.match_id, args.config_dir))
    mdir = match_dir(args.match_id, args.tmp_dir)
    pm = load_player_map(args.match_id, args.tmp_dir)
    if pm is None:
        raise SystemExit("player_map.json missing - run tools/track_identities.py first")
    tracks = pd.read_parquet(mdir / "tracks.parquet")
    links = pd.read_parquet(mdir / "identity_links.parquet")
    timeline = pd.read_parquet(mdir / "possession_timeline.parquet")
    names = name_index(pm)
    f_lo, f_hi = pm["window"]["start_frame"], pm["window"]["end_frame"]
    n_frames = f_hi - f_lo + 1
    kickoff = cfg["clock_sync"]["kickoff_frame"]
    out_dir = OUTPUTS_DIR / "verification" / args.match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-3. per-player continuity, hand-offs, stability
    players = tracks[(tracks["class"] == "player") & tracks["track_id"].isin(names)]
    rows, moments = [], []
    for key, info in pm["players"].items():
        tid = info["track_ids"][0]
        seq = links[links["identity"] == tid].sort_values("start")
        fr = np.sort(players.loc[players["track_id"] == tid, "frame"].unique())
        gaps = np.diff(np.r_[f_lo - 1, fr, f_hi + 1]) - 1
        bad, n_links = 0, 0
        for prev, nxt in zip(seq.iloc[:-1].itertuples(), seq.iloc[1:].itertuples()):
            gap = max((nxt.start - prev.end) / fps, 1 / fps)
            dist = float(np.hypot(nxt.entry_x - prev.exit_x, nxt.entry_y - prev.exit_y))
            speed = dist / gap
            n_links += 1
            implausible = speed > args.vmax and dist > 3.0
            bad += int(implausible)
            fragile = bool(np.isfinite(nxt.stability) and nxt.stability < 1)
            if implausible or fragile or gap >= args.long_gap_s:
                moments.append({
                    "player": key, "name": info.get("name"), "frame": int(nxt.start),
                    "video_time": f"{int(nxt.start / fps // 60):02d}:{nxt.start / fps % 60:04.1f}",
                    "match_clock": clock_label(int(nxt.start), cfg, fps),
                    "gap_s": round(gap, 1), "dist_m": round(dist, 1), "implied_speed": round(speed, 1),
                    "stability": None if not np.isfinite(nxt.stability) else round(float(nxt.stability), 2),
                    "why": ", ".join(w for w, c in (("implausible speed", implausible), ("fragile", fragile),
                                                     ("long gap", gap >= args.long_gap_s)) if c)})
        team = info["team"]
        mine = tracks[(tracks["track_id"] == tid) & tracks["p_team_a"].notna()] if "p_team_a" in tracks else None
        kit_ok = None  # keepers wear neither outfield kit - the check doesn't apply
        if mine is not None and len(mine) and info.get("role") != "GK":
            agree = (mine["p_team_a"] > 0.5) if team == "team_a" else (mine["p_team_a"] < 0.5)
            kit_ok = round(float(agree.mean()), 3)
        rows.append({"player": key, "name": info.get("name"), "team": team, "coverage": info["coverage"],
                     "tracklets": info["tracklets"], "links": n_links, "implausible_links": bad,
                     "fragile_links": info.get("fragile_links", 0),
                     "longest_unseen_s": round(float(gaps.max()) / fps, 1) if len(gaps) else None,
                     "kit_consistency": kit_ok})
    per_player = pd.DataFrame(rows)
    pd.DataFrame(moments).sort_values("frame").to_csv(out_dir / "uncertain_moments.csv", index=False)

    # 5. possession at Opta events: which team had the ball WHEN it happened.
    # Shots/goals: the shooter's team just before; offsides: the attacking
    # team as the pass was played; fouls: committed against the team on the
    # ball; corners: the attacking spell that won it. (An earlier version
    # scored fouls/corners in the dead ball AFTER the event, where the
    # fouling defender is usually nearest the ball - unfair to the tracker.)
    windows = {"shot": (-2.0, -0.2), "goal": (-3.0, -0.5), "offside": (-4.0, -0.5),
               "foul_won": (-2.0, 0.0), "corner_won": (-6.0, -1.5)}
    tl = timeline.set_index("frame")["state"]
    ev_rows = []
    for ev in ref.get("first_half_events", []):
        f = kickoff + int(round(ev["s"] * fps))
        w0, w1 = windows.get(ev["type"], (-3.0, -0.5))
        a, b = max(f + int(w0 * fps), f_lo), min(f + int(w1 * fps), f_hi)
        st = tl.loc[a:b] if b > a else tl.iloc[:0]
        known = st[st != "none"]
        share = float((known == ev["team"]).mean()) if len(known) else None
        ev_rows.append({"s": ev["s"], "clock": f"{ev['s'] // 60:02d}:{ev['s'] % 60:02d}", "type": ev["type"],
                        "team": ev["team"], "player": ev.get("player"), "frame": f,
                        "possession_share": None if share is None else round(share, 2),
                        "possession_ok": None if share is None else bool(share >= 0.5)})

    # 6. named player on the ball: the event team's identified player nearest
    # the ball at the LAST ball sighting before the event (the shot, the
    # touch that got fouled). Opponents are excluded - at a shot the nearest
    # body is often the blocking defender. Offsides are skipped: the named
    # player is the pass receiver, nowhere near the ball.
    named_windows = {"shot": (-3.0, -0.1), "goal": (-4.0, -0.3), "foul_won": (-2.0, 0.3)}
    balls = tracks[tracks["class"] == "ball"]
    if "interp" in balls.columns:
        balls = balls[~balls["interp"].astype("boolean").fillna(False).to_numpy(dtype=bool)]
    team_of = {t: pm["players"][k]["team"] for t, (k, _) in names.items()}
    for ev, row in zip(ref.get("first_half_events", []), ev_rows):
        row["nearest_to_ball"] = row["named_player_hit"] = row["named_player_rank"] = None
        if not ev.get("player") or ev["type"] not in named_windows:
            continue
        f = row["frame"]
        w0, w1 = named_windows[ev["type"]]
        win_b = balls[(balls["frame"] >= f + int(w0 * fps)) & (balls["frame"] <= f + int(w1 * fps))]
        target = [t for t, (_, nm) in names.items() if norm(nm) == norm(ev["player"])]
        for b in win_b.sort_values("frame", ascending=False).itertuples():
            pl = players[(players["frame"] == b.frame)
                         & players["track_id"].map(lambda t: team_of.get(int(t)) == ev["team"])]
            if pl.empty:
                continue
            d = np.hypot(pl["pitch_x"] - b.pitch_x, pl["pitch_y"] - b.pitch_y).to_numpy()
            if d.min() > args.ball_radius_m:
                continue
            order = np.argsort(d)
            tids = pl["track_id"].to_numpy()[order]
            row["nearest_to_ball"] = names[int(tids[0])][1]
            row["named_player_hit"] = bool(target and int(tids[0]) in target)
            ranks = [k + 1 for k, t in enumerate(tids) if target and int(t) in target]
            row["named_player_rank"] = ranks[0] if ranks else None  # None = not on screen
            break
    events = pd.DataFrame(ev_rows)
    events.to_csv(out_dir / "event_checks.csv", index=False)

    poss_scored = events["possession_ok"].dropna()
    named = events.dropna(subset=["named_player_hit"]) if "named_player_hit" in events else events.iloc[:0]
    sep = role_separability(players, pm)
    for team, s in sep.items():
        for tid, v in s.pop("role_spread_m").items():
            per_player.loc[per_player["player"] == names[tid][0], "role_spread_m"] = v
    summary = {
        "role_separability": sep,
        "window": {"start_frame": f_lo, "end_frame": f_hi, "minutes": round(n_frames / fps / 60, 1)},
        "identified_detection_fraction": pm.get("identified_detection_fraction"),
        "link_stability": pm.get("link_stability"),
        "mean_outfield_coverage": round(float(per_player.loc[~per_player["player"].isin(
            [k for k, v in pm["players"].items() if v.get("role") == "GK"]), "coverage"].mean()), 3),
        "implausible_links": int(per_player["implausible_links"].sum()),
        "total_links": int(per_player["links"].sum()),
        "mean_kit_consistency": (round(float(per_player["kit_consistency"].dropna().mean()), 3)
                                 if per_player["kit_consistency"].notna().any() else None),
        "events_possession_agree": f"{int(poss_scored.sum())}/{len(poss_scored)}",
        "events_named_player_nearest_ball": (f"{int(named['named_player_hit'].sum())}/{len(named)}"
                                             if len(named) else "0/0"),
        "events_named_player_near_ball_at_all": (f"{int((named['named_player_rank'] <= 2).sum())}/{len(named)}"
                                                 if len(named) else "0/0"),
        "uncertain_moments": len(moments),
        "per_player": per_player.to_dict(orient="records"),
        "events": events.where(events.notna(), None).to_dict(orient="records"),
    }
    write_json(mdir / "verification.json", summary)
    per_player.to_csv(out_dir / "identity_report.csv", index=False)

    if not args.no_sheets:
        contact_sheets(meta["video"], tracks, pm, cfg, out_dir, fps)

    print(f"[{args.match_id}] verification over {summary['window']['minutes']} min")
    print(f"  identified detections {summary['identified_detection_fraction']:.0%}, mean outfield coverage "
          f"{summary['mean_outfield_coverage']:.0%}, link stability {summary['link_stability']}")
    print(f"  hand-offs implying > {args.vmax:g} m/s: {summary['implausible_links']}/{summary['total_links']}; "
          f"kit consistency {summary['mean_kit_consistency']}")
    print(f"  Opta events - possession agrees: {summary['events_possession_agree']}; named player is our "
          f"player on the ball: {summary['events_named_player_nearest_ball']} (among the team's 2 nearest: "
          f"{summary['events_named_player_near_ball_at_all']})")
    print("  role separability (distinct places in the team shape; collapses when identities mix): "
          + ", ".join(f"{t} {s['separability']} (pairwise {s['mean_pairwise_m']} m / spread "
                      f"{s['mean_role_spread_m']} m)" for t, s in sep.items()))
    print(f"  {len(moments)} uncertain moments -> {out_dir / 'uncertain_moments.csv'}")
    print(per_player[["player", "name", "coverage", "tracklets", "fragile_links", "implausible_links",
                      "longest_unseen_s", "kit_consistency", "role_spread_m"]].to_string(index=False))


if __name__ == "__main__":
    main()
