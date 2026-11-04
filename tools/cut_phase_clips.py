"""Cut every out-of-possession (or in-possession) phase of one team into a
single video -> outputs/videos/<id>_<team>_<phase>.mp4 + a segment index CSV.

Stage 7 of workflows/track_identities.md. Phases come from
possession_timeline.parquet (tools/compute_possession.py): a team is out of
possession while the timeline's state is the opponent. Phase building:
  - runs of the opponent's possession separated by < --merge-s of anything
    else are merged (a deflection or a 1-second loose ball doesn't end a
    defensive phase);
  - phases shorter than --min-s are dropped (blips aren't 'phases');
  - each phase is padded by --pre-s / --post-s so you see the turnover that
    starts it and the regain that ends it.

The compilation carries a slim bar: phase number, match clock, phase length
and a strip showing where in the half you are. --shape marks every visible
outfield player of the team and joins the back line and midfield line -
found per frame by depth from their own goal, NOT from identities (which
swap between rotating players on broadcast footage) - the quickest way to
read a defensive block. Encodes H.264 (tools/_common.VideoWriter), ~5 MB/min.

Usage:
    python tools/cut_phase_clips.py --match-id <id> --team team_a                 # out of possession
    python tools/cut_phase_clips.py --match-id <id> --team team_a --shape
    python tools/cut_phase_clips.py --match-id <id> --team team_a --phase in_possession
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from _common import (OUTPUTS_DIR, VideoWriter, clock_label, load_meta, load_player_map, match_config,
                     match_dir)

BAR_H = 30


def phases(timeline, team, want, fps, merge_s, min_s, pre_s, post_s, lo, hi):
    """[(start_frame, end_frame, core_start, core_end)] for the wanted phase."""
    other = "team_b" if team == "team_a" else "team_a"
    target = other if want == "out_of_possession" else team
    on = (timeline["state"].to_numpy() == target)
    frames = timeline["frame"].to_numpy()
    runs, start = [], None
    for k, v in enumerate(on):
        if v and start is None:
            start = k
        if (not v or k == len(on) - 1) and start is not None:
            end = k if v else k - 1
            runs.append([frames[start], frames[end]])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_s * fps:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    kept = [r for r in merged if (r[1] - r[0]) / fps >= min_s]
    out = []
    for c0, c1 in kept:
        s, e = max(lo, c0 - int(pre_s * fps)), min(hi, c1 + int(post_s * fps))
        if out and s <= out[-1][1]:
            out[-1][1], out[-1][3] = e, c1
        else:
            out.append([s, e, c0, c1])
    return out


def draw_bar(canvas, text, frac_pos, color):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - BAR_H), (w, h), (20, 20, 20), -1)
    cv2.putText(canvas, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x0, x1 = w - 260, w - 12  # where in the half this phase sits
    cv2.rectangle(canvas, (x0, h - 19), (x1, h - 11), (90, 90, 90), -1)
    cv2.rectangle(canvas, (x0, h - 19), (x0 + int((x1 - x0) * frac_pos), h - 11), color, -1)


def depth_lines(rows, own_goal_x, sizes=(4, 3), max_spread_m=12.0, min_visible=7):
    """IDENTITY-FREE lines: the team's visible outfield players sorted by
    depth from their own goal; the deepest sizes[0] form the back line, the
    next sizes[1] the midfield line (sizes = the formation's first lines).
    A line is returned only if it is coherent right now (depth spread <=
    max_spread_m) and enough outfielders are on screen (>= min_visible) that
    the deepest ones really are the defenders - when the camera only shows
    the attack, no line is drawn rather than a wrong one. Identities aren't
    used because on this footage they swap among rotating players.
    (A first version split on depth gaps; with players spread evenly it
    chained 7-10 players into one zig-zag.)"""
    if len(rows) < min_visible:
        return []
    depth = np.abs(rows["pitch_x"].to_numpy() - own_goal_x)
    order = np.argsort(depth)
    lines, k0 = [], 0
    for size in sizes:
        idx = order[k0:k0 + size]
        k0 += size
        if len(idx) < size:
            break
        lines.append(idx if np.ptp(depth[idx]) <= max_spread_m else None)
    return lines


def draw_shape(canvas, rows, own_goal_x, color, sizes=(4, 3)):
    """Feet ellipses on every team outfield player; back line (team colour)
    and midfield line (white) joined in order ACROSS the pitch (pitch y) - a
    back four runs touchline to touchline, which from a side-on camera is
    not left-to-right in the image."""
    if rows.empty:
        return
    for r in rows.itertuples():
        w = max(int(r.x2 - r.x1), 8)
        cv2.ellipse(canvas, (int(r.foot_x), int(r.foot_y)), (int(0.6 * w), int(0.2 * w)), 0, -35, 235,
                    color, 2, cv2.LINE_AA)
    for k, idx in enumerate(depth_lines(rows, own_goal_x, sizes)):
        if idx is None:
            continue
        grp = rows.iloc[idx].sort_values("pitch_y")
        pts = grp[["foot_x", "foot_y"]].to_numpy().astype(np.int32)
        cv2.polylines(canvas, [pts], False, color if k == 0 else (255, 255, 255), 3, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--team", default="team_a", choices=["team_a", "team_b"])
    parser.add_argument("--phase", default="out_of_possession", choices=["out_of_possession", "in_possession"])
    parser.add_argument("--merge-s", type=float, default=2.0, help="Merge phases separated by less than this")
    parser.add_argument("--min-s", type=float, default=3.0, help="Drop phases shorter than this")
    parser.add_argument("--pre-s", type=float, default=1.5, help="Lead-in before each phase")
    parser.add_argument("--post-s", type=float, default=1.0, help="Run-out after each phase")
    parser.add_argument("--shape", action="store_true", help="Overlay the team's on-screen lines + names")
    parser.add_argument("--crf", type=int, default=26)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    fps = meta["fps"]
    cfg = match_config(args.match_id, args.tmp_dir, args.config_dir)
    mdir = match_dir(args.match_id, args.tmp_dir)
    timeline = pd.read_parquet(mdir / "possession_timeline.parquet")
    lo, hi = int(timeline["frame"].min()), int(timeline["frame"].max())
    segs = phases(timeline, args.team, args.phase, fps, args.merge_s, args.min_s, args.pre_s, args.post_s, lo, hi)
    if not segs:
        raise SystemExit("No phases found - check possession_timeline.parquet")

    names = {"team_a": cfg.get("team_a_name", "team_a"), "team_b": cfg.get("team_b_name", "team_b")}
    other = "team_b" if args.team == "team_a" else "team_a"
    label = (f"{names[args.team].upper()} OUT OF POSSESSION" if args.phase == "out_of_possession"
             else f"{names[args.team].upper()} IN POSSESSION")
    holder = names[other] if args.phase == "out_of_possession" else names[args.team]
    suffix = f"{args.team}_{args.phase}" + ("_shape" if args.shape else "")
    out_path = args.output or (OUTPUTS_DIR / "videos" / f"{args.match_id}_{suffix}.mp4")

    index = []
    total = 0.0
    for k, (s, e, c0, c1) in enumerate(segs, 1):
        dur = (e - s + 1) / fps
        index.append({"phase": k, "start_frame": s, "end_frame": e,
                      "video_start": f"{int(s / fps // 60):02d}:{s / fps % 60:04.1f}",
                      "match_clock_start": clock_label(c0, cfg, fps), "match_clock_end": clock_label(c1, cfg, fps),
                      "phase_s": round((c1 - c0 + 1) / fps, 1), "clip_s": round(dur, 1),
                      "compilation_start": f"{int(total // 60):02d}:{total % 60:04.1f}"})
        total += dur
    idx_path = OUTPUTS_DIR / "stats" / f"{args.match_id}_{args.team}_{args.phase}_phases.csv"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index).to_csv(idx_path, index=False)
    core = sum((c1 - c0 + 1) for _, _, c0, c1 in segs) / fps
    print(f"[{args.match_id}] {len(segs)} {args.phase.replace('_', ' ')} phases for {names[args.team]}: "
          f"{core / 60:.1f} min of phase time, {total / 60:.1f} min with lead-in/run-out")

    rows_by_frame, own_goal_x = {}, 0.0
    if args.shape:
        pm = load_player_map(args.match_id, args.tmp_dir) or {"players": {}}
        team_ids = {int(t) for v in pm["players"].values() if v["team"] == args.team for t in v["track_ids"]}
        gk_ids = {int(t) for v in pm["players"].values()
                  if v["team"] == args.team and v.get("role") == "GK" for t in v["track_ids"]}
        ignored = {int(i["track_id"]) for i in pm.get("ignored_tracks", [])}
        tracks = pd.read_parquet(mdir / "tracks.parquet", columns=["frame", "track_id", "class", "x1", "x2",
                                                                    "foot_x", "foot_y", "pitch_x", "pitch_y", "kit"])
        tracks = tracks[tracks["class"] == "player"]
        gk = tracks[tracks["track_id"].isin(list(gk_ids))]
        own_goal_x = 0.0 if (gk["pitch_x"].median() if len(gk) else 0.0) < 52.5 else 105.0
        # Every visible outfield player of the team: identified, or an
        # unlinked fragment wearing the team's kit.
        mine = (tracks["track_id"].isin(list(team_ids - gk_ids))
                | (~tracks["track_id"].isin(list(team_ids | ignored)) & (tracks["kit"] == args.team)))
        tracks = tracks[mine]
        keep = np.zeros(len(tracks), bool)
        fr = tracks["frame"].to_numpy()
        for s, e, _, _ in segs:
            keep |= (fr >= s) & (fr <= e)
        rows_by_frame = {int(f): g for f, g in tracks[keep].groupby("frame")}

    color = (230, 190, 90) if args.team == "team_a" else (240, 240, 240)
    writer = VideoWriter(out_path, fps, meta["width"], meta["height"], crf=args.crf)
    cap = cv2.VideoCapture(str(meta["video"]))
    written = 0
    for k, (s, e, c0, c1) in enumerate(segs, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for f in range(s, e + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if args.shape and f in rows_by_frame:
                draw_shape(frame, rows_by_frame[f], own_goal_x, color)
            in_core = c0 <= f <= c1
            clk = clock_label(f, cfg, fps) or ""
            text = (f"{label}  |  phase {k}/{len(segs)}  |  {clk}  |  "
                    + (f"{holder} on the ball" if in_core else "transition"))
            draw_bar(frame, text, (f - lo) / max(hi - lo, 1), color)
            writer.write(frame)
            written += 1
        if k % 10 == 0:
            print(f"  {k}/{len(segs)} phases written", flush=True)
    cap.release()
    writer.close()
    print(f"Wrote {out_path} ({written / fps / 60:.1f} min, {writer.codec}) and {idx_path}")


if __name__ == "__main__":
    main()
