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

--shape-selection replaces the per-frame depth guess with lines through
players YOU picked (tools/select_shape_players.py): one pick per phase, then
those players are followed through the phase (track ids, hand-off to a fresh
track nearby if one fragments, short hold over detector drop-outs). A line
only loses a segment when its player is genuinely off-screen.

Usage:
    python tools/cut_phase_clips.py --match-id <id> --team team_a                 # out of possession
    python tools/cut_phase_clips.py --match-id <id> --team team_a --shape
    python tools/cut_phase_clips.py --match-id <id> --team team_a --shape-selection
    python tools/cut_phase_clips.py --match-id <id> --team team_a --phase in_possession
    python tools/cut_phase_clips.py --match-id <id> --team team_a --shape-selection --export-selection --separate

--export-selection (issue #37) reads config/<id>_<team>_<phase>_export.json,
written by the Stopframe Lines tab: per phase (keyed by start frame) whether
to keep it and where to trim its start and end. Dropped phases are left out
and kept ones are trimmed. A phase missing from the file is kept untrimmed.
--separate writes one MP4 per kept phase (phase_NN_MMmSS.mp4, NN = the
phase number in the picker) and packs them, with the phase index CSV, into
outputs/videos/<id>_<suffix>_clips.zip for PowerPoint. Clips from an
earlier export are replaced, so dropped phases disappear.
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _common import (CONFIG_DIR, OUTPUTS_DIR, Progress, VideoWriter, clock_label, load_calibration, load_meta,
                     load_player_map, match_config, match_dir)

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


def selection_path(match_id, team, phase, config_dir=CONFIG_DIR):
    return Path(config_dir) / f"{match_id}_{team}_{phase}_shape_selection.json"


def load_selection(path):
    """{start_frame: {"keyframe": f, "back": [tid..], "mid": [tid..]}} (missing file -> {})."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f)["phases"].items()}


def save_selection(path, phases_sel):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"phases": {str(k): v for k, v in sorted(phases_sel.items())}}, f, indent=1)
    tmp.replace(path)


def export_path(match_id, team, phase, config_dir=CONFIG_DIR):
    return Path(config_dir) / f"{match_id}_{team}_{phase}_export.json"


def load_export(path):
    """{start_frame: {"keep": bool, "in": frame|None, "out": frame|None}} (missing file -> {})."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f)["phases"].items()}


def save_export(path, phases_exp):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"phases": {str(k): v for k, v in sorted(phases_exp.items())}}, f, indent=1)
    tmp.replace(path)


def export_range(seg, entry):
    """(keep, first_frame, last_frame) of one phase under its export entry:
    the trim points clamped to the phase, ignored if they cross."""
    s, e = int(seg[0]), int(seg[1])
    if not entry:
        return True, s, e
    a = s if entry.get("in") is None else min(max(int(entry["in"]), s), e)
    b = e if entry.get("out") is None else min(max(int(entry["out"]), s), e)
    if b <= a:
        a, b = s, e
    return bool(entry.get("keep", True)), a, b


def clip_name(n, c0, cfg, fps):
    clk = clock_label(c0, cfg, fps)
    return f"phase_{n:02d}" + (f"_{clk.replace(':', 'm')}s" if clk else "") + ".mp4"


def load_team_rows(mdir, tmp_dir, match_id, team, segs):
    """(rows_by_frame, own_goal_x): every visible outfield player of `team`
    inside the phases - identified, or an unlinked fragment in the team's kit."""
    pm = load_player_map(match_id, tmp_dir) or {"players": {}}
    team_ids = {int(t) for v in pm["players"].values() if v["team"] == team for t in v["track_ids"]}
    gk_ids = {int(t) for v in pm["players"].values()
              if v["team"] == team and v.get("role") == "GK" for t in v["track_ids"]}
    ignored = {int(i["track_id"]) for i in pm.get("ignored_tracks", [])}
    tracks = pd.read_parquet(Path(mdir) / "tracks.parquet",
                             columns=["frame", "track_id", "class", "x1", "y1", "x2", "y2", "foot_x", "foot_y",
                                      "pitch_x", "pitch_y", "kit"])
    tracks = tracks[tracks["class"] == "player"]
    gk = tracks[tracks["track_id"].isin(list(gk_ids))]
    own_goal_x = 0.0 if (gk["pitch_x"].median() if len(gk) else 0.0) < 52.5 else 105.0
    mine = (tracks["track_id"].isin(list(team_ids - gk_ids))
            | (~tracks["track_id"].isin(list(team_ids | ignored)) & (tracks["kit"] == team)))
    tracks = tracks[mine]
    keep = np.zeros(len(tracks), bool)
    fr = tracks["frame"].to_numpy()
    for s, e, _, _ in segs:
        keep |= (fr >= s) & (fr <= e)
    return {int(f): g for f, g in tracks[keep].groupby("frame")}, own_goal_x


def entry_slots(entry):
    """[{"line": "back"|"mid", "anchors": [[frame, track_id], ...]}] from a saved
    entry: the slot format, or the first-version back/mid tid lists (anchored
    at the entry's keyframe). One slot = one player chain; anchors are frames
    where the analyst said "this track id is that player". An anchor with a
    third element [frame, track_id, 1] is a CUT: "from this frame on" (used
    to undo an identity swap at a collision), instead of the default
    halfway split between neighbouring anchors."""
    if entry.get("slots") is not None:
        return [{"line": sl["line"], "anchors": norm_anchors(sl["anchors"])} for sl in entry["slots"]]
    kf = entry.get("keyframe")
    return ([{"line": "back", "anchors": [[int(kf), int(t)]]} for t in entry.get("back", [])]
            + [{"line": "mid", "anchors": [[int(kf), int(t)]]} for t in entry.get("mid", [])])


def norm_anchors(anchors):
    """Sorted [[frame, tid]] / [[frame, tid, 1]] (cut) with ints; a cut and a
    plain anchor on the same frame keep only the cut."""
    out = {}
    for a in anchors:
        f, t = int(a[0]), int(a[1])
        cut = len(a) > 2 and bool(a[2])
        if f not in out or cut:
            out[f] = [f, t, 1] if cut else [f, t]
    return [out[f] for f in sorted(out)]


def index_rows(rows_by_frame):
    """{frame: {track_id: row}} - built once, the chain follower reads it a lot."""
    return {f: {int(r.track_id): r for r in g.itertuples()} for f, g in rows_by_frame.items()}


def track_run(rows_idx, tid, frames, fps, forbidden=(), handoff_s=3.0, vmax=8.0):
    """Follow one player over `frames` (consecutive frame numbers in one
    direction, the first being where `tid` is known to be the player).
    Detections only: {frame: (tid, pitch_x, pitch_y, foot_x, foot_y, handoff)}.
    If the track id vanishes, hand off to a FRESH track (absent in the previous
    frame) that appears within reach (vmax * gap + 2.5 m, <= 15 m) of where the
    player was last seen, within `handoff_s`. Fresh-only stops a lost chain
    jumping onto an unpicked neighbour who was standing there all along."""
    out, cur_tid, last, seen, prev = {}, tid, None, None, None
    for f in frames:
        cur = rows_idx.get(f)
        if not cur:
            continue
        r, hand = cur.get(cur_tid), False
        if r is None and seen is not None and prev is not None:
            gap = abs(f - seen) / fps
            if gap <= handoff_s:
                best, bd = None, min(15.0, 2.5 + vmax * gap)
                for t, c in cur.items():
                    if t in prev or t in forbidden:
                        continue
                    d = float(np.hypot(c.pitch_x - last[0], c.pitch_y - last[1]))
                    if d <= bd:
                        best, bd = t, d
                if best is not None:
                    cur_tid, r, hand = best, cur[best], True
        if r is not None:
            last, seen = (float(r.pitch_x), float(r.pitch_y)), f
            out[f] = (cur_tid, float(r.pitch_x), float(r.pitch_y), float(r.foot_x), float(r.foot_y), hand)
        prev = set(cur)
    return out


def follow_chain(rows_idx, anchors, lo, hi, fps, forbidden=()):
    """One player's detections across lo..hi from its anchors: each anchor runs
    forwards and backwards, the runs of neighbouring anchors meeting halfway -
    except at a CUT anchor, which owns everything from its own frame on (the
    previous anchor's run stops the frame before it)."""
    obs, anchors = {}, norm_anchors(anchors)
    for i, a in enumerate(anchors):
        af, at = a[0], a[1]
        cut = len(a) > 2
        if i + 1 == len(anchors):
            end = hi
        elif len(anchors[i + 1]) > 2:
            end = anchors[i + 1][0] - 1
        else:
            end = (af + anchors[i + 1][0]) // 2
        if i == 0:
            begin = lo
        elif cut:
            begin = af
        else:
            begin = (anchors[i - 1][0] + af) // 2 + 1
        obs.update(track_run(rows_idx, at, range(af, min(end, hi) + 1), fps, forbidden))
        obs.update(track_run(rows_idx, at, range(af, max(begin, lo) - 1, -1), fps, forbidden))
    return obs


def fill_gaps(obs, fps, H_at, lo, hi, fill_s=4.0, end_hold=12):
    """Detections -> {frame: (tid, pitch_x, pitch_y, foot_x, foot_y, status)},
    status "seen" | "link" (first frame after a track hand-off) | "filled".
    A gap between two sightings of <= fill_s is filled by interpolating the
    PITCH position and projecting it back through that frame's homography
    (pixel-space interpolation is wrong while the camera pans; without a
    homography it falls back to pixels). Longer gaps stay open. The ends are
    held `end_hold` frames at the last pixel position."""
    out = {f: (o[0], o[1], o[2], o[3], o[4], "link" if o[5] else "seen") for f, o in obs.items()}
    fr = sorted(obs)

    def project(f, px, py):
        H = H_at(f) if H_at else None
        if H is None:
            return None
        q = np.linalg.solve(H, np.array([px, py, 1.0]))
        return (q[0] / q[2], q[1] / q[2]) if abs(q[2]) > 1e-9 else None

    for a, b in zip(fr, fr[1:]):
        n = b - a
        if n <= 1 or n > fill_s * fps:
            continue
        oa, ob = obs[a], obs[b]
        for f in range(a + 1, b):
            u = (f - a) / n
            px, py = oa[1] + (ob[1] - oa[1]) * u, oa[2] + (ob[2] - oa[2]) * u
            xy = project(f, px, py) or (oa[3] + (ob[3] - oa[3]) * u, oa[4] + (ob[4] - oa[4]) * u)
            if np.isfinite(xy).all() and abs(xy[0]) < 4000 and abs(xy[1]) < 4000:
                out[f] = (oa[0], px, py, float(xy[0]), float(xy[1]), "filled")
    if fr:
        for edge, step in ((fr[0], -1), (fr[-1], 1)):
            o = obs[edge]
            for k in range(1, end_hold + 1):
                f = edge + step * k
                if lo <= f <= hi and f not in out:
                    out[f] = (o[0], o[1], o[2], o[3], o[4], "filled")
    return out


def _run_start(raw, f, tid):
    """First frame of the unbroken run of `tid` in a chain that contains frame f."""
    k = f
    while (k - 1) in raw and raw[k - 1][0] == tid:
        k -= 1
    return k


def _claim(raw, f, tid, fps):
    """How strongly a chain owns `tid` at frame f: (start of its unbroken run
    on tid, pitch distance from the chain's previous sighting to where tid
    starts). Earlier run wins; on a tie (both handed off onto the same fresh
    track) the chain that was closer wins."""
    k = _run_start(raw, f, tid)
    here = raw[k]
    for b in range(k - 1, k - int(4 * fps) - 1, -1):
        prev = raw.get(b)
        if prev is not None:
            return k, float(np.hypot(here[1] - prev[1], here[2] - prev[2]))
    return k, 0.0


def build_chains(rows_idx, slots, lo, hi, fps, H_at, max_rounds=6):
    """[{"line", "pts": {frame: (tid, px, py, fx, fy, status)}, "raw": detections}] for one phase.

    One track belongs to at most one chain per frame (issue #34: chains were
    "merging" - after a drop-out a chain handed off onto a fresh track that
    another picked chain was already following, so two markers sat on one
    player on ~26% of picked frames). Conflicts are resolved jointly: the
    chain that has the track among its own anchors keeps it, else the one
    that has held it longest; the other chain is re-followed with that track
    forbidden, so it shows a gap (filled or lost) instead of doubling up."""
    every = {a[1] for sl in slots for a in sl["anchors"]}
    mine = [{a[1] for a in sl["anchors"]} for sl in slots]
    extra = [set() for _ in slots]

    def follow(i):
        return follow_chain(rows_idx, slots[i]["anchors"], lo, hi, fps,
                            forbidden=(every - mine[i]) | (extra[i] - mine[i]))

    raws = [follow(i) for i in range(len(slots))]
    for _ in range(max_rounds):
        users = {}
        for i, raw in enumerate(raws):
            for f, o in raw.items():
                users.setdefault((f, o[0]), []).append(i)
        losers = {}
        for (f, tid), cis in users.items():
            if len(cis) < 2:
                continue
            anchored = [i for i in cis if tid in mine[i]]
            if len(anchored) == 1:
                win = anchored[0]
            else:
                win = min(cis, key=lambda i: (*_claim(raws[i], f, tid, fps), i))
            for i in cis:
                if i != win and tid not in mine[i]:
                    losers.setdefault(i, set()).add(tid)
        losers = {i: t for i, t in losers.items() if not t <= extra[i]}
        if not losers:
            break
        for i, tids in losers.items():
            extra[i] |= tids
            raws[i] = follow(i)
    return [{"line": sl["line"], "raw": raw, "pts": fill_gaps(raw, fps, H_at, lo, hi)}
            for sl, raw in zip(slots, raws)]


def chains_by_frame(chains):
    """{frame: [(line, foot_x, foot_y, pitch_y, tid, status)]}"""
    out = {}
    for ch in chains:
        for f, (tid, px, py, fx, fy, st) in ch["pts"].items():
            out.setdefault(f, []).append((ch["line"], fx, fy, py, tid, st))
    return out


def homography_lookup(calib):
    """frame -> 3x3 image->pitch homography (or None), from a loaded calibration."""
    if calib is None:
        return None
    if "frame_H" in calib:
        table = dict(zip(calib["frame_index"].tolist(), calib["frame_H"]))
        return table.get
    if "homography" in calib:
        return lambda f: calib["homography"]
    return None


def draw_selected_shape(canvas, items, color):
    """items from chains_by_frame for one frame: ellipse on each chosen player
    (thin where the position is interpolated); back line in team colour,
    midfield line white, joined across the pitch."""
    by_line = {}
    for line, fx, fy, py, _, status in items:
        c = color if line == "back" else (255, 255, 255)
        cv2.ellipse(canvas, (int(fx), int(fy)), (14, 5), 0, -35, 235, c, 1 if status == "filled" else 2, cv2.LINE_AA)
        by_line.setdefault(line, []).append((py, fx, fy))
    for line, pts in by_line.items():
        if len(pts) >= 2:
            arr = np.array([[fx, fy] for _, fx, fy in sorted(pts)], np.int32)
            cv2.polylines(canvas, [arr], False, color if line == "back" else (255, 255, 255), 3, cv2.LINE_AA)


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
    parser.add_argument("--shape-selection", nargs="?", const="auto", default=None, metavar="JSON",
                        help="Draw lines through hand-picked players (select_shape_players.py) instead of the "
                             "per-frame depth guess. Default file: config/<id>_<team>_<phase>_shape_selection.json")
    parser.add_argument("--unselected", default="none", choices=["none", "auto"],
                        help="Phases without a pick: draw nothing (default) or the old depth-based guess")
    parser.add_argument("--export-selection", nargs="?", const="auto", default=None, metavar="JSON",
                        help="Keep / trim phases as chosen in the Lines tab. Default file: "
                             "config/<id>_<team>_<phase>_export.json")
    parser.add_argument("--separate", action="store_true",
                        help="One MP4 per phase, zipped (for PowerPoint), instead of one compilation")
    parser.add_argument("--crf", type=int, default=26)
    parser.add_argument("--output", default=None, help="Compilation .mp4, or the .zip with --separate")
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
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

    # (phase number, start, end, core start, core end, first rendered frame, last rendered frame)
    plan = [(k, s, e, c0, c1, s, e) for k, (s, e, c0, c1) in enumerate(segs, 1)]
    if args.export_selection:
        exp_path = (export_path(args.match_id, args.team, args.phase, args.config_dir)
                    if args.export_selection == "auto" else Path(args.export_selection))
        export = load_export(exp_path)
        plan = []
        for k, sg in enumerate(segs, 1):
            keep, a, b = export_range(sg, export.get(int(sg[0])))
            if keep:
                plan.append((k, sg[0], sg[1], sg[2], sg[3], a, b))
        print(f"Export selection: {len(plan)}/{len(segs)} phases kept, "
              f"{sum(1 for p in plan if (p[5], p[6]) != (p[1], p[2]))} trimmed ({exp_path})")
        if not plan:
            raise SystemExit("Every phase is dropped in the export selection - keep at least one in the Lines tab")

    names = {"team_a": cfg.get("team_a_name", "team_a"), "team_b": cfg.get("team_b_name", "team_b")}
    other = "team_b" if args.team == "team_a" else "team_a"
    label = (f"{names[args.team].upper()} OUT OF POSSESSION" if args.phase == "out_of_possession"
             else f"{names[args.team].upper()} IN POSSESSION")
    holder = names[other] if args.phase == "out_of_possession" else names[args.team]
    suffix = (f"{args.team}_{args.phase}"
              + ("_shape_selected" if args.shape_selection else "_shape" if args.shape else ""))
    if args.separate:
        zip_path = Path(args.output) if args.output else OUTPUTS_DIR / "videos" / f"{args.match_id}_{suffix}_clips.zip"
        clips_dir = zip_path.with_suffix("")
    out_path = args.output or (OUTPUTS_DIR / "videos" / f"{args.match_id}_{suffix}.mp4")

    index = []
    total = 0.0
    for k, s, e, c0, c1, a, b in plan:
        dur = (b - a + 1) / fps
        row = {"phase": k, "start_frame": a, "end_frame": b,
               "video_start": f"{int(a / fps // 60):02d}:{a / fps % 60:04.1f}",
               "match_clock_start": clock_label(c0, cfg, fps), "match_clock_end": clock_label(c1, cfg, fps),
               "phase_s": round((c1 - c0 + 1) / fps, 1), "clip_s": round(dur, 1)}
        if args.separate:
            row["file"] = clip_name(k, c0, cfg, fps)
        else:
            row["compilation_start"] = f"{int(total // 60):02d}:{total % 60:04.1f}"
        index.append(row)
        total += dur
    idx_name = f"{args.match_id}_{args.team}_{args.phase}_phases" + ("_export" if args.export_selection else "")
    idx_path = OUTPUTS_DIR / "stats" / f"{idx_name}.csv"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index).to_csv(idx_path, index=False)
    core = sum((c1 - c0 + 1) for _, _, _, c0, c1, _, _ in plan) / fps
    print(f"[{args.match_id}] {len(plan)} {args.phase.replace('_', ' ')} phases for {names[args.team]}: "
          f"{core / 60:.1f} min of phase time, {total / 60:.1f} min rendered")

    rows_by_frame, own_goal_x = {}, 0.0
    selection, followed = {}, {}
    full = [(s, e, c0, c1) for _, s, e, c0, c1, _, _ in plan]
    if args.shape or args.shape_selection:
        rows_by_frame, own_goal_x = load_team_rows(mdir, args.tmp_dir, args.match_id, args.team, full)
    if args.shape_selection:
        sel_path = (selection_path(args.match_id, args.team, args.phase, args.config_dir)
                    if args.shape_selection == "auto" else Path(args.shape_selection))
        selection = load_selection(sel_path)
        if not selection and not args.export_selection:
            raise SystemExit(f"No selection in {sel_path} - run tools/select_shape_players.py first")
        rows_idx = index_rows(rows_by_frame)
        H_at = homography_lookup(load_calibration(args.match_id, args.config_dir))
        if H_at is None:
            print("WARNING: no calibration - gaps are filled in pixel space (wrong while the camera pans)")
        n_sel = 0
        for s, e, _, _ in full:  # chains run over the whole phase: anchors may sit outside a trim
            slots = entry_slots(selection[s]) if s in selection else []
            if slots:
                n_sel += 1
                followed.update(chains_by_frame(build_chains(rows_idx, slots, s, e, fps, H_at)))
        print(f"Selection: {n_sel}/{len(full)} phases have picked players ({sel_path})")

    color = (230, 190, 90) if args.team == "team_a" else (240, 240, 240)
    cap = cv2.VideoCapture(str(meta["video"]))
    prog = Progress(sum(b - a + 1 for *_, a, b in plan), "rendering clips")
    written = 0

    def render(writer, k, s, c0, c1, a, b):
        nonlocal written
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for f in range(a, b + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if args.shape_selection:
                if f in followed:
                    draw_selected_shape(frame, followed[f], color)
                elif args.unselected == "auto" and not selection.get(s) and f in rows_by_frame:
                    draw_shape(frame, rows_by_frame[f], own_goal_x, color)
            elif args.shape and f in rows_by_frame:
                draw_shape(frame, rows_by_frame[f], own_goal_x, color)
            in_core = c0 <= f <= c1
            clk = clock_label(f, cfg, fps) or ""
            text = (f"{label}  |  phase {k}/{len(segs)}  |  {clk}  |  "
                    + (f"{holder} on the ball" if in_core else "transition"))
            draw_bar(frame, text, (f - lo) / max(hi - lo, 1), color)
            writer.write(frame)
            written += 1
            prog.update(written)

    if args.separate:
        if clips_dir.exists():
            shutil.rmtree(clips_dir)
        clips_dir.mkdir(parents=True)
        for row, (k, s, e, c0, c1, a, b) in zip(index, plan):
            writer = VideoWriter(clips_dir / row["file"], fps, meta["width"], meta["height"], crf=args.crf)
            render(writer, k, s, c0, c1, a, b)
            writer.close()
        shutil.copy(idx_path, clips_dir / "phases.csv")
        tmp_zip = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as z:  # MP4s don't compress further
            for p in sorted(clips_dir.iterdir()):
                z.write(p, arcname=p.name)
        tmp_zip.replace(zip_path)
        shutil.rmtree(clips_dir)  # the zip is the deliverable; don't keep two copies on the 15 GB disk
        prog.done()
        cap.release()
        print(f"Wrote {zip_path}: {len(plan)} clips, {written / fps / 60:.1f} min, "
              f"{zip_path.stat().st_size / 1e6:.0f} MB, and {idx_path}")
        return

    writer = VideoWriter(out_path, fps, meta["width"], meta["height"], crf=args.crf)
    for i, (k, s, e, c0, c1, a, b) in enumerate(plan, 1):
        render(writer, k, s, c0, c1, a, b)
        if i % 10 == 0:
            print(f"  {i}/{len(plan)} phases written", flush=True)
    prog.done()
    cap.release()
    writer.close()
    print(f"Wrote {out_path} ({written / fps / 60:.1f} min, {writer.codec}) and {idx_path}")


if __name__ == "__main__":
    main()
