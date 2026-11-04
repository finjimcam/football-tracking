"""Pick the back-line and midfield-line players for each phase, by hand - and
SEE where the tracker's chain on each player breaks.

Companion to `cut_phase_clips.py --shape-selection` (workflows/track_identities.md,
stage 7). The automatic --shape lines are rebuilt every frame from "the 4
deepest visible players", so they flicker. Here YOU choose the players once
per phase; the render follows exactly those players. Each picked player is a
CHAIN: the tracker's track id, handed on to a fresh track nearby when it
fragments, with the missing frames filled in from the sightings either side
(interpolated in pitch space, projected back through each frame's homography).

For each phase the tool shows a frame where most of the team is visible, with
the old automatic guess pre-selected. Scrub through the phase and watch the
STRIP at the bottom - one row per picked player:

    green  = detected          amber = interpolated (filled from other frames)
    red    = lost (gap too long to fill)      white tick = track hand-off
                                              (where the tracker's id changed)
    yellow = the frame you're looking at; dashed ring = interpolated player

Controls:
    left-click player     cycle  none -> BACK -> MIDFIELD -> none (the whole chain)
    right-click player    RE-LINK: attach him to the lost chain nearest to him -
                          step to a frame after a red/amber gap, right-click the
                          right player, the chain jumps onto him
    click the strip       jump to that frame
    a / d                 -/+ 0.5 s      A / D  -/+ 2 s      , / .   -/+ 1 frame
    Enter / Space         accept this phase, next        s  skip (no lines)
    r                     reset to the automatic guess   c  clear all picks
    p                     previous phase                 q / Esc  save + quit

Picks are saved after every phase to
config/<match_id>_<team>_<phase>_shape_selection.json (hand-made input, so it
lives in config/). Resume by re-running: it restarts at the first phase with
no entry. Phases are keyed by start frame - keep the same
--merge-s/--min-s/--pre-s/--post-s as cut_phase_clips.py.

--auto-fill writes the automatic guess for every phase without a window.

Usage:
    python tools/select_shape_players.py --match-id <id> --team team_a
    python tools/select_shape_players.py --match-id <id> --team team_a --from-phase 12
    python tools/cut_phase_clips.py --match-id <id> --team team_a --shape-selection
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _common import CONFIG_DIR, clock_label, load_calibration, load_meta, match_config, match_dir
from cut_phase_clips import (build_chains, chains_by_frame, depth_lines, entry_slots, homography_lookup,
                             index_rows, load_selection, load_team_rows, phases, save_selection, selection_path)

WIN = "select back line + midfield line"
STRIP_ROW, STRIP_X = 9, 10
GREEN, AMBER, RED = (90, 200, 90), (40, 170, 240), (60, 60, 170)


def auto_slots(rows, frame, own_goal_x, sizes=(4, 3)):
    """Slots from the identity-free depth guess at `frame` (the pre-fill)."""
    slots = []
    lines = depth_lines(rows, own_goal_x, sizes, min_visible=7)
    tids = rows["track_id"].to_numpy()
    for k, idx in enumerate(lines):
        if idx is not None:
            slots += [{"line": "back" if k == 0 else "mid", "anchors": [[int(frame), int(tids[i])]]} for i in idx]
    return slots


def pick_keyframe(rows_by_frame, c0, c1, own_goal_x, step=5):
    """First frame in the phase core where both automatic lines are coherent
    (so the pre-fill is sensible); else the frame with the most players."""
    best_f, best_n = c0, -1
    for f in range(c0, c1 + 1, step):
        rows = rows_by_frame.get(f)
        if rows is None:
            continue
        lines = depth_lines(rows, own_goal_x)
        if len(lines) == 2 and all(l is not None for l in lines):
            return f
        if len(rows) > best_n:
            best_f, best_n = f, len(rows)
    return best_f


def hit_test(rows, x, y):
    """Track id of the player whose box contains (x, y); else the nearest
    feet within 40 px; else None."""
    best, bd = None, 1e9
    for r in rows.itertuples():
        if r.x1 <= x <= r.x2 and r.y1 <= y <= r.y2:
            d = abs(x - (r.x1 + r.x2) / 2) + abs(y - r.foot_y) * 0.25
            if d < bd:
                best, bd = int(r.track_id), d
    if best is not None:
        return best
    for r in rows.itertuples():
        d = float(np.hypot(x - r.foot_x, y - r.foot_y))
        if d < min(bd, 40):
            best, bd = int(r.track_id), d
    return best


def relink_target(chains, frame, px, py):
    """Index of the chain to attach a player at pitch (px, py) to: among chains
    with no real detection at `frame`, the one whose nearest sighting in time
    is closest to him in the pitch. None when every chain is currently seen."""
    best, bd = None, 1e9
    for i, ch in enumerate(chains):
        cur = ch["pts"].get(frame)
        if cur is not None and cur[5] != "filled":
            continue
        if not ch["raw"]:
            continue
        near = min(ch["raw"], key=lambda f: abs(f - frame))
        d = float(np.hypot(ch["raw"][near][1] - px, ch["raw"][near][2] - py))
        if d < bd:
            best, bd = i, d
    return best


def draw_strip(img, chains, s, e, cur, color):
    """One row per chain across the phase; returns the strip's top y."""
    h, w = img.shape[:2]
    x0, x1 = STRIP_X, w - STRIP_X
    top = h - 24 - 6 - len(chains) * STRIP_ROW - 4
    cv2.rectangle(img, (0, top - 3), (w, h - 24), (20, 20, 20), -1)
    span = max(e - s, 1)
    cols = np.arange(x0, x1)
    frames = s + ((cols - x0) / max(x1 - x0 - 1, 1) * span).astype(int)
    for i, ch in enumerate(chains):
        y = top + i * STRIP_ROW
        for x, f in zip(cols, frames):
            p = ch["pts"].get(int(f))
            c = RED if p is None else (AMBER if p[5] == "filled" else GREEN)
            img[y:y + STRIP_ROW - 2, x] = c
        for f, p in ch["pts"].items():
            if p[5] == "link":
                x = x0 + int((f - s) / span * (x1 - x0 - 1))
                img[y:y + STRIP_ROW - 2, max(x - 1, 0):x + 1] = (255, 255, 255)
        tag = "B" if ch["line"] == "back" else "M"
        cv2.putText(img, tag, (2, y + STRIP_ROW - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    color if ch["line"] == "back" else (255, 255, 255), 1, cv2.LINE_AA)
    xc = x0 + int((cur - s) / span * (x1 - x0 - 1))
    cv2.line(img, (xc, top - 3), (xc, h - 24), (0, 235, 255), 1)
    return top


def render(frame, rows, chains, cur, s, e, header, color, msg=""):
    img = frame.copy()
    taken = {p[0] for ch in chains for f, p in [(cur, ch["pts"].get(cur))] if p and p[5] != "filled"}
    for r in rows.itertuples():
        if int(r.track_id) not in taken:
            cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (150, 150, 150), 1)
    by_line = {}
    for ch in chains:
        p = ch["pts"].get(cur)
        if p is None:
            continue
        tid, px, py, fx, fy, st = p
        c = color if ch["line"] == "back" else (255, 255, 255)
        mark = "B" if ch["line"] == "back" else "M"
        if st == "filled":
            for ang in range(0, 360, 40):  # dashed ring = interpolated
                cv2.ellipse(img, (int(fx), int(fy)), (16, 6), 0, ang, ang + 22, c, 1, cv2.LINE_AA)
            mark += "~"
        else:
            cv2.ellipse(img, (int(fx), int(fy)), (16, 6), 0, -35, 235, c, 2, cv2.LINE_AA)
            if st == "link":
                mark += "*"
        cv2.putText(img, mark, (int(fx) - 8, int(fy) - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
        by_line.setdefault(ch["line"], []).append((py, fx, fy))
    for line, pts in by_line.items():
        if len(pts) >= 2:
            arr = np.array([[x, y] for _, x, y in sorted(pts)], np.int32)
            cv2.polylines(img, [arr], False, color if line == "back" else (255, 255, 255), 2, cv2.LINE_AA)
    top = draw_strip(img, chains, s, e, cur, color)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(img, header + (f"   {msg}" if msg else ""), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    n_b = sum(1 for c in chains if c["line"] == "back")
    n_m = len(chains) - n_b
    foot = (f"BACK {n_b} MID {n_m} | click=cycle right-click=re-link strip=seek a/d A/D ,/. Enter=ok "
            "r=auto c=clear s=skip p=prev q=quit")
    cv2.rectangle(img, (0, img.shape[0] - 24), (img.shape[1], img.shape[0]), (20, 20, 20), -1)
    cv2.putText(img, foot, (8, img.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img, top


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--team", default="team_a", choices=["team_a", "team_b"])
    parser.add_argument("--phase", default="out_of_possession", choices=["out_of_possession", "in_possession"])
    parser.add_argument("--merge-s", type=float, default=2.0)
    parser.add_argument("--min-s", type=float, default=3.0)
    parser.add_argument("--pre-s", type=float, default=1.5)
    parser.add_argument("--post-s", type=float, default=1.0)
    parser.add_argument("--from-phase", type=int, default=None, help="Start at this phase number (1-based)")
    parser.add_argument("--redo", action="store_true", help="Revisit phases that already have a pick")
    parser.add_argument("--auto-fill", action="store_true",
                        help="Write the automatic guess for phases without a pick; no window")
    parser.add_argument("--output", default=None, help="Default: config/<id>_<team>_<phase>_shape_selection.json")
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
    out_path = Path(args.output) if args.output else selection_path(args.match_id, args.team, args.phase,
                                                                    args.config_dir)
    sel = load_selection(out_path)
    rows_by_frame, own_goal_x = load_team_rows(mdir, args.tmp_dir, args.match_id, args.team, segs)
    rows_idx = index_rows(rows_by_frame)
    H_at = homography_lookup(load_calibration(args.match_id, args.config_dir))
    color = (230, 190, 90) if args.team == "team_a" else (240, 240, 240)
    print(f"{len(segs)} phases; {len(sel)} already have an entry in {out_path}")

    def entry_of(kf, slots):
        return {"keyframe": int(kf), "slots": slots}

    if args.auto_fill:
        for s, e, c0, c1 in segs:
            if s in sel:
                continue
            kf = pick_keyframe(rows_by_frame, c0, c1, own_goal_x)
            rows = rows_by_frame.get(kf)
            sel[s] = entry_of(kf, auto_slots(rows, kf, own_goal_x) if rows is not None else [])
        save_selection(out_path, sel)
        print(f"Auto-filled -> {out_path}")
        return

    cap = cv2.VideoCapture(str(meta["video"]))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open {meta['video']}")

    def read(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        return fr if ok else np.zeros((meta["height"], meta["width"], 3), np.uint8)

    empty = pd.DataFrame(columns=["track_id", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "pitch_x", "pitch_y"])
    st = {"slots": [], "chains": [], "cur": 0, "seek": None, "msg": "", "s": 0, "e": 0, "strip_top": 10 ** 6}

    def rebuild():
        st["chains"] = build_chains(rows_idx, st["slots"], st["s"], st["e"], fps, H_at)

    def on_mouse(event, x, y, *_):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        cur, s, e = st["cur"], st["s"], st["e"]
        if y >= st["strip_top"] - 3 and y < meta["height"] - 24:
            w = meta["width"]
            st["seek"] = int(np.clip(s + (x - STRIP_X) / max(w - 2 * STRIP_X - 1, 1) * (e - s), s, e))
            return
        rows = rows_by_frame.get(cur, empty)
        tid = hit_test(rows, x, y) if len(rows) else None
        if tid is None:
            return
        owner = next((i for i, ch in enumerate(st["chains"])
                      if ch["pts"].get(cur) and ch["pts"][cur][0] == tid and ch["pts"][cur][5] != "filled"), None)
        if event == cv2.EVENT_LBUTTONDOWN:
            if owner is None:
                st["slots"].append({"line": "back", "anchors": [[cur, tid]]})
            elif st["slots"][owner]["line"] == "back":
                st["slots"][owner]["line"] = "mid"
            else:
                st["slots"].pop(owner)
            st["msg"] = ""
        else:
            if owner is not None:
                st["msg"] = "already on a chain - right-click a player whose chain is lost/amber"
                return
            r = rows[rows["track_id"].astype(int) == tid].iloc[0]
            tgt = relink_target(st["chains"], cur, float(r.pitch_x), float(r.pitch_y))
            if tgt is None:
                st["msg"] = "every chain is detected on this frame - nothing to re-link"
                return
            st["slots"][tgt]["anchors"].append([cur, tid])
            st["msg"] = f"re-linked chain {tgt + 1} ({st['slots'][tgt]['line']})"
        rebuild()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    if args.from_phase:
        k = max(0, min(args.from_phase - 1, len(segs) - 1))
    elif args.redo:
        k = 0
    else:
        k = next((i for i, sg in enumerate(segs) if sg[0] not in sel), len(segs))
    quit_all = False
    while 0 <= k < len(segs) and not quit_all:
        s, e, c0, c1 = segs[k]
        prior = sel.get(s)
        kf = prior["keyframe"] if prior and prior.get("keyframe") is not None else pick_keyframe(
            rows_by_frame, c0, c1, own_goal_x)
        st.update(s=s, e=e, cur=int(kf), seek=None, msg="")
        if prior is not None:
            st["slots"] = entry_slots(prior)
        else:
            rows = rows_by_frame.get(kf)
            st["slots"] = auto_slots(rows, kf, own_goal_x) if rows is not None else []
        rebuild()
        shown, frame = None, None
        go = None
        while go is None:
            if st["seek"] is not None:
                st["cur"], st["seek"] = st["seek"], None
            cur = st["cur"]
            if cur != shown:
                frame, shown = read(cur), cur
            rows = rows_by_frame.get(cur, empty)
            hdr = (f"phase {k + 1}/{len(segs)}  {clock_label(cur, cfg, fps) or ''}  frame {cur}  "
                   f"({(cur - s) / fps:+.1f}s from phase start)")
            img, st["strip_top"] = render(frame, rows, st["chains"], cur, s, e, hdr, color, st["msg"])
            cv2.imshow(WIN, img)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32):
                sel[s] = entry_of(kf, st["slots"])
                save_selection(out_path, sel)
                go = k + 1
            elif key == ord("s"):
                sel[s] = entry_of(kf, [])
                save_selection(out_path, sel)
                go = k + 1
            elif key == ord("p"):
                go = max(0, k - 1)
            elif key in (ord("q"), 27):
                save_selection(out_path, sel)
                quit_all, go = True, k
            elif key == ord("r"):
                rws = rows_by_frame.get(kf)
                st["slots"] = auto_slots(rws, kf, own_goal_x) if rws is not None else []
                st["msg"] = "reset to automatic guess"
                rebuild()
            elif key == ord("c"):
                st["slots"] = []
                st["msg"] = ""
                rebuild()
            elif key != 255:
                step = {ord("a"): -int(0.5 * fps), ord("d"): int(0.5 * fps), ord("A"): -int(2 * fps),
                        ord("D"): int(2 * fps), ord(","): -1, ord("."): 1}.get(key)
                if step is not None:
                    st["cur"] = int(np.clip(cur + step, s, e))
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                save_selection(out_path, sel)
                quit_all, go = True, k
        k = go
    cap.release()
    cv2.destroyAllWindows()
    save_selection(out_path, sel)
    done = sum(1 for sg in segs if sg[0] in sel)
    print(f"Saved {done}/{len(segs)} phases -> {out_path}")
    if done < len(segs):
        print("Re-run to continue where you left off.")
    print(f"Render: python tools/cut_phase_clips.py --match-id {args.match_id} --team {args.team} "
          f"--phase {args.phase} --shape-selection")


if __name__ == "__main__":
    main()
