"""Curate + label the people on anchor frames -> config/<match_id>_roster.json.

Run AFTER tracking (and ideally after stitch_tracks.py + assign_teams.py +
calibration). The tool draws the tracked boxes on each anchor frame and you
attach identities to the real players — and curate everyone else out.

Auto-filtering (v2): boxes are pre-classified so you only deal with actual
players:
  - feet projecting OUTSIDE the pitch (via the calibration, --pitch-margin-m)
    are drawn dim + dashed with an "out" tag — spectators, coaches, ball-boys;
  - jerseys far from BOTH team kit clusters (team_map.json says `unknown`)
    are drawn dim + dashed with a "kit?" tag — likely referees/odd keepers;
  - clicking a dimmed box RESCUES it (your override always wins) and selects
    it for labeling. Boxes near a kit cluster get their team pre-filled once
    enough of your labels establish which cluster is `a` and which is `b`.

Controls (all in the OpenCV window):
  - CLICK a box to select it (yellow). Dimmed boxes are rescued by the click.
  - Type the identity: `a`/`b` for the team (may be pre-filled from the kit
    color), then the shirt number digits. ENTER confirms. ESC cancels.
    BACKSPACE deletes a digit.
  - `x` with a box selected: IGNORE that person's whole track (referee or
    spectator on the pitch) — excluded from ALL downstream stats. `x` on an
    already-ignored box un-ignores it. `x` on a manual box deletes it.
  - `d` with a labeled box selected removes its label.
  - DRAG on open ground: draw a box over a player the detector missed, then
    label it as normal. tools/retrack_manual.py later tracks them through
    the video. Dragging over an already-tracked person just selects them.
  - `o` toggles BALL MODE: click marks the ball on this frame (corrects or
    fills in sparse automatic ball detection; manual beats auto downstream).
    `d` undoes the last mark. `o`/ESC exits ball mode.
  - ENTER with nothing selected finishes the current anchor frame.
  - Q quits early but SAVES everything done so far (roster flagged
    "partial" — rerun this tool to finish; anchors merge, nothing is lost).

Clicks are stored as (frame, image x/y) pairs and re-resolved against
tracks.parquet by assign_identities.py, so the roster survives a .tmp wipe
and re-runs of stitch_tracks (track_ids may change; your clicks don't).

Pick anchor frames like calibration keyframes: with a panning camera no
single frame shows all 22 players, so label one early frame and add 1-2 more
where the camera points elsewhere (--frames N1,N2,...). Re-opening an
existing anchor frame MERGES with what you did before (v2 — nothing is
dropped); players labeled on other anchors show up pre-labeled if their
track is still alive.

Requires a desktop session (opens a GUI window).

Usage:
    python tools/label_roster.py --match-id <match_id> --frame 0
    python tools/label_roster.py --match-id <match_id> --frames 0,900,1800
    python tools/label_roster.py --match-id <match_id> --frame 0 \
        --team-a-name "Reds" --team-b-name "Blues"
"""
import argparse

import cv2

from _common import (apply_calibration, load_calibration, load_meta, load_roster,
                     load_tracks, load_team_map, pitch_dims, resolve_click_to_track,
                     roster_path, write_json)

TEAM_DRAW = {"team_a": (60, 60, 230), "team_b": (230, 130, 50)}  # BGR, drawing only
UNLABELED = (170, 170, 170)
SELECTED = (0, 255, 255)
DIMMED = (100, 100, 100)
IGNORED = (60, 60, 170)
MANUAL = (255, 0, 255)
BALL = (0, 215, 255)

MIN_DRAG_PX = 8  # smaller mouse movement counts as a click, not a drag


def nearest_frame_with_players(players, frame_idx):
    frames = players["frame"].unique()
    return int(min(frames, key=lambda f: abs(int(f) - frame_idx)))


def draw_dashed_rect(img, p1, p2, color, thickness=1, dash=7):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def suggest_team(tid, team_map, labels_iter):
    """Pre-fill the roster team for a track from its kit cluster, once >=2 of
    the user's existing labels establish the cluster<->a/b alignment."""
    kit = team_map.get(tid)
    if kit not in ("team_a", "team_b"):
        return None
    agree = disagree = 0
    for t, lab in labels_iter:
        k = team_map.get(t)
        if k in ("team_a", "team_b"):
            if k == lab["team"]:
                agree += 1
            else:
                disagree += 1
    if agree + disagree < 2 or agree == disagree:
        return None
    if agree > disagree:
        return kit
    return {"team_a": "team_b", "team_b": "team_a"}[kit]


def label_one_frame(frame_img, frame_idx, rows, ball_rows, known_labels,
                    known_ignored, auto_out, kit_flag, team_map, initial):
    """Curate/label loop for one anchor frame.

    rows: player-class slice of tracks.parquet at this frame.
    known_labels / known_ignored: labels + ignore decisions from OTHER anchors
    (shown, not re-saved here). auto_out / kit_flag: track_ids dimmed by the
    pitch-boundary / kit-color filter. initial: this frame's existing anchor
    content (merged, so re-opening an anchor never loses work).

    Returns (anchor_content dict, aborted bool)."""
    mouse = {"down": None, "cur": None, "click": None, "drag": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = (x, y)
            mouse["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["down"]:
            mouse["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and mouse["down"]:
            x0, y0 = mouse["down"]
            mouse["down"] = None
            if abs(x - x0) < MIN_DRAG_PX and abs(y - y0) < MIN_DRAG_PX:
                mouse["click"] = (x, y)
            else:
                mouse["drag"] = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))

    win = f"label_roster — frame {frame_idx}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    frame_tids = set(rows["track_id"].astype(int))
    new_labels = {lab["track_id_at_label"]: dict(lab)
                  for lab in initial.get("labels", [])
                  if lab.get("track_id_at_label") in frame_tids}
    ignores = {ig["track_id_at_label"]: dict(ig)
               for ig in initial.get("ignores", [])}
    rescues = {rc["track_id_at_label"]: dict(rc)
               for rc in initial.get("rescues", [])}
    manual_boxes = [dict(mb) for mb in initial.get("manual_boxes", [])]
    ball_marks = [list(bm) for bm in initial.get("ball_marks", [])]

    selected = None  # ("t", track_id) or ("m", manual index)
    entry_team, entry_num = None, ""
    prefilled = False
    ball_mode = False

    def label_for(tid):
        return new_labels.get(tid) or known_labels.get(tid)

    def is_ignored(tid):
        return tid in ignores or (tid in known_ignored and tid not in rescues)

    def foot_of(tid):
        r = rows[rows["track_id"] == tid].iloc[0]
        return [round(float(r["foot_x"]), 1), round(float(r["foot_y"]), 1)]

    def all_labels_iter():
        seen = []
        for tid, lab in list(known_labels.items()) + list(new_labels.items()):
            if tid is not None:
                seen.append((tid, lab))
        return seen

    def select_track(tid):
        nonlocal selected, entry_team, entry_num, prefilled
        selected = ("t", tid)
        entry_num, prefilled = "", False
        entry_team = None
        lab = label_for(tid)
        if lab:
            print(f"  Track #{tid} currently labeled {lab['key']} "
                  "(type a new identity to replace, d = remove, x = ignore)")
        elif is_ignored(tid):
            print(f"  Track #{tid} is IGNORED — x to un-ignore, or label it to rescue")
        else:
            sugg = suggest_team(tid, team_map, all_labels_iter())
            if sugg:
                entry_team, prefilled = sugg, True

    def redraw():
        canvas = frame_img.copy()
        for _, r in rows.iterrows():
            tid = int(r["track_id"])
            x1, y1, x2, y2 = (int(r[c]) for c in ("x1", "y1", "x2", "y2"))
            lab = label_for(tid)
            sel = selected == ("t", tid)
            if is_ignored(tid):
                color = SELECTED if sel else IGNORED
                draw_dashed_rect(canvas, (x1, y1), (x2, y2), color, 2 if sel else 1)
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
                cv2.line(canvas, (x1, y2), (x2, y1), color, 1)
                cv2.putText(canvas, "ign", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                continue
            dim_tag = None
            if not lab and tid not in rescues:
                if tid in auto_out:
                    dim_tag = "out"
                elif tid in kit_flag:
                    dim_tag = "kit?"
            if dim_tag:
                color = SELECTED if sel else DIMMED
                draw_dashed_rect(canvas, (x1, y1), (x2, y2), color, 2 if sel else 1)
                cv2.putText(canvas, dim_tag, (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                continue
            color = TEAM_DRAW[lab["team"]] if lab else UNLABELED
            thick = 3 if sel else (2 if lab else 1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), SELECTED if sel else color, thick)
            tag = f"{lab['team'][-1].upper()}{lab['number']}" if lab else f"#{tid}"
            cv2.putText(canvas, tag, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for i, mb in enumerate(manual_boxes):
            x1, y1, x2, y2 = (int(v) for v in mb["box"])
            sel = selected == ("m", i)
            color = SELECTED if sel else MANUAL
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if sel else 2)
            tag = mb["key"] if mb.get("key") else "new"
            cv2.putText(canvas, tag, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for _, r in ball_rows.iterrows():
            c = (int((r["x1"] + r["x2"]) / 2), int((r["y1"] + r["y2"]) / 2))
            cv2.circle(canvas, c, 8, BALL, 1, cv2.LINE_AA)
        for bx, by in ball_marks:
            cv2.circle(canvas, (int(bx), int(by)), 6, BALL, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(bx), int(by)), 6, (0, 0, 0), 1, cv2.LINE_AA)
        if mouse["down"] and mouse["cur"]:
            cv2.rectangle(canvas, mouse["down"], mouse["cur"], MANUAL, 1)

        if ball_mode:
            line1 = f"BALL MODE — click the ball ({len(ball_marks)} mark(s) on this frame)"
            line2 = "o / ESC = exit ball mode   d = undo last mark"
        elif selected is not None:
            pend_t = entry_team[-1].upper() if entry_team else "_"
            note = " (pre-filled from kit)" if prefilled and entry_team else ""
            what = f"Track #{selected[1]}" if selected[0] == "t" else "Manual box"
            line1 = f"{what} selected — typing: {pend_t}{entry_num or '_'}{note}   (a/b, digits, ENTER = confirm)"
            line2 = "x = ignore person   d = remove label   ESC = deselect   BACKSPACE = delete digit"
        else:
            n_lab = sum(1 for tid in frame_tids if label_for(tid))
            n_ign = sum(1 for tid in frame_tids if is_ignored(tid))
            n_dim = sum(1 for tid in frame_tids
                        if not label_for(tid) and not is_ignored(tid)
                        and tid not in rescues and (tid in auto_out or tid in kit_flag))
            line1 = (f"Anchor {frame_idx}: {n_lab} labeled, {n_ign} ignored, "
                     f"{n_dim} auto-dimmed, {len(manual_boxes)} manual, {len(ball_marks)} ball")
            line2 = "CLICK = select/rescue   DRAG = add missed player   o = ball   ENTER = next frame   Q = save+quit"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(canvas, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(canvas, line2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow(win, canvas)

    def resolve_click(pt):
        """-> ("m", idx) | ("t", tid) | None. Manual boxes win (smallest)."""
        hits = [(i, mb["box"]) for i, mb in enumerate(manual_boxes)
                if mb["box"][0] <= pt[0] <= mb["box"][2] and mb["box"][1] <= pt[1] <= mb["box"][3]]
        if hits:
            i, _ = min(hits, key=lambda h: (h[1][2] - h[1][0]) * (h[1][3] - h[1][1]))
            return ("m", i)
        row = resolve_click_to_track(rows, pt)
        return ("t", int(row["track_id"])) if row is not None else None

    def confirm_label():
        nonlocal selected, entry_team, entry_num, prefilled
        if not entry_team or not entry_num:
            print("  Need a team (a/b) and a number before ENTER")
            return
        keyname = f"{entry_team[-1]}{int(entry_num)}"
        # One body per number: evict the key from other boxes on this frame.
        for t in [t for t, l in new_labels.items() if l["key"] == keyname]:
            if selected != ("t", t):
                del new_labels[t]
                print(f"  {keyname} moved off track #{t}")
        for mb in manual_boxes:
            if mb.get("key") == keyname and selected != ("m", manual_boxes.index(mb)):
                mb["key"] = None
                print(f"  {keyname} moved off a manual box")
        if selected[0] == "t":
            tid = selected[1]
            new_labels[tid] = {
                "key": keyname, "team": entry_team, "number": int(entry_num),
                "image": foot_of(tid), "track_id_at_label": tid,
            }
            ignores.pop(tid, None)
            if tid in known_ignored or tid in auto_out or tid in kit_flag:
                rescues.setdefault(tid, {"image": foot_of(tid), "track_id_at_label": tid})
            print(f"  Track #{tid} -> {keyname}")
        else:
            mb = manual_boxes[selected[1]]
            mb.update({"key": keyname, "team": entry_team, "number": int(entry_num)})
            print(f"  Manual box -> {keyname} (tools/retrack_manual.py will track them)")
        selected, entry_team, entry_num, prefilled = None, None, "", False

    while True:
        redraw()
        key = cv2.waitKey(30) & 0xFF

        if mouse["click"] is not None:
            pt = mouse["click"]
            mouse["click"] = None
            if ball_mode:
                ball_marks.append([round(float(pt[0]), 1), round(float(pt[1]), 1)])
                print(f"  Ball marked at {pt} on frame {frame_idx}")
                continue
            hit = resolve_click(pt)
            if hit is None:
                continue
            if hit[0] == "t":
                tid = hit[1]
                if (tid in auto_out or tid in kit_flag) and tid not in rescues \
                        and not label_for(tid) and not is_ignored(tid):
                    rescues[tid] = {"image": foot_of(tid), "track_id_at_label": tid}
                    print(f"  Track #{tid} rescued from the auto-filter — label it, or x to re-ignore")
                select_track(tid)
            else:
                selected = hit
                entry_team, entry_num, prefilled = None, "", False
                if manual_boxes[hit[1]].get("key"):
                    print(f"  Manual box labeled {manual_boxes[hit[1]]['key']} "
                          "(type to replace, d = remove label, x = delete box)")
            continue

        if mouse["drag"] is not None:
            box = mouse["drag"]
            mouse["drag"] = None
            if ball_mode:
                continue
            overlap = [(box_iou(box, (r["x1"], r["y1"], r["x2"], r["y2"])), int(r["track_id"]))
                       for _, r in rows.iterrows()]
            best = max(overlap, default=(0.0, None))
            if best[0] >= 0.4:
                print(f"  That person is already tracked (#{best[1]}) — selecting them instead")
                if best[1] in auto_out or best[1] in kit_flag:
                    rescues.setdefault(best[1], {"image": foot_of(best[1]),
                                                 "track_id_at_label": best[1]})
                select_track(best[1])
            else:
                manual_boxes.append({"box": [round(float(v), 1) for v in box], "key": None})
                selected = ("m", len(manual_boxes) - 1)
                entry_team, entry_num, prefilled = None, "", False
                print("  Manual box added — now type its identity (a/b + number)")
            continue

        if key == 255:
            continue
        if key in (ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            return _pack(new_labels, ignores, rescues, manual_boxes, ball_marks), True
        if key in (ord("o"), ord("O")):
            ball_mode = not ball_mode
            selected, entry_team, entry_num, prefilled = None, None, "", False
            continue

        if ball_mode:
            if key == 27:
                ball_mode = False
            elif key in (ord("d"), ord("D")) and ball_marks:
                gone = ball_marks.pop()
                print(f"  Removed ball mark at ({gone[0]:.0f}, {gone[1]:.0f})")
            continue

        if selected is None:
            if key in (13, 10):  # done with this frame
                break
            continue

        # A box is selected: build the identity / curate.
        if key == 27:  # ESC
            selected, entry_team, entry_num, prefilled = None, None, "", False
        elif key in (ord("a"), ord("A")):
            entry_team, prefilled = "team_a", False
        elif key in (ord("b"), ord("B")):
            entry_team, prefilled = "team_b", False
        elif ord("0") <= key <= ord("9"):
            if len(entry_num) < 2:
                entry_num += chr(key)
        elif key in (8, 127):  # backspace
            entry_num = entry_num[:-1]
        elif key in (ord("x"), ord("X")):
            if selected[0] == "m":
                manual_boxes.pop(selected[1])
                print("  Manual box deleted")
            else:
                tid = selected[1]
                if tid in ignores:
                    del ignores[tid]
                    print(f"  Track #{tid} un-ignored")
                elif tid in known_ignored:
                    if tid in rescues:
                        del rescues[tid]
                        print(f"  Track #{tid} back to ignored (earlier anchor's decision)")
                    else:
                        rescues[tid] = {"image": foot_of(tid), "track_id_at_label": tid}
                        print(f"  Track #{tid} un-ignored (was ignored on an earlier anchor)")
                else:
                    ignores[tid] = {"image": foot_of(tid), "track_id_at_label": tid}
                    new_labels.pop(tid, None)
                    rescues.pop(tid, None)
                    print(f"  Track #{tid} IGNORED — excluded from all stats")
            selected, entry_team, entry_num, prefilled = None, None, "", False
        elif key in (ord("d"), ord("D")):
            if selected[0] == "m":
                mb = manual_boxes[selected[1]]
                if mb.get("key"):
                    print(f"  Manual box: label {mb['key']} removed")
                    mb.update({"key": None, "team": None, "number": None})
            elif selected[1] in new_labels:
                del new_labels[selected[1]]
                print(f"  Track #{selected[1]}: label removed")
            elif selected[1] in known_labels:
                print(f"  Track #{selected[1]} was labeled on an earlier anchor — "
                      "relabel it here to override; earlier anchors are kept as-is.")
            selected, entry_team, entry_num, prefilled = None, None, "", False
        elif key in (13, 10):  # confirm
            confirm_label()

    cv2.destroyAllWindows()
    return _pack(new_labels, ignores, rescues, manual_boxes, ball_marks), False


def _pack(new_labels, ignores, rescues, manual_boxes, ball_marks):
    labels = list(new_labels.values())
    for mb in manual_boxes:
        if mb.get("key"):
            x1, y1, x2, y2 = mb["box"]
            labels.append({
                "key": mb["key"], "team": mb["team"], "number": mb["number"],
                "image": [round((x1 + x2) / 2, 1), round(y2, 1)],
                "track_id_at_label": None, "manual": True,
            })
    return {
        "labels": labels,
        "ignores": list(ignores.values()),
        "rescues": list(rescues.values()),
        "manual_boxes": manual_boxes,
        "ball_marks": ball_marks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None, help="Video path (default: from meta.json)")
    parser.add_argument("--frame", type=int, default=0, help="Anchor frame index")
    parser.add_argument("--frames", default=None,
                        help="Comma-separated anchor frame indices, e.g. 0,900,1800")
    parser.add_argument("--pitch-margin-m", type=float, default=2.0,
                        help="Feet projecting farther than this outside the pitch are auto-dimmed")
    parser.add_argument("--no-auto-filter", action="store_true",
                        help="Disable the pitch-boundary/kit-color dimming")
    parser.add_argument("--team-a-name", default=None)
    parser.add_argument("--team-b-name", default=None)
    parser.add_argument("--output", default=None, help="Default: config/<match_id>_roster.json")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    video = args.video or meta["video"]
    tracks = load_tracks(args.match_id, args.tmp_dir)
    players = tracks[(tracks["class"] == "player") & (tracks["track_id"] >= 0)]
    balls = tracks[tracks["class"] == "ball"]
    if players.empty:
        raise SystemExit("No player tracks in tracks.parquet — run tools/run_tracking.py first")

    calib = None if args.no_auto_filter else load_calibration(args.match_id, args.config_dir)
    if calib is None and not args.no_auto_filter:
        print("WARNING: no calibration — the pitch-boundary filter is off; expect "
              "spectators among the boxes (use x / rely on the kit-color dimming)")
    if calib is not None:
        players, unit = apply_calibration(players, calib, out_cols=("pitch_x", "pitch_y"))
        length, width = pitch_dims(calib)

    team_map = {}
    if not args.no_auto_filter:
        try:
            team_map = load_team_map(args.match_id, args.tmp_dir)
        except SystemExit:
            print("No team_map.json — kit-color dimming/pre-fill off "
                  "(run tools/assign_teams.py for a smoother session)")

    frame_indices = ([int(f) for f in args.frames.split(",")] if args.frames else [args.frame])
    frame_indices = sorted(set(frame_indices))

    out_path = args.output or roster_path(args.match_id, args.config_dir)
    roster = load_roster(args.match_id, args.config_dir, args.output) or {
        "match_id": args.match_id, "video": str(video),
        "team_names": {}, "anchors": [],
    }
    if roster["anchors"]:
        state = "PARTIAL (saved on quit) " if roster.get("partial") else ""
        print(f"Existing {state}roster found ({len(roster['anchors'])} anchor(s)) — "
              "anchors merge; nothing you did before is lost.")
    if args.team_a_name:
        roster["team_names"]["team_a"] = args.team_a_name
    if args.team_b_name:
        roster["team_names"]["team_b"] = args.team_b_name

    def anchor_for(frame):
        return next((a for a in roster["anchors"] if a["frame"] == frame), {})

    def known_from_other_anchors(frame):
        labels, ignored = {}, set()
        for anchor in roster["anchors"]:
            if anchor["frame"] == frame:
                continue
            for lab in anchor["labels"]:
                if lab.get("track_id_at_label") is not None:
                    labels[lab["track_id_at_label"]] = lab
            for ig in anchor.get("ignores", []):
                ignored.add(ig["track_id_at_label"])
            for rc in anchor.get("rescues", []):
                ignored.discard(rc["track_id_at_label"])
        return labels, ignored

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open {video}")
    aborted = False
    for want in frame_indices:
        frame_idx = want
        rows = players[players["frame"] == frame_idx]
        if rows.empty:
            frame_idx = nearest_frame_with_players(players, want)
            rows = players[players["frame"] == frame_idx]
            print(f"Frame {want} has no player boxes — using nearest tracked frame {frame_idx} instead")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"Failed to read frame {frame_idx} from {video}")

        auto_out = set()
        if calib is not None:
            m = args.pitch_margin_m
            outside = (~rows["pitch_x"].between(-m, length + m)
                       | ~rows["pitch_y"].between(-m, width + m))
            auto_out = set(rows.loc[outside, "track_id"].astype(int))
        kit_flag = {int(t) for t in rows["track_id"] if team_map.get(int(t)) == "unknown"}
        known_labels, known_ignored = known_from_other_anchors(frame_idx)

        print(f"--- Anchor frame {frame_idx} ({len(rows)} tracked boxes, "
              f"{len(auto_out)} off-pitch, {len(kit_flag - auto_out)} kit-flagged) ---")
        content, aborted = label_one_frame(
            img, frame_idx, rows, balls[balls["frame"] == frame_idx],
            known_labels, known_ignored, auto_out, kit_flag, team_map,
            anchor_for(frame_idx))
        if any(content[k] for k in ("labels", "ignores", "rescues", "manual_boxes", "ball_marks")):
            roster["anchors"] = [a for a in roster["anchors"] if a["frame"] != frame_idx]
            roster["anchors"].append({"frame": frame_idx, **content})
        else:
            print(f"Nothing recorded on frame {frame_idx} — anchor not saved")
        if aborted:
            print("Quit early — saving what you've done so far (roster flagged partial).")
            break
    cap.release()

    if not any(any(a.get(k) for k in ("labels", "ignores", "rescues", "manual_boxes", "ball_marks"))
               for a in roster["anchors"]):
        raise SystemExit("Nothing recorded on any anchor — nothing written")

    if aborted:
        roster["partial"] = True
    else:
        roster.pop("partial", None)

    roster["anchors"].sort(key=lambda a: a["frame"])
    per_team = {"team_a": set(), "team_b": set()}
    n_ign = n_manual = n_ball = 0
    for anchor in roster["anchors"]:
        for lab in anchor["labels"]:
            per_team[lab["team"]].add(lab["number"])
        n_ign += len(anchor.get("ignores", []))
        n_manual += sum(1 for mb in anchor.get("manual_boxes", []))
        n_ball += len(anchor.get("ball_marks", []))
    for team, nums in per_team.items():
        name = roster["team_names"].get(team, team)
        print(f"{name}: {len(nums)} player(s) — numbers {sorted(nums)}")
        if len(nums) > 11:
            print(f"WARNING: {name} has more than 11 distinct numbers — subs are fine, "
                  "but check for typos (a wrong number creates a phantom 12th player).")
    if n_ign:
        print(f"{n_ign} person(s) ignored — excluded from all stats")

    write_json(out_path, roster)
    print(f"Wrote {out_path}")
    if n_manual or n_ball:
        print(f"{n_manual} manual box(es), {n_ball} ball mark(s) — run "
              f"python tools/retrack_manual.py --match-id {args.match_id} BEFORE assign_identities")
    else:
        print("Next: python tools/assign_identities.py --match-id", args.match_id)


if __name__ == "__main__":
    main()
