"""Export one team's kept clips from the Lines tab's Clips panel as a zip of MP4s (issue #58).

Reads .tmp/<id>/clips.json (detect_clips.py) and the keep / trim choices in
config/<id>_clips_export.json (kept unless dropped; "add_f" frames added to the
end with the panel's "+5 s"), and writes

    outputs/videos/<id>_<team>_clips.zip
        Set pieces/Corners/03_corner_23m41s.mp4
        Shots & key chances/Shots/...
        Passes into the final third/...
        clips.csv

Each clip has a bar along the bottom: what it is, the team, the player (when
known) and the match clock. Lines and highlighted players drawn on a clip in
the panel (config/<id>_clips_marks.json, clip_marks.py) are drawn in too. Clips with no team (the detector couldn't tell)
go in with both teams. --group limits it to one heading.

    python tools/export_clips.py --match-id <id> --team team_b [--group set_pieces]
"""
import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import cv2

import clip_marks as CM
from _common import (CONFIG_DIR, OUTPUTS_DIR, Progress, VideoWriter, clock_label, load_calibration, load_meta,
                     load_player_map, match_config, match_dir)

GROUP_TITLES = {"set_pieces": "Set pieces", "chances": "Shots & key chances",
                "final_third": "Passes into the final third"}
KIND_TITLES = {"goal_kick": "Goal kicks", "throw_in": "Throw-ins", "corner": "Corners", "free_kick": "Free kicks",
               "penalty": "Penalties", "shot": "Shots", "box_entry": "Box entries", "pass": "Passes"}
KIND_LABEL = {"goal_kick": "Goal kick", "throw_in": "Throw-in", "corner": "Corner", "free_kick": "Free kick",
              "penalty": "Penalty", "shot": "Shot", "box_entry": "In the box", "pass": "Pass into the final third"}
BAR_H = 30


def export_path(match_id, config_dir=CONFIG_DIR):
    return Path(config_dir) / f"{match_id}_clips_export.json"


def zip_path(match_id, team, group=None, outputs_dir=OUTPUTS_DIR):
    return Path(outputs_dir) / "videos" / f"{match_id}_{team}{'_' + group if group else ''}_clips.zip"


def load_clips(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_export(path):
    """{clip_id: {"keep": bool, "in": frame|None, "out": frame|None}} (missing file -> {})."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return dict(json.load(f).get("clips", {}))


def save_export(path, entries):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"clips": dict(sorted(entries.items()))}, f, indent=1)
    tmp.replace(path)


def extended(clips, exp, last_frame=None):
    """The clips with their end moved on by the time added in the panel ("+5 s": the export
    entry's "add_f" frames), at most to last_frame. An extended clip keeps its detected end in
    "detected_end" and the frames added in "added_f"."""
    out = []
    for c in clips:
        add = int((exp.get(c["id"]) or {}).get("add_f") or 0)
        if add > 0:
            e = c["end"] + add if last_frame is None else min(c["end"] + add, int(last_frame))
            c = dict(c, end=e, detected_end=c["end"], added_f=e - c["end"])
        out.append(c)
    return out


def export_range(clip, entry):
    """(keep, first_frame, last_frame): the trim points clamped to the clip, ignored if they cross."""
    s, e = int(clip["start"]), int(clip["end"])
    if not entry:
        return True, s, e
    a = s if entry.get("in") is None else min(max(int(entry["in"]), s), e)
    b = e if entry.get("out") is None else min(max(int(entry["out"]), s), e)
    if b <= a:
        a, b = s, e
    return bool(entry.get("keep", True)), a, b


def plan(clips, exp, team, group=None):
    """[(clip, first, last, relative path)] of the kept clips, in heading / subheading / time order."""
    order = {g: i for i, g in enumerate(GROUP_TITLES)}
    korder = {k: i for i, k in enumerate(KIND_TITLES)}
    chosen = []
    for c in clips:
        if c.get("team") not in (team, None) or (group and c["group"] != group):
            continue
        keep, a, b = export_range(c, exp.get(c["id"]))
        if keep:
            chosen.append((c, a, b))
    chosen.sort(key=lambda t: (order.get(t[0]["group"], 9), korder.get(t[0]["kind"], 99), t[0]["frame"]))
    out, n = [], {}
    for c, a, b in chosen:
        n[c["kind"]] = n.get(c["kind"], 0) + 1
        clk = (c.get("clock") or "").replace(":", "m")
        name = f"{n[c['kind']]:02d}_{c['kind']}" + (f"_{clk}s" if clk else f"_f{c['frame']}") + ".mp4"
        out.append((c, a, b, f"{GROUP_TITLES[c['group']]}/{KIND_TITLES[c['kind']]}/{name}"))
    return out


def bar_text(c, team_names, clock):
    who = c.get("player_name")
    if c["kind"] == "pass" and c.get("to_player_name"):
        who = f"{who or '?'} to {c['to_player_name']}"
    team = team_names.get(c.get("team"), "team unknown")
    what = KIND_LABEL.get(c["kind"], c["kind"])
    if c.get("also"):
        what += " (+ " + ", ".join(KIND_LABEL.get(k, k).lower() for k in c["also"]) + ")"
    return "  |  ".join(x for x in (what, team, who, clock) if x)


def marks_plan(match_id, clip, s, b, fps, tmp_dir, config_dir, cfg, marks, H_at, people):
    """(per-frame marks, labels, colours) for one clip, or None when it has no marks."""
    slots = marks.get(clip["id"])
    if not slots:
        return None
    names, team_of = people
    rows = CM.clip_rows(match_id, tmp_dir, config_dir, s, b, fps)
    chain_list = CM.chains(rows, slots, s, b, fps, H_at)
    return (CM.marks_by_frame(chain_list), CM.slot_labels(slots, names),
            CM.slot_colors(slots, team_of, cfg.get("colors")))


def people(match_id, tmp_dir):
    """({track id: "7 Name"}, {track id: team})"""
    pm = load_player_map(match_id, tmp_dir) or {"players": {}}
    names, team_of = {}, {}
    for p in pm.get("players", {}).values():
        label = p.get("name") or (f"#{p['number']}" if p.get("number") is not None else None)
        if p.get("name") and p.get("number") is not None:
            label = f"{p['number']} {p['name']}"
        for t in p.get("track_ids", []):
            team_of[int(t)] = p.get("team")
            if label:
                names[int(t)] = label
    return names, team_of


def draw_bar(canvas, text, frac, color=(80, 200, 255)):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, h - BAR_H), (w, h), (20, 20, 20), -1)
    cv2.putText(canvas, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    x0, x1 = w - 200, w - 12  # how far through the clip
    cv2.rectangle(canvas, (x0, h - 19), (x1, h - 11), (90, 90, 90), -1)
    cv2.rectangle(canvas, (x0, h - 19), (x0 + int((x1 - x0) * frac), h - 11), color, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--team", choices=("team_a", "team_b"), default="team_a")
    ap.add_argument("--group", choices=tuple(GROUP_TITLES), default=None, help="Only this heading")
    ap.add_argument("--output", default=None, help="The .zip (default outputs/videos/<id>_<team>_clips.zip)")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--tmp-dir", default=".tmp")
    ap.add_argument("--outputs-dir", default=str(OUTPUTS_DIR))
    a = ap.parse_args()

    mdir = match_dir(a.match_id, a.tmp_dir)
    cp = mdir / "clips.json"
    if not cp.exists():
        raise SystemExit(f"{cp} not found: run detect_clips.py first")
    d = load_clips(cp)
    meta = load_meta(a.match_id, a.tmp_dir)
    exp = load_export(export_path(a.match_id, a.config_dir))
    d["clips"] = extended(d["clips"], exp, meta.get("frame_count", 0) - 1 if meta.get("frame_count") else None)
    cfg = match_config(a.match_id, a.tmp_dir, a.config_dir)
    fps = float(meta["fps"])
    items = plan(d["clips"], exp, a.team, a.group)
    if not items:
        raise SystemExit("No kept clips for this team: tick some in the Clips panel")
    out = Path(a.output) if a.output else zip_path(a.match_id, a.team, a.group, a.outputs_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_suffix(".parts")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    names = d.get("teams", {})
    marks = CM.load_marks(CM.marks_path(a.match_id, a.config_dir))
    H_at, ppl = None, None
    if any(marks.get(c["id"]) for c, *_ in items):
        from cut_phase_clips import homography_lookup
        try:
            H_at = homography_lookup(load_calibration(a.match_id, a.config_dir))
        except SystemExit:
            H_at = None
        ppl = people(a.match_id, a.tmp_dir)
    scale = meta["width"] / 1920
    cap = cv2.VideoCapture(str(meta["video"]))
    prog = Progress(sum(b - s + 1 for _, s, b, _ in items), "exporting clips")
    done = 0
    rows = []
    for c, s, b, rel in items:
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(dst, fps, meta["width"], meta["height"], crf=a.crf)
        mk = marks_plan(a.match_id, c, s, b, fps, a.tmp_dir, a.config_dir, cfg, marks, H_at, ppl) if ppl else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        for f in range(s, b + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if mk and mk[0].get(f):
                CM.draw_marks(frame, mk[0][f], mk[1], mk[2], scale)
            draw_bar(frame, bar_text(c, names, clock_label(f, cfg, fps) or ""), (f - s) / max(b - s, 1))
            writer.write(frame)
            done += 1
            prog.update(done)
        writer.close()
        rows.append({"file": rel, "heading": GROUP_TITLES[c["group"]], "type": KIND_TITLES[c["kind"]],
                     "team": names.get(c.get("team"), ""), "clock": c.get("clock", ""), "frame": c["frame"],
                     "first_frame": s, "last_frame": b, "seconds": round((b - s + 1) / fps, 1),
                     "also": "; ".join(KIND_LABEL.get(k, k) for k in c.get("also", [])),
                     "marks": len(marks.get(c["id"], [])),
                     "player": c.get("player_name", ""), "to_player": c.get("to_player_name", ""),
                     "x": c.get("x", ""), "y": c.get("y", "")})
    cap.release()
    with open(work / "clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # MP4s don't compress further
        for p in sorted(work.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(work)))
    tmp.replace(out)
    shutil.rmtree(work)  # the zip is the deliverable; don't keep two copies on the 15 GB disk
    prog.done()
    print(f"Wrote {out}: {len(items)} clips, {done / fps / 60:.1f} min, {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
