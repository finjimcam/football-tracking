"""Lines and player highlights drawn on a clip from the Lines tab's Clips panel
(issue #58), and the small preview videos the panel plays.

A clip's marks are "slots", as in the phase line picker (cut_phase_clips.py):
one slot follows one player through the clip from the frames where the analyst
tapped him (anchors), with the same chain follower and gap filling.

    line "back"  back line, in the player's team colour
    line "mid"   midfield line, white
    line "hl"    highlighted player: a ring and his name

Saved in config/<id>_clips_marks.json:

    {"clips": {"<clip id>": {"slots": [{"line": "hl", "anchors": [[frame, track_id], ...]}]}}}

export_clips.py draws them into the exported MP4s; app/clips.py serves them.
"""
import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _common import CONFIG_DIR, match_dir
from cut_phase_clips import build_chains, chains_by_frame, index_rows, norm_anchors
from gap_status import observed

LINES = ("back", "mid", "hl")
HL_COLOR = (77, 216, 255)  # BGR of #FFD84D, the app's accent


def marks_path(match_id, config_dir=CONFIG_DIR):
    return Path(config_dir) / f"{match_id}_clips_marks.json"


def load_marks(path):
    """{clip_id: [slot, ...]} (missing file -> {})."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {k: v.get("slots", []) for k, v in json.load(f).get("clips", {}).items()}


def save_marks(path, marks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"clips": {k: {"slots": v} for k, v in sorted(marks.items()) if v}}, f, indent=1)
    tmp.replace(path)


def clean_slots(slots):
    out = []
    for sl in slots or []:
        if sl.get("line") not in LINES:
            raise ValueError("slot line must be back, mid or hl")
        anchors = norm_anchors(sl.get("anchors", []))
        if not anchors:
            raise ValueError("every slot needs an anchor")
        out.append({"line": sl["line"], "anchors": anchors})
    return out


def clip_rows(match_id, tmp_dir, config_dir, start, end, fps):
    """{frame: DataFrame} of every detected player (both teams) between start and end, with the
    Tracking tab's hand fixes applied."""
    from track_edits import apply_to

    path = match_dir(match_id, tmp_dir) / "tracks.parquet"
    t = pd.read_parquet(path, filters=[("frame", ">=", int(start)), ("frame", "<=", int(end))])
    t = apply_to(t, match_id, config_dir, fps, quiet=True)
    t = observed(t[t["class"] == "player"])
    t = t[t["pitch_x"].notna()]
    return {int(f): g for f, g in t.groupby("frame")}


def chains(rows_by_frame, slots, start, end, fps, H_at):
    return build_chains(index_rows(rows_by_frame), slots, start, end, fps, H_at)


def hex_bgr(h, default=(221, 171, 108)):
    try:
        h = h.lstrip("#")
        return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)
    except (AttributeError, ValueError, IndexError):
        return default


def draw_marks(canvas, items, labels, line_colors, scale=1.0):
    """items: chains_by_frame()[frame] = [(line, foot_x, foot_y, pitch_y, tid, status)], with
    chain index appended by marks_by_frame(); labels: {chain index: name}; line_colors:
    {chain index: BGR}. Back / midfield lines are joined across the pitch (sorted by pitch y)."""
    by_line = {}
    for line, fx, fy, py, _tid, status, ci in items:
        x, y = int(fx * scale), int(fy * scale)
        thick = 1 if status == "filled" else 2
        if line == "hl":
            cv2.ellipse(canvas, (x, y), (int(20 * scale) + 4, int(7 * scale) + 2), 0, 0, 360, HL_COLOR, thick + 1,
                        cv2.LINE_AA)
            name = labels.get(ci)
            if name:
                fs = 0.6 * max(scale, 0.6)
                (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
                ty = max(th + 4, y - int(90 * scale))
                cv2.rectangle(canvas, (x - tw // 2 - 5, ty - th - 5), (x + tw // 2 + 5, ty + 5), (20, 20, 20), -1)
                cv2.putText(canvas, name, (x - tw // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, HL_COLOR, 2, cv2.LINE_AA)
            continue
        c = line_colors.get(ci, (255, 255, 255))
        cv2.ellipse(canvas, (x, y), (int(14 * scale) + 2, int(5 * scale) + 1), 0, -35, 235, c, thick, cv2.LINE_AA)
        by_line.setdefault((line, c), []).append((py, x, y))
    for (_line, c), pts in by_line.items():
        if len(pts) >= 2:
            arr = np.array([[x, y] for _, x, y in sorted(pts)], np.int32)
            cv2.polylines(canvas, [arr], False, c, max(2, int(3 * scale)), cv2.LINE_AA)


def marks_by_frame(chain_list):
    """{frame: [(line, fx, fy, py, tid, status, chain index)]}"""
    out = {}
    for ci, ch in enumerate(chain_list):
        for f, items in chains_by_frame([ch]).items():
            out.setdefault(f, []).extend(it + (ci,) for it in items)
    return out


def slot_labels(slots, names):
    """{chain index: player name} from each slot's first anchor's track."""
    out = {}
    for i, sl in enumerate(slots):
        n = names.get(int(sl["anchors"][0][1]))
        if n:
            out[i] = n
    return out


def slot_colors(slots, team_of, colors):
    """{chain index: BGR}: back line in the team colour of its player, midfield white."""
    out = {}
    for i, sl in enumerate(slots):
        if sl["line"] == "back":
            team = team_of.get(int(sl["anchors"][0][1]))
            out[i] = hex_bgr((colors or {}).get(team) or ("#6CABDD" if team != "team_b" else "#F0F0F0"))
        else:
            out[i] = (255, 255, 255)
    return out


def make_preview(video, start, end, fps, out, width=1280, crf=28):
    """A small H.264 MP4 of frames start..end for the browser: short keyframe interval (quick
    seeking), moov atom up front (plays while it downloads). Decoded, scaled and encoded by PyAV
    without a round trip through numpy (~3x faster than via OpenCV)."""
    import av

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    src = av.open(str(video))
    vs = src.streams.video[0]
    vs.thread_type = "AUTO"
    w = min(int(width), vs.codec_context.width) // 2 * 2
    h = int(round(vs.codec_context.height * w / vs.codec_context.width)) // 2 * 2
    tb = float(vs.time_base)
    t0 = float(vs.start_time or 0) * tb
    rate = Fraction(fps).limit_denominator(1000)
    box = av.open(str(tmp), "w", options={"movflags": "faststart"})
    st = box.add_stream("libx264", rate=rate)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": str(crf), "preset": "ultrafast", "g": "15", "tune": "fastdecode"}
    src.seek(int((t0 + start / fps - 1.0) / tb), stream=vs, backward=True)  # keyframe before start
    n = 0
    for fr in src.decode(vs):
        if fr.pts is None:
            continue
        idx = int(round((fr.pts * tb - t0) * fps))
        if idx < start:
            continue
        if idx > end:
            break
        o = fr.reformat(width=w, height=h, format="yuv420p")
        o.pts = n
        o.time_base = 1 / rate
        for pkt in st.encode(o):
            box.mux(pkt)
        n += 1
    for pkt in st.encode():
        box.mux(pkt)
    box.close()
    src.close()
    tmp.replace(out)
    return n
