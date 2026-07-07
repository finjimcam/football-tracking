"""Probe a match video with OpenCV, mint its match_id, and write
.tmp/<match_id>/meta.json (fps, duration, resolution, frame count, config).

Exits non-zero if the video can't be opened or decoded. Prints a WARNING (but
still writes meta.json) if the duration looks implausible for a match
recording — the workflow leaves that judgment call to the agent.

Usage:
    python tools/ingest_video.py --video match.mp4
    python tools/ingest_video.py --video match.mp4 \
        --config '{"team_a_name": "Reds", "team_b_name": "Blues", "kickoff": "2026-07-05T15:00"}'
"""
import argparse
import json
from pathlib import Path

import cv2

from _common import compute_match_id, match_dir, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to the match video (.mp4)")
    parser.add_argument("--config", default="{}",
                        help="JSON match config: team_a_name, team_b_name, kickoff, camera — or a path to a JSON file")
    parser.add_argument("--tmp-dir", default=".tmp", help="Root for pipeline intermediates")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    if Path(args.config).is_file():
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        config = json.loads(args.config)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open {video} — file may be corrupt or an unsupported codec")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()

    if not ok:
        raise SystemExit(f"Could not decode the first frame of {video}")
    if fps <= 0 or frame_count <= 0:
        raise SystemExit(f"Implausible probe result (fps={fps}, frames={frame_count}) — container metadata is broken")

    duration_s = frame_count / fps
    match_id = compute_match_id(video)

    meta = {
        "match_id": match_id,
        "video": str(video.resolve()),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": round(duration_s, 2),
        "width": width,
        "height": height,
        "config": config,
    }
    out = write_json(match_dir(match_id, args.tmp_dir, create=True) / "meta.json", meta)

    print(f"match_id: {match_id}")
    print(f"  {width}x{height} @ {fps:.2f} fps, {frame_count} frames, {duration_s / 60:.1f} min")
    print(f"  meta written to {out}")
    if duration_s < 300:
        print(f"WARNING: duration is only {duration_s / 60:.1f} min — short for a match recording; confirm this is the right file")
    elif duration_s > 3 * 3600:
        print(f"WARNING: duration is {duration_s / 3600:.1f} h — long for a single match; confirm this is the right file")


if __name__ == "__main__":
    main()
