"""Assign each player track to a team by jersey color →
.tmp/<match_id>/team_map.json (track_id → team_a | team_b | unknown).

For each player track: sample up to N of its boxes across the video, crop the
upper half (jersey), mask out pitch-green pixels, take the median color.
K-means (k=2) over the per-track colors in Lab space splits the teams; tracks
far from both cluster centers (referees, keepers) → unknown.

team_a is the larger cluster. Prints each cluster's mean BGR color so the
user can map team_a/team_b to real team names, plus the cluster balance —
if one team gets < 25% of assigned tracks the workflow says stop and flag.

Usage:
    python tools/assign_teams.py --match-id <match_id>
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np

from _common import load_meta, load_tracks, match_dir, write_json


def jersey_color(crop):
    """Median non-green color of a jersey crop (BGR), or None if the crop is
    unusable. Green masking keeps grass background out of the median."""
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h > 35) & (h < 90) & (s > 40) & (v > 40)
    pixels = crop[~green]
    if len(pixels) < 30:  # crop is nearly all grass — fall back to everything
        pixels = crop.reshape(-1, 3)
    return np.median(pixels, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", default=None, help="Video path (default: from meta.json)")
    parser.add_argument("--samples-per-track", type=int, default=10)
    parser.add_argument("--outlier-scale", type=float, default=2.5,
                        help="Tracks farther than this × the median own-cluster distance go to `unknown`")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    video = args.video or meta["video"]
    tracks = load_tracks(args.match_id, args.tmp_dir)
    players = tracks[(tracks["class"] == "player") & (tracks["track_id"] >= 0)]
    if players.empty:
        raise SystemExit("No player tracks in tracks.parquet — check the tracking pass")

    # Sample up to N evenly spaced boxes per track, then group by frame so we
    # can read the video once, sequentially.
    frame_crops = defaultdict(list)  # frame -> [(track_id, box)]
    for tid, grp in players.groupby("track_id"):
        grp = grp.sort_values("frame")
        idx = np.linspace(0, len(grp) - 1, min(args.samples_per_track, len(grp))).astype(int)
        for _, row in grp.iloc[idx].iterrows():
            frame_crops[int(row["frame"])].append(
                (int(tid), (row["x1"], row["y1"], row["x2"], row["y2"])))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open {video}")
    track_colors = defaultdict(list)
    needed = sorted(frame_crops)
    frame_idx = 0
    for target in needed:
        while frame_idx < target:  # skip decode of unneeded frames
            cap.grab()
            frame_idx += 1
        ok, frame = cap.read()
        frame_idx += 1
        if not ok:
            break
        H, W = frame.shape[:2]
        for tid, (x1, y1, x2, y2) in frame_crops[target]:
            x1, x2 = max(0, int(x1)), min(W, int(x2))
            y1, y2 = max(0, int(y1)), min(H, int(y2))
            color = jersey_color(frame[y1:(y1 + y2) // 2, x1:x2])  # upper half
            if color is not None:
                track_colors[tid].append(color)
    cap.release()

    tids = sorted(players["track_id"].unique())
    sampled = [t for t in tids if track_colors[t]]
    if len(sampled) < 2:
        raise SystemExit("Fewer than 2 tracks produced usable jersey crops — cannot cluster")

    # Per-track median color, clustered in Lab (perceptually uniform-ish).
    bgr = np.array([np.median(track_colors[t], axis=0) for t in sampled], dtype=np.float32)
    lab = cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(lab, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    dist_own = np.linalg.norm(lab - centers[labels], axis=1)
    threshold = args.outlier_scale * max(np.median(dist_own), 1e-6)

    # Larger cluster = team_a (deterministic naming; user maps to real names).
    order = [0, 1] if np.sum(labels == 0) >= np.sum(labels == 1) else [1, 0]
    cluster_team = {order[0]: "team_a", order[1]: "team_b"}

    team_map = {str(t): "unknown" for t in tids}
    for t, lbl, d in zip(sampled, labels, dist_own):
        team_map[str(t)] = cluster_team[lbl] if d <= threshold else "unknown"

    out_path = write_json(match_dir(args.match_id, args.tmp_dir) / "team_map.json", team_map)

    counts = {v: sum(1 for x in team_map.values() if x == v) for v in ("team_a", "team_b", "unknown")}
    assigned = counts["team_a"] + counts["team_b"]
    for team, cluster in (("team_a", order[0]), ("team_b", order[1])):
        mean_bgr = bgr[labels == cluster].mean(axis=0).astype(int).tolist()
        print(f"{team}: {counts[team]} tracks, mean jersey BGR {mean_bgr}")
    print(f"unknown: {counts['unknown']} tracks (outliers/referees/keepers)")
    print(f"Wrote {out_path}")

    if assigned and min(counts["team_a"], counts["team_b"]) / assigned < 0.25:
        print("WARNING: cluster balance is worse than 25/75 — likely a clustering failure "
              "(similar kits or keeper-heavy sample). The workflow says stop and check before building a report.")


if __name__ == "__main__":
    main()
