"""Per-team average positions and positional heatmaps →
.tmp/<match_id>/positions.json + .tmp/<match_id>/figures/*.png.
Reads only pipeline artifacts, never the video.

Positions are player foot-points in pitch meters when a calibration exists
(with a pitch outline drawn on the figures), otherwise raw pixels, flagged.

Usage:
    python tools/compute_positions.py --match-id <match_id>
"""
import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import (PITCH_LENGTH, PITCH_WIDTH, apply_calibration, load_calibration,
                     load_meta, load_player_map, load_team_map, load_tracks, match_dir,
                     resolve_track_teams, write_json)

TEAM_COLORS = {"team_a": "#d62728", "team_b": "#1f77b4"}


def draw_pitch(ax, length, width):
    """Standard pitch markings in meters, origin top-left, y down."""
    kw = dict(color="0.4", lw=1)
    ax.plot([0, length, length, 0, 0], [0, 0, width, width, 0], **kw)
    ax.plot([length / 2, length / 2], [0, width], **kw)
    ax.add_patch(plt.Circle((length / 2, width / 2), 9.15, fill=False, **kw))
    pw, gw = 40.32 / 2, 18.32 / 2
    for x0, box_dir in ((0, 1), (length, -1)):
        for depth, half in ((16.5, pw), (5.5, gw)):
            x1 = x0 + box_dir * depth
            ax.plot([x0, x1, x1, x0],
                    [width / 2 - half, width / 2 - half, width / 2 + half, width / 2 + half], **kw)
    ax.set_xlim(-2, length + 2)
    ax.set_ylim(width + 2, -2)  # y down, like image coordinates
    ax.set_aspect("equal")
    ax.axis("off")


def heatmap_figure(xy, title, unit, length, width, calibrated, out_path):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    extent = (0, length, 0, width)  # meters if calibrated, else pixel frame dims
    bins = (int(length), int(width)) if calibrated else (80, 50)
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins,
                                range=[extent[:2], extent[2:]])
    hist = cv2.GaussianBlur(hist.T.astype(np.float32), (0, 0), sigmaX=1.5)
    ax.imshow(hist, extent=(extent[0], extent[1], extent[3], extent[2]),
              cmap="hot", alpha=0.85, interpolation="bilinear", origin="upper")
    if calibrated:
        draw_pitch(ax, length, width)
    else:
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.invert_yaxis()
        ax.set_aspect("equal")
    ax.set_title(title + ("" if calibrated else "  [UNCALIBRATED — pixel coords]"))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--calibration", default=None, help="Calibration file (default: config/<match_id>_calibration.json)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    meta = load_meta(args.match_id, args.tmp_dir)
    team_map = load_team_map(args.match_id, args.tmp_dir)
    calib = load_calibration(args.match_id, args.config_dir, args.calibration)
    tracks, unit = apply_calibration(load_tracks(args.match_id, args.tmp_dir), calib)
    calibrated = calib is not None

    if calibrated:
        length, width = calib.get("pitch_length", PITCH_LENGTH), calib.get("pitch_width", PITCH_WIDTH)
    else:
        length, width = meta["width"], meta["height"]

    players = tracks[tracks["class"] == "player"].copy()
    # Roster-space teams with curation applied: ignored tracks never match a
    # team below, so spectators stay off the heatmaps.
    pm = load_player_map(args.match_id, args.tmp_dir)
    track_teams = resolve_track_teams(players["track_id"].unique(), team_map, pm)
    players["team"] = players["track_id"].map(track_teams)
    n_ignored = sum(1 for v in track_teams.values() if v == "ignored")
    if n_ignored:
        print(f"Excluding {n_ignored} ignored track(s) from position maps")

    fig_dir = match_dir(args.match_id, args.tmp_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    teams, figures = {}, {}
    for team in ("team_a", "team_b"):
        pts = players.loc[players["team"] == team, ["pitch_x", "pitch_y"]].to_numpy()
        if len(pts) == 0:
            print(f"WARNING: no points for {team} — skipping its heatmap")
            continue
        teams[team] = {
            "avg_x": round(float(pts[:, 0].mean()), 2),
            "avg_y": round(float(pts[:, 1].mean()), 2),
            "n_points": len(pts),
        }
        out = fig_dir / f"heatmap_{team}.png"
        heatmap_figure(pts, f"Positional heatmap — {team}", unit, length, width, calibrated, out)
        figures[f"heatmap_{team}"] = str(out)

    # Average-position map: one dot per track (players' typical spot), colored by team.
    fig, ax = plt.subplots(figsize=(9, 6.2))
    if calibrated:
        draw_pitch(ax, length, width)
    else:
        ax.set_xlim(0, length)
        ax.set_ylim(width, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    for team in teams:
        sub = players[players["team"] == team]
        avg = sub.groupby("track_id")[["pitch_x", "pitch_y"]].mean()
        ax.scatter(avg["pitch_x"], avg["pitch_y"], s=60, label=team,
                   color=TEAM_COLORS[team], edgecolors="white", zorder=3)
    ax.legend(loc="upper right")
    ax.set_title("Average track positions" + ("" if calibrated else "  [UNCALIBRATED — pixel coords]"))
    avg_path = fig_dir / "avg_positions.png"
    fig.savefig(avg_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    figures["avg_positions"] = str(avg_path)

    result = {"calibrated": calibrated, "unit": unit, "teams": teams, "figures": figures}
    out = write_json(match_dir(args.match_id, args.tmp_dir) / "positions.json", result)
    for team, t in teams.items():
        print(f"{team}: avg position ({t['avg_x']}, {t['avg_y']}) {unit}, {t['n_points']} points")
    print(f"Wrote {out} and {len(figures)} figures to {fig_dir}")


if __name__ == "__main__":
    main()
