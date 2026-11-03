"""Per-team average positions and positional heatmaps →
.tmp/<match_id>/positions.json + .tmp/<match_id>/figures/*.png.
Reads only pipeline artifacts, never the video.

Positions are player foot-points in pitch meters when a calibration exists
(with a pitch outline drawn on the figures), otherwise raw pixels, flagged.

With identities (player_map.json) + possession_timeline.parquet
(compute_possession.py) it also draws, per team, every player's average
position IN vs OUT of possession (avg_positions_<team>_by_phase.png) with
shape metrics - back-line height, team length and width - in positions.json
"by_phase". Team colours come from the match config "colors" if present.

Usage:
    python tools/compute_positions.py --match-id <match_id>
"""
import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (PITCH_LENGTH, PITCH_WIDTH, apply_calibration, load_calibration,
                     load_meta, load_player_map, load_team_map, load_tracks, match_config, match_dir,
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


def phase_figures(players, pm, timeline, cfg, length, width, fig_dir, figures):
    """Per team: average position of every identified player in possession vs
    out of possession (possession_timeline.parquet), one figure with two
    panels, plus shape metrics. Returns {team: {phase: metrics}}."""
    state = dict(zip(timeline["frame"].tolist(), timeline["state"].tolist()))
    t2p = pm["track_to_player"]
    info = pm["players"]
    ident = players[players["track_id"].isin(list(t2p))].copy()
    ident["state"] = ident["frame"].map(state)
    names = {"team_a": cfg.get("team_a_name", "team_a"), "team_b": cfg.get("team_b_name", "team_b")}
    colors = cfg.get("colors", TEAM_COLORS)
    out = {}
    for team in ("team_a", "team_b"):
        other = "team_b" if team == "team_a" else "team_a"
        sub_team = ident[ident["team"] == team]
        gk_x = sub_team[sub_team["track_id"].map(lambda t: info[t2p[t]].get("role") == "GK")]["pitch_x"].median()
        own_goal = 0.0 if gk_x < length / 2 else length
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))
        out[team] = {}
        for ax, (phase, holder) in zip(axes, (("in_possession", team), ("out_of_possession", other))):
            sub = sub_team[sub_team["state"] == holder]
            avg = sub.groupby("track_id")[["pitch_x", "pitch_y"]].mean()
            draw_pitch(ax, length, width)
            ax.scatter(avg["pitch_x"], avg["pitch_y"], s=170, color=colors.get(team, "#444"),
                       edgecolors="black", zorder=3)
            outfield, back = [], []
            for tid, r in avg.iterrows():
                p = info[t2p[tid]]
                label = (p.get("name") or str(p["number"])).split(" ")[-1]
                ax.annotate(label, (r["pitch_x"], r["pitch_y"]), xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=8, zorder=4)
                if p.get("role") != "GK":
                    outfield.append((r["pitch_x"], r["pitch_y"]))
                    if str(p.get("role", "")).startswith("line1"):
                        back.append(r["pitch_x"])
            if outfield:
                xy = np.array(outfield)
                m = {"frames": int(sub["frame"].nunique()),
                     "defensive_line_m_from_own_goal": round(float(abs(np.mean(back) - own_goal)), 1) if back else None,
                     "avg_shape_length_m": round(float(np.ptp(xy[:, 0])), 1),
                     "avg_shape_width_m": round(float(np.ptp(xy[:, 1])), 1)}
                out[team][phase] = m
                ax.set_title(f"{names[team]} {phase.replace('_', ' ')} - {m['frames'] / 25 / 60:.1f} min; "
                             f"back line {m['defensive_line_m_from_own_goal']} m from goal, "
                             f"avg-shape length {m['avg_shape_length_m']} m, width {m['avg_shape_width_m']} m", fontsize=9)
        attack = "->" if own_goal == 0 else "<-"
        fig.suptitle(f"{names[team]} average positions by phase (attacking {attack}); names role-inferred")
        path = fig_dir / f"avg_positions_{team}_by_phase.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figures[f"avg_positions_{team}_by_phase"] = str(path)
    return out


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

    phase_shape = {}
    tl_path = match_dir(args.match_id, args.tmp_dir) / "possession_timeline.parquet"
    if pm and calibrated and tl_path.exists():
        phase_shape = phase_figures(players, pm, pd.read_parquet(tl_path, columns=["frame", "state"]),
                                    match_config(args.match_id, args.tmp_dir, args.config_dir),
                                    length, width, fig_dir, figures)

    result = {"calibrated": calibrated, "unit": unit, "teams": teams, "figures": figures,
              "by_phase": phase_shape}
    out = write_json(match_dir(args.match_id, args.tmp_dir) / "positions.json", result)
    for team, t in teams.items():
        print(f"{team}: avg position ({t['avg_x']}, {t['avg_y']}) {unit}, {t['n_points']} points")
    print(f"Wrote {out} and {len(figures)} figures to {fig_dir}")


if __name__ == "__main__":
    main()
