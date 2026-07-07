"""Build the PDF match report → outputs/reports/<match_id>_report.pdf.

matplotlib PdfPages only (no new deps). Pages: cover (teams, date, match_id),
possession, positions/heatmaps, physical table, and always a final
data-quality appendix (ball detection rate, track count, fragmentation
estimate, calibration status). Sections whose compute artifact is missing
render a "not computed" note instead of failing.

Team display names come from meta.json config (team_a_name/team_b_name, set
via ingest_video.py --config) or the CLI flags below; default Team A / Team B.

Usage:
    python tools/build_report.py --match-id <match_id>
    python tools/build_report.py --match-id <match_id> --team-a-name Reds --team-b-name Blues
"""
import argparse
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from _common import load_meta, match_dir, read_json

PAGE = (8.27, 11.69)  # A4 portrait, inches
TEAM_COLORS = {"team_a": "#d62728", "team_b": "#1f77b4"}


def optional(path):
    return read_json(path) if Path(path).exists() else None


def text_page(pdf, title, lines):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.92, title, fontsize=20, weight="bold")
    fig.text(0.08, 0.88, "\n".join(lines), fontsize=11, va="top", linespacing=1.8)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--team-a-name", default=None)
    parser.add_argument("--team-b-name", default=None)
    parser.add_argument("--output", default=None, help="Default: outputs/reports/<match_id>_report.pdf")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    mdir = match_dir(args.match_id, args.tmp_dir)
    meta = load_meta(args.match_id, args.tmp_dir)
    config = meta.get("config", {})
    possession = optional(mdir / "possession.json")
    positions = optional(mdir / "positions.json")
    physical = optional(mdir / "physical.json")
    team_map = optional(mdir / "team_map.json") or {}
    player_map = optional(mdir / "player_map.json")

    names = {
        "team_a": args.team_a_name or config.get("team_a_name", "Team A"),
        "team_b": args.team_b_name or config.get("team_b_name", "Team B"),
    }
    calibrated = any(a and a.get("calibrated") for a in (possession, positions, physical))

    out_path = Path(args.output) if args.output else (
        Path(args.outputs_dir) / "reports" / f"{args.match_id}_report.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        # --- Cover ---
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.62, "Match Report", ha="center", fontsize=30, weight="bold")
        fig.text(0.5, 0.54, f"{names['team_a']}  vs  {names['team_b']}", ha="center", fontsize=20)
        kickoff = config.get("kickoff", date.today().isoformat())
        fig.text(0.5, 0.47, str(kickoff), ha="center", fontsize=13, color="0.35")
        fig.text(0.5, 0.12,
                 f"match_id: {args.match_id}\n"
                 f"video: {meta['duration_s'] / 60:.1f} min @ {meta['fps']:.1f} fps, "
                 f"{meta['width']}x{meta['height']}\n"
                 f"calibration: {'yes (pitch meters)' if calibrated else 'NO — pixel units throughout'}",
                 ha="center", fontsize=9, color="0.45", linespacing=1.8)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Possession ---
        if possession and possession["team_a_pct"] is not None:
            fig = plt.figure(figsize=PAGE)
            fig.text(0.08, 0.92, "Possession", fontsize=20, weight="bold")
            ax = fig.add_axes([0.1, 0.62, 0.8, 0.12])
            a, b = possession["team_a_pct"], possession["team_b_pct"]
            ax.barh([0], [a], color=TEAM_COLORS["team_a"])
            ax.barh([0], [b], left=[a], color=TEAM_COLORS["team_b"])
            ax.text(a / 2, 0, f"{names['team_a']}\n{a}%", ha="center", va="center", color="white", weight="bold")
            ax.text(a + b / 2, 0, f"{names['team_b']}\n{b}%", ha="center", va="center", color="white", weight="bold")
            ax.set_xlim(0, 100)
            ax.axis("off")
            fig.text(0.08, 0.52,
                     f"Based on {possession['attributed_frames']} attributed of "
                     f"{possession['ball_frames']} ball-detected frames "
                     f"({possession['possession_switches']} possession switches).\n"
                     f"Ball detection rate: {possession['ball_detection_rate'] * 100:.1f}% of all frames.\n"
                     f"Confidence: {possession['confidence'].upper()}",
                     fontsize=11, va="top", linespacing=1.8)
            if possession["confidence"] == "low":
                fig.text(0.08, 0.40, "⚠ Low ball detection rate — treat possession as indicative only.",
                         fontsize=11, color="#b30000")
            pdf.savefig(fig)
            plt.close(fig)
        else:
            text_page(pdf, "Possession", ["Not computed — run tools/compute_possession.py "
                                          "(or the ball was never detected near a team player)."])

        # --- Positions / heatmaps ---
        if positions:
            figures = [(k, p) for k, p in positions["figures"].items() if Path(p).exists()]
            fig = plt.figure(figsize=PAGE)
            fig.text(0.08, 0.94, "Positions", fontsize=20, weight="bold")
            slots = [[0.06, 0.62, 0.88, 0.28], [0.06, 0.33, 0.88, 0.28], [0.06, 0.04, 0.88, 0.28]]
            order = ["avg_positions", f"heatmap_team_a", f"heatmap_team_b"]
            imgs = dict(figures)
            for slot, key in zip(slots, [k for k in order if k in imgs]):
                ax = fig.add_axes(slot)
                ax.imshow(mpimg.imread(imgs[key]))
                ax.axis("off")
            if not positions["calibrated"]:
                fig.text(0.08, 0.915, "Uncalibrated — positions are in raw pixels.", fontsize=10, color="#b30000")
            pdf.savefig(fig)
            plt.close(fig)
        else:
            text_page(pdf, "Positions", ["Not computed — run tools/compute_positions.py."])

        # --- Physical ---
        if physical:
            unit, sunit = physical["unit"], physical["speed_unit"]
            fig = plt.figure(figsize=PAGE)
            fig.text(0.08, 0.92, "Physical", fontsize=20, weight="bold")
            if not physical["calibrated"]:
                fig.text(0.08, 0.89, "Uncalibrated — distances/speeds are in pixel units.",
                         fontsize=10, color="#b30000")
            ax = fig.add_axes([0.08, 0.66, 0.84, 0.18])
            ax.axis("off")
            team_rows = [[names[t], f"{v['distance_total']:,.0f} {unit}", f"{v['avg_speed']} {sunit}",
                          f"{v['max_speed']} {sunit}", v["n_tracks"]]
                         for t, v in physical["teams"].items()]
            tbl = ax.table(cellText=team_rows,
                           colLabels=["Team", "Total distance", "Avg speed", "Max speed", "Tracks"],
                           loc="upper center", cellLoc="center")
            tbl.scale(1, 1.6)
            per_player = physical.get("players", [])
            ax2 = fig.add_axes([0.08, 0.12, 0.84, 0.46])
            ax2.axis("off")
            if per_player:
                fig.text(0.08, 0.60, "Per player (roster identities; distance during untracked "
                         "gaps is missing — see coverage):", fontsize=11)
                rows = [[f"#{p['number']}", names.get(p["team"], p["team"]),
                         f"{p['distance']:,.0f} {unit}", f"{p['duration_s']:,.0f} s",
                         f"{p['max_speed']} {sunit}", f"{p['coverage']:.0%}"]
                        for p in per_player[:18]]
                tbl2 = ax2.table(cellText=rows,
                                 colLabels=["Player", "Team", "Distance", "Tracked time",
                                            "Max speed", "Coverage"],
                                 loc="upper center", cellLoc="center")
            else:
                fig.text(0.08, 0.60, "Top tracks by distance (best-effort: one player may span several track IDs):",
                         fontsize=11)
                top = physical["tracks"][:15]
                rows = [[f"track {r['track_id']}", names.get(r["team"], r["team"]),
                         f"{r['distance']:,.0f} {unit}", f"{r['duration_s']:,.0f} s", f"{r['max_speed']} {sunit}"]
                        for r in top]
                tbl2 = ax2.table(cellText=rows,
                                 colLabels=["Track", "Team", "Distance", "Tracked time", "Max speed"],
                                 loc="upper center", cellLoc="center")
            tbl2.scale(1, 1.4)
            pdf.savefig(fig)
            plt.close(fig)
        else:
            text_page(pdf, "Physical", ["Not computed — run tools/compute_physical.py."])

        # --- Data-quality appendix (always present) ---
        n_tracks = sum(1 for v in team_map.values() if v in ("team_a", "team_b"))
        n_unknown = sum(1 for v in team_map.values() if v == "unknown")
        total_tracks = len(team_map)
        frag = f"{total_tracks / 22:.1f}× the ~22 expected on-pitch people" if total_tracks else "n/a"
        lines = [
            f"Calibration: {'applied — stats in pitch meters' if calibrated else 'NONE — stats in raw pixels, flagged'}",
            f"Player tracks: {total_tracks} total ({n_tracks} assigned to a team, {n_unknown} unknown)",
            f"Track fragmentation estimate: {frag}",
            "  (well above 1× means IDs fragmented; team totals stand, per-player numbers are best-effort)",
        ]
        if player_map:
            ignored = player_map.get("ignored_tracks", [])
            n_user = sum(1 for i in ignored if i.get("reason") == "user")
            lines.insert(2, f"Curation: {len(player_map.get('players', {}))} identified players; "
                            f"{len(ignored)} track(s) excluded from all stats "
                            f"({n_user} marked by hand, {len(ignored) - n_user} off-pitch)")
        if possession:
            rate = possession["ball_detection_rate"]
            lines.append(f"Ball detection rate: {rate * 100:.1f}%"
                         + (" — LOW (<30%), possession is low-confidence" if rate < 0.30 else ""))
        else:
            lines.append("Ball detection rate: possession not computed")
        lines += [
            "",
            "Every number in this report is reproducible from tracks.parquet + team_map.json",
            "(+ calibration) without re-running tracking. See outputs/stats/ for the tidy CSV",
            "with per-metric confidence flags.",
        ]
        text_page(pdf, "Data-quality appendix", lines)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
