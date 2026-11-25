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

import pandas as pd

from _common import load_meta, match_config, match_dir, read_json

PAGE = (8.27, 11.69)  # A4 portrait, inches
TEAM_COLORS = {"team_a": "#d62728", "team_b": "#1f77b4"}


def table_page(pdf, title, subtitle, header, rows, col_widths=None, font=8):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.94, title, fontsize=20, weight="bold")
    fig.text(0.08, 0.915, subtitle, fontsize=9, color="0.35", va="top", linespacing=1.5)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.82])
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="left", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font)
    tbl.scale(1, 1.35)
    pdf.savefig(fig)
    plt.close(fig)


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
    config = match_config(args.match_id, args.tmp_dir)
    colors = {**TEAM_COLORS, **config.get("colors", {})}
    verification = optional(mdir / "verification.json")
    ref_csv = Path(args.outputs_dir) / "stats" / f"{args.match_id}_vs_reference.csv"
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
            timed = possession.get("team_a_time_pct") is not None
            a, b = ((possession["team_a_time_pct"], possession["team_b_time_pct"]) if timed
                    else (possession["team_a_pct"], possession["team_b_pct"]))
            ax.barh([0], [a], color=colors["team_a"])
            ax.barh([0], [b], left=[a], color=colors["team_b"])
            ax.text(a / 2, 0, f"{names['team_a']}\n{a}%", ha="center", va="center", color="white", weight="bold")
            ax.text(a + b / 2, 0, f"{names['team_b']}\n{b}%", ha="center", va="center", color="white", weight="bold")
            ax.set_xlim(0, 100)
            ax.axis("off")
            fig.text(0.08, 0.52,
                     (f"Share of time each team held the ball (possession carried through ball dropouts for up to "
                      f"{possession.get('max_hold_s', 8):g} s).\n"
                      f"Ball-next-to-a-player frames only: {names['team_a']} {possession['team_a_pct']}% - "
                      f"{names['team_b']} {possession['team_b_pct']}%.\n" if timed else "")
                     + f"Based on {possession['attributed_frames']} attributed of "
                     f"{possession['ball_frames']} ball-located frames "
                     f"({possession['possession_switches']} possession switches).\n"
                     f"Ball located in {possession['ball_detection_rate'] * 100:.1f}% of frames.\n"
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

        # --- Shape in / out of possession (identity pipeline) ---
        phase_figs = [p for k, p in (positions or {}).get("figures", {}).items()
                      if k.endswith("_by_phase") and Path(p).exists()]
        if phase_figs:
            fig = plt.figure(figsize=PAGE)
            fig.text(0.08, 0.94, "Shape in and out of possession", fontsize=20, weight="bold")
            fig.text(0.08, 0.915, "Average position of each player while his team had / didn't have the ball. "
                                  "Names are role-inferred from the official lineup.", fontsize=9, color="0.35")
            for i, p in enumerate(sorted(phase_figs)):
                ax = fig.add_axes([0.03, 0.47 - i * 0.43, 0.94, 0.42])
                ax.imshow(mpimg.imread(p))
                ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

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
                pinfo = (player_map or {}).get("players", {})
                robust = any(p.get("top_speed_p99") is not None for p in per_player)
                rows = [[f"#{p['number']} {pinfo.get(p['player'], {}).get('name') or ''}".strip(),
                         names.get(p["team"], p["team"]),
                         f"{p['distance']:,.0f} {unit}", f"{p['duration_s']:,.0f} s",
                         f"{p['top_speed_p99'] if robust else p['max_speed']} {sunit}", f"{p['coverage']:.0%}"]
                        for p in per_player[:22]]
                tbl2 = ax2.table(cellText=rows,
                                 colLabels=["Player", "Team", "Distance", "Tracked time",
                                            "Top speed (p99)" if robust else "Max speed", "Coverage"],
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

        # --- Identity verification (tools/verify_tracking.py) ---
        if verification:
            v = verification
            stab = ", ".join(f"{names.get(t, t)} {s:.0%}" for t, s in (v.get("link_stability") or {}).items()
                             if s is not None)
            sep = v.get("role_separability", {})
            sep_txt = ", ".join(f"{names.get(t, t)} {s['separability']}" for t, s in sep.items())
            sub = (f"Window: {v['window']['minutes']} min. Identified detections "
                   f"{v['identified_detection_fraction']:.0%}; mean outfield coverage "
                   f"{v['mean_outfield_coverage']:.0%}.\n"
                   f"Links stable under all perturbations: {stab}. "
                   f"Hand-offs implying >8 m/s: {v['implausible_links']}/{v['total_links']}.\n"
                   f"Opta events - possession agrees {v['events_possession_agree']}; named player is our player "
                   f"on the ball {v['events_named_player_nearest_ball']} (team's 2 nearest "
                   f"{v['events_named_player_near_ball_at_all']}). Role separability: {sep_txt} (<1 = mixing).\n"
                   "Reading: identities hold over seconds-to-a-minute and stay within their team, but over the\n"
                   "half they swap among rotating players (attackers, midfield) - treat per-player numbers and\n"
                   "names as indicative; team-level numbers do not depend on identities.")

            def pct_or_blank(x):
                return "" if x is None or pd.isna(x) else f"{x:.0%}"

            rows = [[p["player"], p.get("name") or "", f"{p['coverage']:.0%}", p["tracklets"], p["fragile_links"],
                     p["implausible_links"], p["longest_unseen_s"], pct_or_blank(p.get("kit_consistency"))]
                    for p in v["per_player"]]
            table_page(pdf, "Identity verification", sub,
                       ["Key", "Name (role-inferred)", "Coverage", "Tracklets", "Fragile links",
                        "Implausible", "Longest unseen (s)", "Kit consistency"], rows)
            ev = [[e["clock"], e["type"], names.get(e["team"], e["team"]), e.get("player") or "",
                   "" if e.get("possession_ok") is None else ("yes" if e["possession_ok"] else "NO"),
                   e.get("nearest_to_ball") or "",
                   "" if e.get("named_player_hit") is None else ("yes" if e["named_player_hit"] else "no")]
                  for e in v["events"]]
            table_page(pdf, "Opta event checks",
                       "Each published first-half event, mapped to video frames by the scoreboard clock sync.",
                       ["Clock", "Event", "Team", "Opta player", "Possession agrees", "Our player nearest ball",
                        "Named player hit"], ev)

        # --- Comparison with published data (tools/compare_reference_stats.py) ---
        if ref_csv.exists():
            cmp = pd.read_csv(ref_csv).fillna("")
            rows = [[r.metric, str(r.ours), str(r.reference), r.reference_scope] for r in cmp.itertuples()]
            table_page(pdf, "Comparison with published data",
                       "Published data is mostly FULL-MATCH; our window is the first half. "
                       "Each row states what the reference covers.",
                       ["Metric", "Ours", "Reference", "Reference scope"], rows,
                       col_widths=[0.38, 0.1, 0.1, 0.42], font=7)

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
        if player_map and "track_identities" in player_map.get("method", ""):
            ignored = player_map.get("ignored_tracks", [])
            stab = ", ".join(f"{names.get(t, t)} {s:.0%}" for t, s in (player_map.get("link_stability") or {}).items()
                             if s is not None)
            lines[1:4] = [
                f"Identities: {len(player_map['players'])} persistent players",
                f"  ({player_map['method'].split('(')[-1].rstrip(')')})",
                f"  {player_map['identified_detection_fraction']:.0%} of on-pitch person detections identified;",
                f"  {len(player_map.get('unidentified_tracks', []))} unlinked fragments; "
                f"{len(ignored)} officials/off-pitch tracks excluded from all stats",
                f"  links stable under perturbation: {stab}",
            ]
        elif player_map:
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
