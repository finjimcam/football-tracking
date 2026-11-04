"""Aggregate the compute-stage JSONs into a tidy stats CSV →
outputs/stats/<match_id>_stats.csv.

Columns: metric, scope, entity, value, unit, confidence — so quality flags
travel with the numbers. Missing compute artifacts are skipped with a warning
(the report can still be built from whatever exists).

Usage:
    python tools/export_stats.py --match-id <match_id>
"""
import argparse
import csv
from pathlib import Path

from _common import load_meta, match_dir, read_json


def optional(path):
    if Path(path).exists():
        return read_json(path)
    print(f"WARNING: {path} missing — its metrics will be absent from the CSV")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--output", default=None, help="Default: outputs/stats/<match_id>_stats.csv")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    mdir = match_dir(args.match_id, args.tmp_dir)
    meta = load_meta(args.match_id, args.tmp_dir)
    possession = optional(mdir / "possession.json")
    positions = optional(mdir / "positions.json")
    physical = optional(mdir / "physical.json")

    rows = []

    def add(metric, scope, entity, value, unit, confidence):
        if value is not None:
            rows.append({"metric": metric, "scope": scope, "entity": entity,
                         "value": value, "unit": unit, "confidence": confidence})

    add("duration", "match", "match", meta["duration_s"], "s", "high")
    add("fps", "match", "match", meta["fps"], "fps", "high")

    if possession:
        conf = possession["confidence"]
        add("possession_pct", "team", "team_a", possession["team_a_pct"], "percent", conf)
        add("possession_pct", "team", "team_b", possession["team_b_pct"], "percent", conf)
        add("possession_switches", "match", "match", possession["possession_switches"], "count", conf)
        add("ball_detection_rate", "match", "match",
            round(100 * possession["ball_detection_rate"], 1), "percent", "high")
        if possession.get("team_a_time_pct") is not None:  # time held, with hold through dropouts
            add("possession_time_pct", "team", "team_a", possession["team_a_time_pct"], "percent", conf)
            add("possession_time_pct", "team", "team_b", possession["team_b_time_pct"], "percent", conf)

    if positions:
        conf = "high" if positions["calibrated"] else "medium_uncalibrated"
        for team, t in positions["teams"].items():
            add("avg_position_x", "team", team, t["avg_x"], positions["unit"], conf)
            add("avg_position_y", "team", team, t["avg_y"], positions["unit"], conf)
        for team, phases in positions.get("by_phase", {}).items():  # shape in / out of possession
            for phase, m in phases.items():
                for k in ("defensive_line_m_from_own_goal", "avg_shape_length_m", "avg_shape_width_m"):
                    if m.get(k) is not None:
                        add(f"{k}_{phase}", "team", team, m[k], "m", "identified")

    if physical:
        conf = "high" if physical["calibrated"] else "medium_uncalibrated"
        for team, t in physical["teams"].items():
            add("distance_total", "team", team, t["distance_total"], physical["unit"], conf)
            add("avg_speed", "team", team, t["avg_speed"], physical["speed_unit"], conf)
            add("max_speed", "team", team, t["max_speed"], physical["speed_unit"], conf)
        for p in physical.get("players", []):
            entity = f"{p['player']} ({p['team']})"
            add("distance_total", "player", entity, p["distance"], physical["unit"], "identified")
            add("max_speed", "player", entity, p["max_speed"], physical["speed_unit"], "identified")
            add("tracking_coverage", "player", entity, round(100 * p["coverage"], 1), "percent", "identified")
        for tr in physical["tracks"]:
            if tr.get("player"):
                continue  # already covered by the identified per-player rows
            add("distance_total", "player", f"track_{tr['track_id']} ({tr['team']})",
                tr["distance"], physical["unit"], "best_effort")

    out_path = Path(args.output) if args.output else (
        Path(args.outputs_dir) / "stats" / f"{args.match_id}_stats.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "scope", "entity", "value", "unit", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} metrics to {out_path}")


if __name__ == "__main__":
    main()
