"""Compare our first-half numbers with the published match data ->
outputs/stats/<id>_vs_reference.csv (+ printed table).

Stage 8 of workflows/track_identities.md. Reads possession.json,
possession_timeline.parquet, physical.json, verification.json (run
compute_possession / compute_physical / verify_tracking first) and
config/<id>_reference.json (Opta team stats, lineups, events - with
sources).

Every row says what the reference covers, because the published data is
mostly FULL-MATCH while our window is the first half:
  - possession %          : ours (time held, first half) vs Opta full match
  - field tilt            : share of located-ball time in each team's
                            attacking third vs Opta's final-third passes and
                            entries ratios (full match) - dominance measured
                            two independent ways
  - distance covered      : observed metres + coverage-scaled to the full
                            half, vs the typical per-half range (no
                            match-specific figure is published)
  - Opta event checks     : possession at the timestamped first-half events,
                            named player nearest the ball (verification.json)
  - shape                 : our average depth ordering vs the official
                            formation lines

Usage:
    python tools/compare_reference_stats.py --match-id <id>
"""
import argparse

import numpy as np
import pandas as pd

from _common import (OUTPUTS_DIR, load_player_map, match_config, match_dir, read_json, reference_path)


def pct(a, b):
    return round(100 * a / (a + b), 1) if (a + b) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--tmp-dir", default=".tmp")
    args = parser.parse_args()

    mdir = match_dir(args.match_id, args.tmp_dir)
    cfg = match_config(args.match_id, args.tmp_dir, args.config_dir)
    ref = read_json(reference_path(args.match_id, args.config_dir))
    poss = read_json(mdir / "possession.json")
    phys = read_json(mdir / "physical.json")
    ver = read_json(mdir / "verification.json")
    pm = load_player_map(args.match_id, args.tmp_dir)
    tl = pd.read_parquet(mdir / "possession_timeline.parquet")
    A, B = cfg.get("team_a_name", "team_a"), cfg.get("team_b_name", "team_b")
    full = ref["full_match_team_stats"]
    half_s = ref["first_half"]["duration_s"]
    window_s = (pm["window"]["end_frame"] - pm["window"]["start_frame"] + 1) / 25.0
    rows = []

    def add(metric, ours, reference, scope, note=""):
        rows.append({"metric": metric, "ours": ours, "reference": reference, "reference_scope": scope, "note": note})

    # possession
    add(f"possession % {A}", poss.get("team_a_time_pct"), full["possession_pct"]["team_a"], "Opta, full match",
        "ours: share of first-half time each team held the ball (8 s hold through dropouts)")
    add(f"possession % {B}", poss.get("team_b_time_pct"), full["possession_pct"]["team_b"], "Opta, full match")
    add(f"ball-near-player % {A}", poss.get("team_a_pct"), full["possession_pct"]["team_a"], "Opta, full match",
        "ours: frames where the ball is within 2.5 m of a player, by that player's team")

    # field tilt: which attacking third the ball is in. Direction from the keepers.
    gk = {k: v for k, v in pm["players"].items() if v.get("role") == "GK"}
    tracks = pd.read_parquet(mdir / "tracks.parquet", columns=["frame", "track_id", "class", "pitch_x"])
    gk_a = tracks[(tracks["class"] == "player") & tracks["track_id"].isin(
        [t for v in gk.values() if v["team"] == "team_a" for t in v["track_ids"]])]["pitch_x"].median()
    a_attacks_right = gk_a < 52.5
    bx = tl.loc[tl["ball"], "ball_x"].dropna()
    a_third = (bx > 70) if a_attacks_right else (bx < 35)
    b_third = (bx < 35) if a_attacks_right else (bx > 70)
    ftp = full["total_final_third_passes"]
    add(f"field tilt % {A} vs final-third passes", pct(a_third.sum(), b_third.sum()),
        pct(ftp["team_a"], ftp["team_b"]), f"Opta final-third passes {ftp['team_a']} v {ftp['team_b']}, full match",
        "ours: share of located-ball time in A's attacking third vs B's")
    add(f"field tilt % {A} vs final-third entries", pct(a_third.sum(), b_third.sum()),
        pct(full["final_third_entries"]["team_a"], full["final_third_entries"]["team_b"]),
        "Opta final-third entries 105 v 23, full match")

    # distance
    lo, hi = ref["physical_reference"]["typical_pl_team_km_per_match"]
    for team, name in (("team_a", A), ("team_b", B)):
        players = [p for p in phys.get("players", []) if p["team"] == team]
        observed = sum(p["distance"] for p in players) / 1000
        scaled = sum(p["distance"] / max(p["duration_s"], 1) * half_s for p in players) / 1000
        add(f"distance km {name} (observed)", round(observed, 1), f"{lo / 2:.0f}-{hi / 2:.0f}",
            "typical PL team per half (context, not match data)",
            f"tracked time only; mean coverage {np.mean([p['coverage'] for p in players]):.0%}")
        add(f"distance km {name} (scaled to full half)", round(scaled, 1), f"{lo / 2:.0f}-{hi / 2:.0f}",
            "typical PL team per half (context, not match data)",
            "each player's tracked m/s x 2824 s; on-camera players skew towards the ball, so this leans high")
    if "team_b_km_per_match_conte_era" in ref["physical_reference"]:
        add(f"distance km {B} reference", None,
            round(ref["physical_reference"]["team_b_km_per_match_conte_era"] / 2, 1),
            "Spurs' Conte-era average per match / 2 (The Analyst)")

    # events
    add("Opta events: possession agrees", ver["events_possession_agree"], "all", "23 timestamped 1st-half events",
        "shots/goals/offsides -> attacking team on the ball just before; fouls won -> fouled team just after")
    add("Opta events: named player nearest ball", ver["events_named_player_nearest_ball"], "all",
        "events naming a player", "checks identity persistence AND the role-inferred name mapping together")
    add("Opta events: named player near ball at all", ver["events_named_player_near_ball_at_all"], "all",
        "events naming a player")

    # shape: does our depth ordering reproduce the official lines?
    for team, name in (("team_a", A), ("team_b", B)):
        lines = ref["lineups"][team]["lines"]
        add(f"formation {name}", "-".join(str(len(l)) for l in lines[1:]), ref["lineups"][team]["formation"],
            "official lineup", "names are mapped onto these lines by average depth/width, so this row is "
                               "an input, not a test - see the average-position figure for the real shape")

    df = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "stats" / f"{args.match_id}_vs_reference.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(df[["metric", "ours", "reference", "reference_scope"]].to_string(index=False))
    print(f"Wrote {out} ({window_s / 60:.1f} min tracked window)")


if __name__ == "__main__":
    main()
