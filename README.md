# Football tracking

A computer-vision pipeline for football match video. From a match recording it
produces pitch-calibrated player and ball tracking, possession, physical and
positional stats, a PDF report, annotated video, and tactical clip
compilations. It also runs a verification pass that checks the tracking
against itself and against published match data.

The repo follows a **WAT** layout: `workflows/` holds step-by-step SOPs,
`tools/` holds the scripts that do the work, and `CLAUDE.md` holds the
instructions for the AI agent that orchestrates them.

## Which workflow to use

| Footage / goal | Workflow | Status |
|---|---|---|
| A broadcast or tactical-camera half: persistent identities for all 22 players, verification against Opta, phase clips (e.g. one team's out-of-possession play) | [workflows/track_identities.md](workflows/track_identities.md) | Run on Man City 2-3 Tottenham, first half (Nov 2026) |
| A short clip from a fixed or panning camera (e.g. Veo): match report with a manually labelled roster | [workflows/generate_match_report.md](workflows/generate_match_report.md) + [workflows/curate_and_identify_players.md](workflows/curate_and_identify_players.md) | Verified on a short Veo clip |
| Fine-tune YOLO on your own annotations (Labellerr → COCO → YOLO) | [annotate_and_export_dataset.md](workflows/annotate_and_export_dataset.md) → [prepare_training_data.md](workflows/prepare_training_data.md) → [train_yolo_model.md](workflows/train_yolo_model.md) | From the original notebook |
| A quick tracking video or heatmap from a trained model | [track_and_visualize.md](workflows/track_and_visualize.md), [generate_heatmap.md](workflows/generate_heatmap.md) | From the original notebook |

## Quick start (broadcast pipeline)

```
pip install -r requirements.txt          # includes openvino for Intel iGPU detection
```

Each match needs three hand-checked files in `config/` (the Man City v Spurs
ones are examples):
- `<id>_match.json`: teams, kit colours, scoreboard clock sync, analysis window
- `<id>_reference.json`: published stats, lineups and timestamped events, with sources
- `<id>_calibration.json`: landmark clicks on one frame (seeds the automatic calibration)

```
python tools/ingest_video.py --video "<match>.mp4" --config config/<id>_match.json
python tools/detect_frames.py --match-id <id>           # the one slow step (~55 min per half on an iGPU)
python tools/calibrate_auto.py --match-id <id>          # can run alongside detection
python tools/track_identities.py --match-id <id>
python tools/track_ball.py --match-id <id>
python tools/compute_possession.py --match-id <id> --include-unidentified
python tools/compute_physical.py --match-id <id> --smooth-s 1.0
python tools/compute_positions.py --match-id <id>
python tools/verify_tracking.py --match-id <id>
python tools/compare_reference_stats.py --match-id <id>
python tools/export_stats.py --match-id <id>
python tools/build_report.py --match-id <id>
python tools/cut_phase_clips.py --match-id <id> --team team_a [--shape]
python tools/render_tracking_video.py --match-id <id> --minimap --possession
```

Detection is cached, so every later stage re-runs in minutes from
`.tmp/<id>/`. That's how you put the footage back through the tracking after
changing a setting. The workflow explains each stage and how to read the
verification.

## Worked example: Man City 2-3 Tottenham (19 Feb 2022), first half

This is the Premier League tactical-camera feed, 46.9 minutes tracked, 11 v 11
throughout (the first substitution came at 68'). Committed outputs:
[report PDF](outputs/reports/Man_City_vs_Spurs_3725_1000MB_9b38498c_report.pdf),
[stats](outputs/stats/Man_City_vs_Spurs_3725_1000MB_9b38498c_stats.csv),
[comparison with published data](outputs/stats/Man_City_vs_Spurs_3725_1000MB_9b38498c_vs_reference.csv),
[City out-of-possession phase index](outputs/stats/Man_City_vs_Spurs_3725_1000MB_9b38498c_team_a_out_of_possession_phases.csv),
[verification CSVs](outputs/verification/Man_City_vs_Spurs_3725_1000MB_9b38498c/).

| | This pipeline (first half) | Published |
|---|---|---|
| Possession, City / Spurs | 60.7% / 39.3% | 71.2% / 28.8% (Opta, **full match**; no first-half split published) |
| City's share of ball time in the attacking thirds | 84.1% | 82.0% (share of final-third entries) |
| Possession agrees with Opta's timestamped first-half events | 18 of 23 | — |
| Distance while tracked, City / Spurs | 51.2 km / 54.0 km | no match figure published (typical: 52-58 km per team per half) |
| City out-of-possession phases | 55 phases, 17.8 min | — |
| Pitch calibration | 70,284 / 70,284 frames locked to the pitch lines | — |

## Known limitations

- **Long-term player identity.** Identities hold over seconds to about a
  minute and never switch teams. Over the half, though, they swap between
  players who rotate (City's front five, Spurs' midfield). At ~30 px tall,
  position is the only thing that tells rotating teammates apart. Player
  names are therefore inferred from the official lineup by average position
  and are indicative only, as are per-player numbers. Team-level numbers
  don't depend on identities. The verification measures all of this; see the
  workflow's results section.
- **Ball at goals.** The ball in the net or goalmouth often goes undetected,
  and a false ball elsewhere wins. Check goal moments in the tracking video.
- **Distance.** A ball-following camera over-samples players who are running
  hard, so distance scaled to the full half comes out ~10-15% high.
- **Possession.** Turnovers shorter than ~0.4 s are absorbed into longer
  phases.
- **Pitch size.** Assumed to be 105 × 68 m unless the calibration says
  otherwise.

## Data policy

This repo is public. Match video, and anything cut from it (rendered videos,
frame crops, contact sheets), is `.gitignore`d and stays local. Committed
outputs contain stats, CSVs, charts and tables only. Credentials live in
`.env` (gitignored).

## Hardware notes

Developed on a CPU-only laptop (Intel Core Ultra 5 135U, 16 GB RAM). Detection
runs on the Intel iGPU through OpenVINO (~36 ms per 720p frame, versus ~115 ms
on the CPU). Plan for ~1 GB of free disk per half, including renders, and run
memory-heavy stages one at a time.

## Repository layout

```
tools/        scripts (one stage each; shared helpers in _common.py and _pitch.py)
workflows/    SOPs: inputs, steps, outputs, edge cases and lessons learned
config/       hand-checked per-match inputs (match, reference, calibration, roster)
outputs/      reports/, stats/, verification/ (committed); videos/ (local only)
.tmp/         disposable intermediates, regenerable from the video
CLAUDE.md     instructions for the agent that runs the workflows
```
