# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Agent Instructions

You're working inside the **WAT framework** (Workflows, Agents, Tools). This architecture separates concerns so that probabilistic AI handles reasoning while deterministic code handles execution. That separation is what makes this system reliable.

## The WAT Architecture

**Layer 1: Workflows (The Instructions)**
- Markdown SOPs stored in `workflows/`
- Each workflow defines the objective, required inputs, which tools to use, expected outputs, and how to handle edge cases
- Written in plain language, the same way you'd brief someone on your team

**Layer 2: Agents (The Decision-Maker)**
- This is your role. You're responsible for intelligent coordination.
- Read the relevant workflow, run tools in the correct sequence, handle failures gracefully, and ask clarifying questions when needed
- You connect intent to execution without trying to do everything yourself
- Example: If you need to pull data from a website, don't attempt it directly. Read `workflows/scrape_website.md`, figure out the required inputs, then execute `tools/scrape_single_site.py`

**Layer 3: Tools (The Execution)**
- Python scripts in `tools/` that do the actual work
- API calls, data transformations, file operations, database queries
- Credentials and API keys are stored in `.env`
- These scripts are consistent, testable, and fast

**Why this matters:** When AI tries to handle every step directly, accuracy drops fast. If each step is 90% accurate, you're down to 59% success after just five steps. By offloading execution to deterministic scripts, you stay focused on orchestration and decision-making where you excel.

## How to Operate

**1. Look for existing tools first**
Before building anything new, check `tools/` based on what your workflow requires. Only create new scripts when nothing exists for that task.

**2. Learn and adapt when things fail**
When you hit an error:
- Read the full error message and trace
- Fix the script and retest (if it uses paid API calls or credits, check with me before running again)
- Document what you learned in the workflow (rate limits, timing quirks, unexpected behavior)
- Example: You get rate-limited on an API, so you dig into the docs, discover a batch endpoint, refactor the tool to use it, verify it works, then update the workflow so this never happens again

**3. Keep workflows current**
Workflows should evolve as you learn. When you find better methods, discover constraints, or encounter recurring issues, update the workflow. That said, don't create or overwrite workflows without asking unless I explicitly tell you to. These are your instructions and need to be preserved and refined, not tossed after one use.

## The Self-Improvement Loop

Every failure is a chance to make the system stronger:
1. Identify what broke
2. Fix the tool
3. Verify the fix works
4. Update the workflow with the new approach
5. Move on with a more robust system

This loop is how the framework improves over time.

## File Structure

**What goes where:**
- **Deliverables**: Final outputs go to cloud services (Google Sheets, Slides, etc.) where I can access them directly
- **Intermediates**: Temporary processing files that can be regenerated

**Directory layout:**
```
.tmp/           # Temporary files (scraped data, intermediate exports). Regenerated as needed.
tools/          # Python scripts for deterministic execution
workflows/      # Markdown SOPs defining what to do and how
.env            # API keys and environment variables (NEVER store secrets anywhere else)
credentials.json, token.json  # Google OAuth (gitignored)
```

**Core principle:** Local files are just for processing. Anything I need to see or use lives in cloud services. Everything in `.tmp/` is disposable.

## Bottom Line

You sit between what I want (workflows) and what actually gets done (tools). Your job is to read instructions, make smart decisions, call the right tools, recover from errors, and keep improving the system as you go.

Stay pragmatic. Stay reliable. Keep learning.

## Current Repository State

The WAT scaffolding described above now exists.
[Fine_tune_yolo_for_football_analytics.ipynb](Fine_tune_yolo_for_football_analytics.ipynb)
is the original Colab-style notebook (Labellerr's Hands-On Computer Vision
cookbook) that the `tools/`/`workflows/` below were decomposed from — it's
still useful as a reference for the underlying library calls, but the
`workflows/` files are the source of truth for how to actually run the
pipeline now (CLI args + `.env` instead of notebook widgets/`input()` prompts).

**Pipeline stages, in order** — each maps to one workflow SOP and one or two
tool scripts:

| Stage | Workflow | Tool(s) |
|---|---|---|
| Upload + annotate a dataset on Labellerr, export as COCO | [workflows/annotate_and_export_dataset.md](workflows/annotate_and_export_dataset.md) | `tools/labellerr_annotate.py` |
| Convert COCO → YOLO dataset layout | [workflows/prepare_training_data.md](workflows/prepare_training_data.md) | `tools/coco_to_yolo.py` |
| Fine-tune YOLO on the dataset | [workflows/train_yolo_model.md](workflows/train_yolo_model.md) | `tools/train_yolo.py` |
| Track players + draw trajectories on video | [workflows/track_and_visualize.md](workflows/track_and_visualize.md) | `tools/track_players.py` |
| Generate a positional heatmap | [workflows/generate_heatmap.md](workflows/generate_heatmap.md) | `tools/generate_heatmap.py` |
| Full match report from a match video (Veo-style clips, manual roster) | [workflows/generate_match_report.md](workflows/generate_match_report.md) + [workflows/curate_and_identify_players.md](workflows/curate_and_identify_players.md) | `tools/ingest_video.py` → … → `tools/build_report.py` |
| **Broadcast/tactical-cam half: persistent identities for all 22, verification vs Opta, phase clips** | [workflows/track_identities.md](workflows/track_identities.md) | `tools/detect_frames.py`, `tools/calibrate_auto.py`, `tools/track_identities.py`, `tools/track_ball.py`, `tools/compute_possession.py`, `tools/verify_tracking.py`, `tools/compare_reference_stats.py`, `tools/render_tracking_video.py`, `tools/cut_phase_clips.py` |

**Web front end (Stopframe):** `tools/serve_app.py` serves a browser UI
(`app/`) that runs the match-report stages as background jobs with logs and
progress. See [app/README.md](app/README.md). Long loops in tools print
`PROGRESS done/total` lines via `_common.Progress` for its progress bars.

**Notes that apply across all of them:**
- `tools/coco_to_yolo.py` depends on the external `yolo_finetune_utils` repo
  (`git clone https://github.com/Labellerr/yolo_finetune_utils.git`), which
  is not vendored in this repo.
- Class IDs referenced in `--team-colors`/`--classes` (the notebook's
  examples use `0` and `4`) are placeholders — they must match whatever
  classes your specific Labellerr annotation template used.
- `.env` holds `LABELLERR_API_KEY`/`LABELLERR_API_SECRET`/`LABELLERR_CLIENT_ID`
  and a default `TEAM_COLORS`; it's gitignored along with `.tmp/`, `runs/`,
  `yolo_format/`, and `*.pt` weights, all of which are regenerable
  intermediates per the File Structure convention above.
- Hand-verified per-match inputs live in `config/`: `<id>_match.json`
  (teams, kits, clock sync, analysis window), `<id>_reference.json`
  (published Opta stats/lineups/events with sources), `<id>_calibration.json`.
  `_common.match_config()` reads the live `_match.json` over meta.json's copy.
- Shared helpers: `tools/_common.py` (paths, calibration, H.264 `VideoWriter`,
  clock labels) and `tools/_pitch.py` (pitch-line model, PTZ camera geometry).
- The disk on this machine runs near-full: check free space before long runs;
  renders use H.264 (~5 MB/min) via PyAV.
- `requirements.txt` lists the dependencies these tools need
  (`ultralytics`, `opencv-python`, the Labellerr SDK, etc.); a GPU-backed
  environment is required for training and for tracking/heatmap generation
  at any real speed.
