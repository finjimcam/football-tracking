# Stopframe: web front end

A browser UI over the `tools/` pipeline (issue #17). It runs each stage of
[workflows/generate_match_report.md](../workflows/generate_match_report.md)
as a background job with a live log and a progress bar, shows each match's
stage status, and serves its outputs (report PDF, stats CSV, tracking video).
The visual design follows the "Match Analysis Workspace" canvas linked from
the issue.

```bash
python tools/serve_app.py                    # http://<host>:8780/
python tools/serve_app.py --busy-check ~/claude-tg/bin/media-busy   # on pphub
```

Use `.venv/bin/python` on pphub. Needs `fastapi` and `uvicorn` (in requirements.txt).

## What it does
- **Matches**: every ingested match (`.tmp/<match_id>/meta.json`) with a
  per-stage status strip, plus the videos in `videos/` with an **Ingest**
  button for the ones that haven't been ingested yet.
- **Match page**: the pipeline stages in workflow order. Each stage shows
  *to do / done / stale / blocked*, based only on which artifacts exist and
  how old they are (`app/matches.py`), with a **Run** button and an options
  form. Leave a box empty to use the tool's own default. Outputs can be
  downloaded from here.
- **Jobs**: the history of every job, with its log, exit code and timings.
  Queued or running jobs can be cancelled.
- **Lines** (broadcast matches): pick the back line and midfield line for each
  of a team's out-of-possession phases. This is the browser version of
  `tools/select_shape_players.py`. Click a player to cycle none → back →
  midfield. Right-click (or use Re-link mode) to put a lost chain back on a
  player. A strip under the frame shows each chain as green (detected), amber
  (filled) or red (lost), with white ticks at hand-offs. Picks are saved to
  the same `config/<id>_<team>_out_of_possession_shape_selection.json` the
  CLI picker uses, so either one can carry on from the other.
  **Render picked lines** queues `cut_phase_clips.py --shape-selection`.
  The chain, gap-filling and re-link logic runs on the server
  (`app/lines.py` calls the same functions), so the page shows exactly what
  the render will draw. Keys: a/d ±0.5 s, A/D ±2 s, ,/. ±1 frame, k
  play, Enter save + next, s skip, r auto guess, c clear, p previous.
- **Code / Review / Sorter / Matrix / Present** tabs are placeholders that
  link to the issues that will build them (#3, #1, #4, #11, #12).

## Two pipelines
A match whose `config/<id>_match.json` exists gets the **broadcast** pipeline
([workflows/track_identities.md](../workflows/track_identities.md)): detect,
automatic calibration, identities, ball, possession, then phase clips and the
picked-lines render. Every other match gets the **match report** pipeline.
When a video is ingested, a `config/<stem>_<sha8>_match.json` with the
matching name is passed in automatically. `detect_frames.py --device auto`
uses CUDA when it's available, so it runs on pphub's GTX 1060.

## How jobs run (`app/jobs.py`)
- Each job runs one subprocess, `python -u tools/<tool> <args>`, from the repo
  root. Args come only from `app/catalog.py`: the browser can pick which
  stage to run and fill in that stage's own options, but can't add other
  flags, and video paths have to point inside `videos/`.
- **Heavy** stages run under `nice -n 10 ionice -c3` in two lanes: GPU
  (tracking, detection, rendering) and CPU (automatic calibration,
  identities, verification, phase clips). Each lane runs one job at a time,
  so the CPU lane can run alongside the GPU one. If you start the app with `--busy-check`,
  they also wait while that command exits 0 (for example, while
  Jellyfin/Kodi is playing). Use **Run now** to start one anyway. Light
  stages run two at a time.
- Records and logs go in `.tmp/jobs/` (disposable). After a restart, jobs
  that were running show as `interrupted`.
- **Progress**: a tool prints `PROGRESS <done>/<total> <label>` lines through
  `_common.Progress`. These lines drive the progress bar and are left out of
  the log. `run_tracking.py`, `estimate_camera_motion.py` and
  `render_tracking_video.py` print them. Tools that don't show a moving bar
  with no percentage.

## Not in the app yet
- `calibrate_pitch.py` and `label_roster.py` are interactive, so they still
  run by hand. The match page shows their status and links to the workflow.
  The `web-calibration` branch has a browser calibration UI that could be
  folded in later.
- No auth: it's meant to sit behind Tailscale. Users are issue #21, and a
  persistent job queue is issue #19.

Ports: 8765/8766 are used by Jellyfin live TV, and 8790 by
`calibrate_pitch.py`'s web mode, so this app defaults to 8780.

Tests: `.venv/bin/python -m pytest tests/test_app.py`
