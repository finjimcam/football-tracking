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
  player. **Fix swap** (`x`) is for chains that jumped to the wrong player
  when two players collided: tap the chain, then the right player. The
  picker finds the swap moment (the two players' closest approach in the
  last 6 s) and cuts the chain there. If the other player was also on a
  picked chain, that chain swaps back too. Cut anchors are saved as
  `[frame, track_id, 1]` ("from this frame on"), which the renderer and the
  CLI picker understand. A strip under the frame shows each chain as green (detected), amber
  (filled) or red (lost), with white ticks at hand-offs. Picks are saved to
  the same `config/<id>_<team>_out_of_possession_shape_selection.json` the
  CLI picker uses, so either one can carry on from the other.
  **Render picked lines** queues `cut_phase_clips.py --shape-selection`.
  **Clip export** (issue #37): tick the phases you want in the list (or use
  Keep all / Drop all) and trim each one with *Start here* `[` and *End
  here* `]`. `e` toggles the current phase, and *Full phase* undoes a trim.
  The strip greys out what's trimmed off. Choices save as you go to
  `config/<id>_<team>_out_of_possession_export.json`. **Export clips (zip)**
  queues `cut_phase_clips.py --shape-selection --export-selection --separate`:
  one MP4 per kept phase, with your lines, zipped for PowerPoint. Download it
  from the match page or from *Download last zip*.
  **Identify** (`i`, issue #16): tell the tracker who players are. In
  Identify mode every box shows its tracked shirt number: white is
  role-inferred, gold is set by you, and a red **?** means unidentified.
  Click a player, or drag a box round one the tracker missed, then pick a
  name from the lineup (`config/<id>_reference.json`). You can also do it while
  picking lines: Shift-click (or long-press on a phone) a player to open the same
  name picker without leaving Pick mode. In Pick mode each B/M marker shows the
  shirt number of the player it's on (gold if you set it), and unpicked
  players show a small number (red **?** if unidentified). Each pick is an
  *anchor*, saved to `config/<id>_identity_anchors.json` (frame plus the
  detection's box). Until you apply it, the box shows your pick at once as
  **27•** for the stretch of track it was made on, and any other box the
  last run called 27 shows **27?**. **Apply to tracking** queues *Track identities* with
  cached tracklets (about 3 min). That run names each identity chain after
  the player its anchors point at, and moves anchored fragments onto the
  right player (`tools/identity_anchors.py`). Re-run *Track ball*, *Compute
  possession* and the stats stages afterwards, so stats and paths for the
  whole match use the corrected identities.
  The chain, gap-filling and re-link logic runs on the server
  (`app/lines.py` calls the same functions), so the page shows exactly what
  the render will draw. Keys: a/d ±0.5 s, A/D ±2 s, ,/. ±1 frame, k
  play, Enter save + next, s skip, r auto guess, c clear, p previous.
- **Players** (issue #16, also *Players* on a broadcast match's page): every
  player's whole-match path, heatmap and running stats, computed from
  `tracks.parquet` by `tools/player_paths.py` (`app/players.py` wraps it).
  Pick a player on the left. The pitch shows where they were (heatmap:
  seconds per 2.5 m cell, square-root scale), their path at 2 Hz, the team's
  average positions with them highlighted, and which way the team attacks.
  The sliders and presets (0–15′, 15–30′, …) limit everything to part of
  the match. Stats: distance, minutes tracked and coverage, per-90
  distance, average and top speed (99th percentile), sprints (25 km/h or
  faster held for at least 1 s) and distance per speed band. Gaps in the tracking
  are left out, so distance is a lower bound. Names come from
  `player_map.json`: a gold dot means anchors you set in Identify mode, and
  the note under the stats says how many of a chain's joins were
  uncertain. A wide heatmap with many uncertain joins usually means the
  chain picks up a team-mate, so add anchors in the Lines tab and press
  *Apply to tracking*. The tab reloads the new identities by itself.
- **Library** (issue #16): teams, players, competitions, seasons and
  squads, plus which match video belongs to which season and teams. It all
  lives in one hand-editable file, `config/library.json`
  (`tools/library.py`, `app/library.py`). *Matches* lists every ingested
  video. Open one to set its season, round, date, the two teams and home
  side, and to see the video and data files linked to it. *Import from
  match config* fills all of that, plus both lineups, from
  `config/<id>_match.json` and `_reference.json`. Running it again only
  fills gaps and never overwrites your edits. *Competitions & seasons*
  adds and renames them. Each team page edits name, short name, colour and
  kit, and the **squad** for a chosen season: shirt numbers belong to the
  squad, not to the player, so a new number or a transfer next season
  keeps history straight. A typed name that's already in the library links
  to that player. Every change is checked before it's saved: no shirt
  number twice in a squad, no deleting a team or season that's still
  used. The Lines tab's Identify picker lists the match lineup, then the
  rest of the squad as subs.
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
