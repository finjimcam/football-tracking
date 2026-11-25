# Workflow: Generate a Match Report

> **Status: implemented (2026-07-06).** All `tools/` scripts below exist and
> were verified end-to-end on a synthetic match (calibrated + uncalibrated
> paths). Shared path/calibration helpers live in `tools/_common.py` — every
> stage imports it, so artifact locations are defined in one place.
> **Moving-camera support added 2026-07-07** (keyframe calibration +
> `tools/estimate_camera_motion.py`), verified against synthetic ground truth
> (max error < 3 cm over a 150-frame pan/zoom) and on real panning Veo
> footage (`testvideo.mp4`, 303 frames, zero flow failures).
> **Roster identities added 2026-07-07** (`label_roster.py` +
> `assign_identities.py` + `render_tracking_video.py`): instead of treating
> every track_id as a player (40+ IDs per short clip), the user labels the
> real players (team + shirt number, max 11 per team) on anchor frames and
> identities are propagated across track fragments — enabling true
> per-player stats and an annotated tracking video.

## Objective
Turn one flattened match video into a PDF match report and a stats CSV,
delivered to `outputs/` locally. Every number in the report must be
reproducible from `tracks.parquet` (+ `team_map.json` + calibration): the
`compute_*` tools read only those artifacts, never the video, so stats can be
recomputed without re-running tracking.

Cloud delivery (Google Drive) is deferred for now — the claude.ai Drive
connector isn't authorized and no local OAuth uploader has been built. See
Step 10.

## Required Inputs
- Match video (`.mp4`), provided by the user — confirm it opens and plays and
  is flattened (not raw 360) before starting. Stop and flag if unreadable.
- Match config (optional): team names, kickoff time, camera position. If team
  names are missing, ask the user for them before building the report.
- Camera assumption: either a **fixed camera** or a camera that
  **pans/tilts/zooms from a fixed position** (tripod, mast, Veo-style unit —
  inter-frame motion of the static scene is then an exact homography). Only
  part of the pitch needs to be visible at any time, at any angle, as long
  as 4+ known landmarks can be clicked on each calibration keyframe. A
  camera that physically travels (drone flyover, handheld walking)
  introduces parallax the model can't fully capture — flag and set
  expectations before proceeding.

## Pipeline Artifacts
Intermediates live in `.tmp/<match_id>/` (disposable); deliverables in
`outputs/`; calibration in `config/` (kept — it's reusable per camera setup).

- `match_id` = video file stem + first 8 chars of the file's SHA-1, so the
  same file always maps to the same ID and different exports never collide.
- `meta.json` — fps, duration, resolution, frame count, match_id, config.
- `tracks.parquet` — the source of truth. One row per detection per frame:
  `frame, t_s, track_id, class (player|ball), x1, y1, x2, y2, foot_x, foot_y, conf`.
  `foot_x/foot_y` is the bottom-center of the box (the point that gets
  projected to pitch coordinates). **Coordinates are raw pixels** —
  calibration is applied downstream at compute time, never baked in here.
  This is what keeps tracking camera-agnostic for a future moving-camera mode.
- `team_map.json` — `track_id → team_a | team_b | unknown`.
- `config/<match_id>_roster.json` — the user's manual player labels from
  `label_roster.py`: per anchor frame, `(image x/y, team, shirt number)`.
  Lives in `config/` (like calibration clicks, it's precious manual work);
  clicks are stored as positions, not track_ids, so re-running
  `stitch_tracks.py` never invalidates a roster.
- `player_map.json` — `track_id → player` (e.g. `a7`) from
  `assign_identities.py`, plus per-player coverage and the list of
  unidentified tracks. Disposable/regenerable, keyed by current track_ids —
  re-run `assign_identities.py` after any re-stitch.
- `config/<match_id>_calibration.json` — pitch dimensions plus one of three
  `mode`s:
  - `"static"` — one homography for the whole video (fixed camera).
  - `"keyframes"` — manual keyframes only (from `calibrate_pitch.py
    --frames ...`); not usable by compute tools yet — an intermediate state
    before `estimate_camera_motion.py` runs.
  - `"dynamic"` — keyframes + a pointer to per-frame homographies in
    `.tmp/<match_id>/frame_homographies.npz`. The npz is disposable and
    regenerable (re-run `estimate_camera_motion.py`); the keyframe clicks in
    the json are the precious manual work and survive a `.tmp` wipe.
- `frame_homographies.npz` — `frames`, `H` (N×3×3 pixel→pitch, one per
  frame), `inliers`, `failed` diagnostic arrays.

## Tools
- `tools/ingest_video.py` — probe with OpenCV, write `meta.json`, mint match_id.
- `tools/run_tracking.py` — **hybrid detection**, two passes merged into one
  `tracks.parquet`:
  - Players: pretrained YOLO `person` class + BoT-SORT using
    `config/botsort_football.yaml` (camera-motion compensation for panning
    footage, 3 s track buffer, ReID, high new-track threshold). `--conf` is
    deliberately low (0.1): the tracker's thresholds gate quality, and
    filtering low-conf boxes before the tracker starves its second-stage
    recovery and fragments IDs (learned 2026-07-07: conf 0.3 + default
    ByteTrack gave 129 IDs / median 4-frame lifetime on a 303-frame clip).
  - Ball: separate **detection-only** pass (no tracker) via `--ball-weights`.
    Trackers silently drop detections below their `new_track_thresh`, which
    is exactly the confidence range a distant ball occupies — routing the
    ball through ByteTrack was a large part of the original 12.9% detection
    rate. Ball rows carry `track_id -1`; possession only needs per-frame
    positions. Default is the pretrained model's `sports ball` class — still
    modest recall at pitch distance; a fine-tuned ball model (via the
    existing [train_yolo_model.md](train_yolo_model.md) pipeline) slots in
    through the same flag.
  - `--imgsz` defaults to 1280, **not** YOLO's 640 — set it ≥ the video
    width. Pitch-scale players are ~40 px tall and the ball ~10-20 px;
    letting YOLO halve the frame at inference was the single biggest recall
    killer for both passes.
- `tools/stitch_tracks.py` — merge fragmented player track IDs, artifact-only
  and cheap (no video access, safe to re-run and tune freely). Extrapolates
  each dying track's exit position along its exit velocity; a track that
  starts within `--max-gap-s` and lands within a gap-scaled radius is greedily
  matched and merged (union-find; merged chain keeps the earliest ID). Uses
  pitch meters when calibration exists (pixel positions jump under a panning
  camera, pitch positions don't — so calibrate before stitching when
  possible), else pixels scaled by median box height. First run backs the
  tracker output up to `tracks_raw.parquet` and rewrites `tracks.parquet`;
  re-runs start from the backup (idempotent). Also drops 1-frame blip tracks.
- `tools/assign_teams.py` — for each player track, crop the upper half of its
  boxes across sampled frames, take the median jersey color, k-means (k=2)
  over tracks. Tracks far from both cluster centers (referees, keepers) →
  `unknown`. Writes `team_map.json` plus a cluster-balance summary to stdout.
- `tools/calibrate_pitch.py` — interactive: shows a chosen frame, user clicks
  4+ known pitch landmarks (corners, penalty-box corners, center spot), picks
  each from a menu of standard pitch points. Pitch dimensions default 105×68 m
  (`--pitch-length/--pitch-width` to override — UK adult 11-a-side grass
  pitches are often ~100×64 m; youth pitches smaller still; wrong dimensions
  scale every distance/speed stat). Computes homography with
  `cv2.findHomography`, writes `config/<match_id>_calibration.json`. For a
  fixed camera (`--frame N`) this is once per camera setup, not per match —
  reuse an existing calibration file if the camera hasn't moved. For a
  moving camera pass `--frames N1,N2,...` (2–5 keyframes spread across the
  camera's range of motion, e.g. panned fully left / center / fully right);
  frame indices are per-video, so this is per-match. Landmarks only need to
  be visible on the keyframe being clicked — partial pitch views are fine.
- `tools/estimate_camera_motion.py` — moving camera only, run after keyframe
  calibration and (ideally) after tracking so player boxes can be masked out
  of feature selection. Sparse optical flow (Shi-Tomasi + pyramidal LK with
  forward-backward check) → RANSAC homography per consecutive frame pair →
  chained from each keyframe and blended between neighboring keyframes so
  drift is pulled to zero at every anchor. Auto-detects and masks static
  overlays (letterbox bars, watermarks, scoreboards). Writes
  `frame_homographies.npz` and flips the calibration to `"mode": "dynamic"`.
  CPU-only, roughly real-time (~30 fps on 720p) — a full match is a
  ~90-minute run.
- `tools/label_roster.py` — interactive (GUI, like calibrate_pitch): shows
  anchor frame(s) with the tracked boxes drawn; the user clicks a player and
  types team (`a`/`b`) + shirt number. With a panning camera no single frame
  shows all 22 players — label one early anchor and add 1–2 more where the
  camera points elsewhere (`--frames N1,N2,...`). Re-running appends anchors,
  so identity drift found in the rendered video is fixed by adding an anchor
  after the drift point. Run after stitch_tracks (it needs boxes to click).
- `tools/assign_identities.py` — propagates the roster across all player
  tracks → `player_map.json`. Anchor-clicked tracks are seeds; remaining
  fragments are matched to free players by extrapolated position (pitch
  meters when calibrated), gated by time gap, distance, and team consistency
  vs `team_map.json` (cluster names are auto-aligned to the roster's a/b by
  majority vote over seeds, so labeling order never matters). Works backwards
  from mid-video anchors too. Anything that isn't a roster player (refs,
  spectators, noise) stays unidentified. Artifact-only and cheap; safe to
  re-run and tune (`--max-gap-s` etc.).
- `tools/render_tracking_video.py` — annotated MP4 →
  `outputs/videos/<match_id>_tracking.mp4`: team-colored ellipse + shirt
  number for identified players, thin gray box + track_id for unidentified
  ones (exactly the tracks to fix with another anchor), ball marker, HUD.
  Falls back to team colors/track ids without a player_map. Decodes and
  re-encodes the whole video (~real-time on 720p); `--start-s/--end-s` for
  clips, `--trail-s` for foot trails (pixel-space: bent under camera pans).
- `tools/compute_possession.py` — per frame where the ball is detected,
  assign possession to the nearest player foot-point within 2.5 m (calibrated)
  or ~60 px (uncalibrated), with hysteresis: possession only switches after
  ~0.4 s (fps-scaled) of consecutive frames for the new team. Team possession
  % = attributed frames / ball-detected frames.
- `tools/compute_positions.py` — per-team average positions and heatmaps in
  pitch coordinates (or pixels if uncalibrated); saves PNGs for the report.
- `tools/compute_physical.py` — smooth foot positions (~0.5 s moving average)
  before differencing so tracker jitter doesn't inflate distance; drop speeds
  > 12 m/s as tracking glitches. Team totals are primary. With
  `player_map.json` present, per-player distance/speed is aggregated across
  each player's fragments and reported by shirt number (confidence
  `identified`, with tracking coverage attached — distance during untracked
  gaps is still missing); without it, per-track numbers flagged best-effort.
- `tools/export_stats.py` — tidy CSV: `metric, scope, entity, value, unit,
  confidence` so flags travel with the numbers.
- `tools/build_report.py` — PDF via matplotlib `PdfPages` (no new deps):
  cover (teams, date, match_id), possession, positions/heatmaps, physical
  table, and always a final **data-quality appendix** (ball detection rate,
  track count, fragmentation estimate, calibration status).

## Steps
1. **Ingest**: `tools/ingest_video.py` → `meta.json`. Sanity-check fps and
   duration; stop and flag if unreadable or if duration is implausible for a
   match recording.
2. **Track**: `tools/run_tracking.py` → `tracks.parquet`. **Expensive — never
   re-run without checking with the user.** Afterwards report the ball
   detection rate; if the ball appears in < 30% of frames, flag that
   possession will be low-confidence before continuing.
3. **Stitch**: `tools/stitch_tracks.py` → rewrites `tracks.parquet` with
   fragmented player IDs merged (original kept as `tracks_raw.parquet`).
   Cheap and artifact-only. Ideally run *after* calibration exists so it can
   match in pitch meters; if calibration arrives later, re-run it (and then
   re-run teams + compute — all cheap). Must run **before** assign_teams:
   `team_map.json` is keyed by track_id.
4. **Teams**: `tools/assign_teams.py` → `team_map.json`. Sanity-check the
   cluster balance: if one team has < 25% of assigned tracks, flag a
   clustering failure and stop rather than producing a nonsense report.
5. **Calibration**: if `config/<match_id>_calibration.json` (or a reusable
   one for the same camera setup) exists, use it. Otherwise ask the user:
   calibrate now via `tools/calibrate_pitch.py` (`--frame N` fixed camera,
   `--frames N1,N2,...` moving camera), or proceed uncalibrated with
   distance/speed in pixel units, flagged everywhere. Deciding fixed vs
   moving: sample a few frames across the video and compare backgrounds, or
   ask the user.
6. **Camera motion (moving camera only)**: `tools/estimate_camera_motion.py`
   → `frame_homographies.npz`, calibration flips to `"mode": "dynamic"`.
   Check the health output: isolated identity-fallback pairs are fine,
   consecutive runs of them mean smeared positions (cuts, heavy blur);
   keyframe drift > ~3 m means keyframes are too far apart for how fast the
   camera moves — ask the user to add a keyframe inside the offending span
   and re-run (motion re-estimation is cheap; tracking is untouched).
7. **Roster & identities (recommended)**: follow
   [curate_and_identify_players.md](curate_and_identify_players.md) — it
   owns this stage (labeling UX, curation filters, manual re-tracking,
   `assign_identities.py` → `player_map.json`) and records the v2 redesign
   agreed 2026-07-07 (v1's show-every-box labeling was aborted by the user
   as too cluttered; v2 implementation may still be pending — check that
   file's status banner). Skipping this stage keeps the old per-track
   behavior everywhere downstream.
8. **Tracking video (recommended)**: `tools/render_tracking_video.py` →
   `outputs/videos/<match_id>_tracking.mp4`. Watch it (or spot-check frames)
   to verify identities stick; if a number jumps to the wrong body, add an
   anchor on a frame after the jump (`label_roster.py --frames N`) and
   re-run `assign_identities.py` + this render. Also the base for tactical
   clips (`--start-s/--end-s`).
9. **Compute**: run `tools/compute_possession.py`, `tools/compute_positions.py`,
   `tools/compute_physical.py` — each reads only the pipeline artifacts, not
   the video.
10. **Export**: `tools/export_stats.py` → `outputs/stats/<match_id>_stats.csv`.
11. **Report**: `tools/build_report.py` → `outputs/reports/<match_id>_report.pdf`.
12. **Upload (deferred)**: not performed in v1 — the report and CSV stay in
    `outputs/` only. Revisit this step if/when the user authorizes the
    claude.ai Google Drive connector or asks for a `tools/upload_outputs.py`
    OAuth uploader; until then, skip straight from Step 11 to done.

## Outputs
- `outputs/stats/<match_id>_stats.csv`
- `outputs/reports/<match_id>_report.pdf`
- `outputs/videos/<match_id>_tracking.mp4` (when Step 8 runs)
- (Google Drive delivery deferred — see Step 12.)

## Edge Cases
- **Ball rarely detected** → still produce the report; mark possession
  low-confidence in both the CSV `confidence` column and the PDF appendix.
- **No calibration** → distance/speed in pixel units, flagged uncalibrated
  everywhere they appear.
- **Track IDs fragment badly** → team-level stats stand (fragmentation
  doesn't change which team a segment belongs to); without a roster,
  per-player numbers are best-effort and say so. With a roster, fragments
  are re-joined per player — coverage in `player_map.json` shows how much
  of the video each player is actually tracked for.
- **Identity jumps in the rendered video** (a shirt number switches bodies)
  → the tracker swapped IDs mid-track or propagation matched the wrong
  fragment; add a roster anchor on a frame after the jump and re-run
  `assign_identities.py` (cheap). A track labeled with two different players
  across anchors is reported at assign time — that's the same symptom.
- **Roster player at 0% coverage** → their anchor click didn't land on any
  box (player untracked on that frame); re-run `label_roster.py` with an
  anchor where they're visibly boxed.
- **Clustering failure** (one team nearly empty) → stop after step 4 and ask
  the user; likely causes are similar kit colors or a keeper-heavy sample.
- **Moving camera** → supported for pan/tilt/zoom from a fixed position via
  keyframe calibration + `estimate_camera_motion.py` (see Steps 4–5).
  Because `tracks.parquet` stores raw pixels and projection happens at
  compute time, tracking never needs re-running when calibration changes.
  Known limits:
  - a camera that *travels* (drone, handheld walking) breaks the
    single-homography-per-frame assumption — results degrade with parallax;
  - hard cuts / replays in edited footage break the flow chain — the tool
    warns about consecutive pair failures; trim to continuous footage or
    put a keyframe right after each cut;
  - long broadcast / tactical-camera footage (a whole half, constant pan and
    zoom): keyframe chaining drifts by tens of metres - use
    [track_identities.md](track_identities.md) instead (`calibrate_auto.py`
    re-anchors every frame to the painted lines; learned 2026-11-03);
  - pixel-space visualizations (`track_players.py --trajectories`) mix
    camera and player motion on moving footage — use the pitch-space
    figures from `compute_positions.py` instead.
- **`.tmp` wiped after calibration** → keyframe clicks survive in `config/`;
  re-run `estimate_camera_motion.py` (and tracking, if `tracks.parquet` is
  gone — that one needs the usual cost check with the user).
