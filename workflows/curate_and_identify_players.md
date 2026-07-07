# Workflow: Curate & Identify Players (roster v2)

> **Status: v2 IMPLEMENTED + VERIFIED 2026-07-07** (synthetic fixture on
> `testvideo_b95783a3`; see "Verification" for what was run). What remains
> is the real labeling session — follow "Run procedure" with the user at
> the keyboard. The GUI itself has only been exercised synthetically, so
> expect to fix small interaction papercuts after the first live session.
>
> **What existed from v1 (built + verified 2026-07-07 on
> `testvideo_b95783a3`, real panning Veo footage):**
> - `tools/label_roster.py` — GUI: click tracked boxes on anchor frames,
>   type team (`a`/`b`) + shirt number → `config/<match_id>_roster.json`
>   (click-anchored, so re-stitching never invalidates it).
> - `tools/assign_identities.py` — propagates roster across track fragments
>   (seeds from anchor clicks; greedy best-match on extrapolated pitch
>   position, gated by time gap / distance / team consistency, with
>   automatic team_a/team_b ↔ a/b alignment) → `.tmp/<match_id>/player_map.json`.
> - `tools/render_tracking_video.py` — annotated MP4 (team-colored ellipse +
>   shirt number, gray boxes for unidentified tracks, ball marker, trails)
>   → `outputs/videos/<match_id>_tracking.mp4`.
> - Per-player stats wiring: `compute_physical.py` aggregates by shirt
>   number when `player_map.json` exists (confidence `identified` +
>   coverage); `export_stats.py` and `build_report.py` show players instead
>   of raw track IDs. The no-roster fallback path still works unchanged.
>
> **Why v2:** the first real labeling attempt was aborted by the user. The
> GUI showed *every* tracked box — spectators behind the pitch, referees,
> coaches — so finding the actual players among the clutter was tedious
> (log showed 10 players labeled fine on frame 0, then abort on frame 205).
> v1 also had no way to remove a non-participant, no way to add a player
> the detector missed, and no way to correct the ball.

## Objective
A curation + identity layer between tracking and stats. After this stage:
- at most 11 players per team, each with team + shirt number, identities
  maintained across the whole video by the tracking;
- referees, spectators, coaches, ball-boys explicitly excluded from ALL
  downstream stats (not just left "unknown");
- ball positions correctable where detection missed;
- an annotated output video to debug identity drift and cut tactical clips.

## Required Inputs
- Completed pipeline through stitching + team assignment + calibration
  (steps 1–6 of [generate_match_report.md](generate_match_report.md)).
  Calibration matters doubly here: the pitch-boundary filter needs it.
- The user at the keyboard for the labeling step (GUI clicks). Always print
  the controls cheat sheet BEFORE launching the GUI, and launch it as a
  background task so they can take their time.

## What v2 adds (implemented 2026-07-07)

### 1. Auto-filter "who's involved" before the user clicks (label_roster.py)
Two automatic filters; boxes failing either are drawn dim + dashed with an
`out`/`kit?` tag instead of as candidates, and a click rescues them
(override always wins):
- **Pitch boundary**: each box's foot point is projected through the
  calibration; outside the pitch rectangle + margin (`--pitch-margin-m`,
  default 2 m) → dimmed. Kills most spectators/coaches deterministically.
  Skipped with a warning when no calibration exists.
- **Kit colors**: `team_map.json`'s `unknown` verdict IS the
  far-from-both-clusters signal (assign_teams.py already computes it), so
  no video access needed — `unknown` tracks are dimmed `kit?`. For boxes
  near a cluster the team letter is pre-filled once ≥2 of your labels
  establish which cluster is `a` and which is `b` (majority vote); you
  usually only type the number.
- `--no-auto-filter` turns both off.

### 2. Curation actions in the GUI (label_roster.py)
- Click box → type `a`/`b` + number, ENTER (unchanged from v1).
- `x` with a box selected → **ignore the whole track**; `x` again
  un-ignores. Persisted per anchor as click-anchored `ignores`/`rescues`.
- **Drag on open ground → manual player box**, then label as normal.
  Dragging over an already-tracked person selects them instead (no
  duplicate bodies; IoU ≥ 0.4 test).
- **`o` = ball mode**: click marks the ball on this frame; `d` undoes the
  last mark; `o`/ESC exits.
- **Q now SAVES partial work** (roster flagged `"partial": true`) — the
  v1 abort that lost 10 labels can't happen again. Re-opening an anchor
  frame MERGES with its existing content instead of replacing it.
- Roster schema per anchor: `labels` (manual-box labels carry
  `"manual": true` and `track_id_at_label: null`), `ignores`, `rescues`,
  `manual_boxes` (box + key), `ball_marks`.

### 3. Manual boxes → full re-tracking (tools/retrack_manual.py)
- **Per-frame YOLO re-detection on a zoomed ROI crop** (predicted from the
  last box + velocity), forward AND backward from the anchor frame.
  NOT CSRT as originally designed: this machine's OpenCV (5.0, no contrib)
  doesn't ship CSRT/KCF, and ROI re-detection was already the agreed
  escalation for CSRT drift — it can't drift onto grass because every
  frame must re-detect an actual person. Verified: tracked a player
  through a synthetic 61-frame detection gap with zero drift, CPU-only.
- Stops on **handoff** (IoU ≥ 0.5 with one auto track for 5 consecutive
  frames → rows trimmed to the handoff point; assign_identities carries
  the identity across the join), on **lost** (12 straight frames without
  an acceptable re-detection, coasting on velocity in between), or at the
  video ends. Size-jump gate (0.4–2.5× area) rejects wrong-person grabs.
- Appends rows with fresh track_ids + `source="manual"`; backs up to
  `tracks_auto.parquet` first and always rebuilds from that backup
  (idempotent; if stitch_tracks is re-run it notices the fresh base and
  re-backs-up — but you must re-run this tool after any re-stitch).
- Ball marks merge here too: `class="ball"`, conf 1.0, `source="manual"`;
  auto ball rows on a marked frame are dropped (manual beats auto).

### 4. Ignores flow downstream
- `assign_identities.py` resolves ignore/rescue clicks, re-runs the
  pitch-boundary filter over EVERY track's whole life (`--outside-frac`,
  default: >60% of rows outside pitch+margin → auto-ignored; catches
  spectators on frames you never anchored), and writes
  `"ignored_tracks": [{track_id, reason: user|off_pitch, detections}]`
  into player_map.json. Labels/rescues always win over both filters.
  Ignored tracks can't take a roster slot during propagation.
- `compute_physical/possession/positions.py` all map teams through
  `_common.resolve_track_teams()`: ignored tracks are excluded from ALL
  stats (team totals, possession, heatmaps), and identified tracks take
  their player's roster team — player_map.json's `team_map_alignment`
  records whether the jersey clusters were flipped relative to your a/b
  labels, so manual tracks (absent from team_map) still count toward the
  right team totals.
- `render_tracking_video.py` hides ignored tracks; `--show-ignored` draws
  them dimmed gray with an `ign#id` tag for debugging.
- `build_report.py`'s data-quality appendix states how many tracks were
  excluded and why.

### 5. Fallback unchanged (regression-checked)
No roster → no player_map → `resolve_track_teams` degrades to raw
team_map; per-track best-effort stats identical to v1.

## Verification (ran 2026-07-07, all passed)
Everything below ran against a scratch COPY of `.tmp/testvideo_b95783a3`
(`--tmp-dir`/`--roster`/`--output` overrides keep the real artifacts
clean) with a synthetic roster generated from team_map at anchors 0 + 205.
- **Pitch-boundary filter**: 5 tracks are ≥78% off-pitch (2, 3, 17, 20,
  43) — the spectators/coaches behind the lines — and 4 of them had been
  jersey-clustered into a team, confirming the stat-inflation bug v2
  fixes. All auto-ignored; a synthetic rescue on track 2 correctly kept
  it un-ignored.
- **retrack_manual**: deleted a track's rows for frames 100–160 and split
  its id; a manual box at frame 130 re-tracked 30 frames each way with no
  drift and stopped with "handoff to track #1" / "handoff to track #99".
  assign_identities then gave that player (a99) all 3 fragments —
  **100% coverage across the manual↔auto joins**. Idempotent re-run
  rebuilt identical output from tracks_auto.parquet. ~60 tracked frames
  took seconds on CPU (backward seeks per frame; fine at this scale).
- **Ignores downstream**: player_map ignored_tracks = 1 user + 4
  off_pitch; compute_physical dropped them from team totals
  (226.5→207.5 m, 78.6→70.5 m), possession stopped attributing frames to
  them, positions excluded them, render hid them (dimmed under
  `--show-ignored`, checked by extracting frames).
- **Regression**: full compute→export→report chain with NO roster
  produced the same fallback behavior/numbers as v1.
- **Not yet exercised**: the GUI interactions themselves (click/drag/x/o)
  — needs the live user session.

## Run procedure (with the user)
1. Prereqs: steps 1–6 of generate_match_report.md done for the match.
2. **Print the controls cheat sheet, then launch** `tools/label_roster.py
   --match-id <id> --frames <anchors>` **in the background** and wait for
   the user to finish. Anchor guidance: 2–3 frames spread across the
   camera's pan range; for testvideo use `0,205`. Controls to communicate:
   - click a box → `a`/`b` + shirt number → ENTER; the team letter may
     already be pre-filled from the kit color once a few labels are in;
     any distinct 1–11 numbering is fine if real numbers are unreadable
   - dim dashed boxes (`out`/`kit?` tag) = auto-filtered spectators/refs;
     click one to rescue it if it's actually a player
   - `x` = ignore selected person (excluded from all stats); `x` again
     un-ignores
   - drag over a missed player → new magenta box, then label it
   - `o` = ball mode, then click the ball (`d` undoes, `o`/ESC exits)
   - ESC deselect, BACKSPACE fix digits, `d` remove a label,
     ENTER (nothing selected) = next frame,
     Q = quit early — SAVES progress as a partial roster (v2; relaunch
     the same command to continue, anchors merge)
3. `tools/retrack_manual.py` — only needed if manual boxes/ball marks were
   made (label_roster prints which next command applies). Safe to re-run.
4. `tools/assign_identities.py` — check the printed coverage table:
   0%-coverage player → their click missed every box (re-anchor them);
   coverage >100% / "two tracks alive at the same time" warning → the
   same number was put on two different people (fix via d/relabel on an
   anchor); substantial unidentified tracks listed → candidates for `x`
   or a label on a new anchor.
5. `tools/render_tracking_video.py` → have the user watch it. Identity
   jumps → add an anchor after the jump (label_roster appends), re-run
   steps 4–5 (both cheap).
6. `tools/compute_physical.py` (+ possession/positions if wanted) →
   `tools/export_stats.py` → `tools/build_report.py` — per-player numbers
   now keyed by shirt number.

## Edge cases
- **No calibration** → pitch-boundary filter unavailable; warn and rely on
  kit colors + manual `x`.
- **Similar kits** → color filter unreliable; assign_identities already
  warns when seeded tracks disagree with clustering after alignment.
- **Manual box over someone who IS auto-tracked** → the GUI selects the
  existing track instead of creating a box (IoU ≥ 0.4).
- **Re-track drift onto the wrong person** → the size-jump gate and
  IoU-with-prediction matching make this rare, but a crowd crossing can
  still swap targets; the user sees it in the rendered video → `x` the
  bad track or re-anchor. If it recurs, raise `--min-iou` / lower
  `--roi-scale`.
- **Ball marks vs auto ball on same frame** → manual wins (auto rows
  dropped on marked frames).
- **Subs** (>11 distinct numbers per team over the match) → allowed, warn
  only; the 11-per-team cap is per-moment, not per-match.
- **User quits the GUI (Q)** → v2 saves everything so far with
  `"partial": true`; relaunching merges. The v1 behavior (Q lost 10
  labels) is gone.
- **Track both labeled and ignored** (conflicting clicks on different
  anchors) → the label wins; assign_identities warns.
- **Re-running stitch_tracks after retrack_manual** → manual rows are
  lost from tracks.parquet; re-run retrack_manual (it detects the fresh
  base and re-backs-up).

## Notes / history
- 2026-07-07: v1 built and verified end-to-end; first user labeling
  attempt aborted due to background-people clutter → this v2 design.
  User decisions: ball marking IN scope; manual boxes get FULL re-tracking
  (CSRT/ROI approach above), not cheap linking; auto-filter = pitch
  boundary + kit colors combined, both user-overridable.
- 2026-07-07 (later): v2 implemented + verified on the synthetic fixture.
  Learnings baked into the tools:
  - CSRT unavailable (opencv-python 5.0 ships no contrib trackers) →
    went straight to ROI YOLO re-detection; it also self-verifies
    (must re-detect a person every frame), so no separate drift check
    was needed. yolov8n.pt at repo root is the default weights.
  - Kit-color filtering needs no new machinery: team_map.json `unknown`
    already means "far from both kit clusters".
  - The team_map↔roster a/b alignment that assign_identities computes is
    now persisted (`team_map_alignment`) because compute tools need it to
    put manual tracks (absent from team_map) into the right team totals.
  - testvideo fixture note: track 1 is the touchline-walking coach — only
    ~21% off-pitch, so the auto filter correctly leaves him; he's the
    poster child for the manual `x`.
  - The real user labeling session has NOT happened yet as of this note.
- 2026-07-07 (evening): **first real v2 labeling session completed** on
  testvideo (13 players, 9 ignores, 8 manual boxes, 2 ball marks). The
  GUI flow held up (rescue→x, drag, ball mode, prefill all used).
  Findings → fixes:
  - **Off-pitch ball false positives**: 100/203 "ball" detections were a
    spare ball behind the touchline; pre-curation possession (55.6/44.4)
    was attributed to the spectators next to it — fiction. Fix:
    compute_possession now drops ball detections projecting off the
    pitch (`--ball-margin-m`) and confidence drops to `low` when <20% of
    ball frames attribute (a % built from a handful of frames is noise).
  - **Same number on two coexisting people** happened 3× (labels on
    different anchors landed on different bodies). assign_identities now
    prints an explicit "two tracks alive at the same time" warning with
    track ids + frame ranges; fix is d/relabel on an anchor.
  - **Users leave manual boxes unlabeled** (6 of 8 here — some were
    probably accidental drags). retrack_manual tracks them anyway and
    warns; they stay unidentified and out of team totals. Papercut
    backlog: the GUI could prompt about unlabeled boxes before leaving a
    frame, and could let you delete a box after deselecting it (today
    you must click it, then `x`).
  - Two ball marks placed ~26 px apart across anchors 0/205 both persist
    (each is per-frame — that's correct behavior, not a dupe).
