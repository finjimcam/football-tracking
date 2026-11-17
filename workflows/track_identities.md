# Workflow: Persistent Player Identities on Broadcast Footage (+ verification, + phase clips)

> **Status: implemented + run 2026-11-03** on *Man City 2-3 Tottenham* (PL, 19 Feb
> 2022), Premier League tactical-camera feed, first half (no subs). Results
> and learnings from that run are recorded at the bottom. Use this workflow
> instead of [generate_match_report.md](generate_match_report.md) steps 2-7
> whenever the footage is a long broadcast/tactical feed: that workflow's
> keyframe + optical-flow calibration drifts by tens of metres over a 45-min
> half, and its per-track IDs + manual roster can't keep 22 identities for
> that long.

## Objective
Every one of the 22 players keeps ONE identity for the whole analysis window
(e.g. a half with no substitutions), on footage from a camera that pans and
zooms non-stop; then **check** that it holds - internally and against the
published match data - and cut tactical clips (e.g. one team's
out-of-possession phases) from it.

## Required inputs
- Match video (`.mp4`), ingested with `tools/ingest_video.py` (match_id is the
  file stem, made path-safe, + SHA-1 prefix).
- `config/<match_id>_match.json` - **hand-verified**: team names, kits
  (colour class -> team map, see below), `clock_sync` (kickoff frame; read the
  scoreboard's seconds-digit ticks), `analysis_window` (start/end frames -
  start where the clean tactical camera begins, end at the whistle).
  Must contain a window **without substitutions** if identities are meant to
  be 11 per team throughout.
- `config/<match_id>_reference.json` - published facts with sources: Opta
  team stats, lineups as formation lines (right-to-left from each team's
  own perspective), timestamped events (seconds since kickoff). For PL games
  the Premier League's own API serves all three (see the file's `sources`).
- `config/<match_id>_calibration.json` - seed: >= 8 landmark clicks on ONE
  frame (calibrate_pitch.py keyframe format). Pick a frame with a penalty
  area + halfway line in view; landmarks can be read off a gridded zoom.

## Tools (in order)
| # | Tool | Cost (Core Ultra 5 135U, 47-min half) | Output |
|---|---|---|---|
| 1 | `tools/detect_frames.py` | **~55 min on the iGPU** (OpenVINO, 21 fps) - the only expensive step | `detections.parquet` (+ resumable chunks) |
| 2 | `tools/calibrate_auto.py` | ~55 min CPU, runs *concurrently* with step 1 (it waits for detection chunks) | `frame_homographies.npz`, calibration -> `"dynamic"` |
| 3 | `tools/track_identities.py` | ~10 min (two tracking passes + 8 flow solves); `--reuse-tracklets` re-links from its cache in ~3 min | `tracks.parquet`, `player_map.json`, `team_map.json`, `identity_links.parquet`, `tracklets_cache.parquet` |
| 3b | `tools/duel_gaps.py` (optional, issue #47) | ~6 s | duel-gap fill rows in `tracks.parquet` (`interp`, `absorbed_by`), `duel_gaps.json` review list |
| 4 | `tools/track_ball.py` | < 1 min | cleaned ball rows in `tracks.parquet`, `ball_track.parquet` |
| 5 | `tools/compute_possession.py --include-unidentified` | < 1 min | `possession.json`, `possession_timeline.parquet` |
| 6 | `tools/compute_physical.py --smooth-s 1.0`, `tools/compute_positions.py` | < 1 min | `physical.json`, `positions.json` + figures |
| 7 | `tools/verify_tracking.py` | ~2 min (contact sheets read the video) | `verification.json`, `outputs/verification/<id>/` |
| 8 | `tools/compare_reference_stats.py` | seconds | `outputs/stats/<id>_vs_reference.csv` |
| 9 | `tools/render_tracking_video.py --minimap --possession` | ~real-time | `outputs/videos/<id>_tracking.mp4` (~5 MB/min) |
| 10 | `tools/cut_phase_clips.py --team team_a [--shape]` | ~real-time on the clip length | `outputs/videos/<id>_team_a_out_of_possession*.mp4` + phase index CSV |

Steps 3-10 read only cached artifacts: **re-running the tracking ("putting
the footage back through") costs minutes, not hours** - tune
`track_identities.py` flags and re-run 3-10. Never re-run step 1 without
checking with the user.

### Commands
```
python tools/ingest_video.py --video "<video>.mp4" --config config/<id>_match.json
python tools/detect_frames.py --match-id <id>                       # background, ~55 min
python tools/calibrate_auto.py --match-id <id> --check-frames 5000,30000,60000   # background, concurrently
python tools/track_identities.py --match-id <id>
python tools/duel_gaps.py --match-id <id>                           # optional: fill duel gaps
python tools/track_ball.py --match-id <id>
python tools/compute_possession.py --match-id <id> --include-unidentified
python tools/compute_physical.py --match-id <id> --smooth-s 1.0
python tools/compute_positions.py --match-id <id>
python tools/verify_tracking.py --match-id <id>
python tools/compare_reference_stats.py --match-id <id>
python tools/render_tracking_video.py --match-id <id> --minimap --possession --team-a-color 235,190,80 --team-b-color 245,245,245
python tools/cut_phase_clips.py --match-id <id> --team team_a
python tools/cut_phase_clips.py --match-id <id> --team team_a --shape
# steadier lines through hand-picked players:
python tools/select_shape_players.py --match-id <id> --team team_a
python tools/cut_phase_clips.py --match-id <id> --team team_a --shape-selection
# only the phases you kept in the Lines tab, trimmed, one MP4 each, zipped for PowerPoint:
python tools/cut_phase_clips.py --match-id <id> --team team_a --shape-selection --export-selection --separate
```

## How each stage works (and why)
- **Detection** (`detect_frames.py`): yolo26n person + sports-ball at full
  resolution, every frame. yolo26n was chosen on this footage: same person
  recall as yolov8n, ~2x the ball hits, same speed. Per person box it stores
  shirt/shorts Lab colour so nothing downstream re-reads the video. Ball conf
  floor 0.05 - the ball is 4-8 px; false positives are removed later by
  geometry, which a confidence threshold can't do.
- **Calibration** (`calibrate_auto.py`): the tactical camera is one fixed
  optical centre that pans/tilts/zooms, so after solving its position once
  (seed landmarks + a self-calibration over 16 views across the window) each
  frame needs only pan, tilt, focal length. Optical flow predicts them; the
  projected pitch-line model is then snapped onto detected white line pixels
  (top-hat filter, truncated chamfer distance). Re-anchoring every frame to
  the painted lines is what keeps it drift-free for 47 minutes. Check the
  `--check-frames` overlays in `.tmp/<id>/calib_check/`.
- **Identities** (`track_identities.py`): duplicate-box suppression ->
  pitch-space tracklets -> a self-trained team colour model (gradient-boosted
  on pure tracklets) relabels flickering kits -> re-track -> cut tracklets at
  sustained team flips -> **global min-cost-flow linking**: exactly 10
  disjoint outfield chains per team, rewarded per second of play covered,
  edges only where the gap is physically runnable, cost = distance from the
  predicted position + (for long gaps) change of place in the team shape.
  Keepers by kit colour + position, inside their own goal zone (<= 22 m from
  the line; the referee's kit reads the same as Ederson's). Names:
  identities matched to the official formation lines by average depth/width
  (**role-inferred**).
- **Ball** (`track_ball.py`): drops detections projecting > 1 m off the pitch
  (spare balls), links the rest into ball tracklets, keeps the strongest
  per frame, interpolates gaps <= 12 frames.
- **Possession** (`compute_possession.py`): nearest player within 2.5 m of the
  ball, 0.4 s hysteresis before switching, possession held through ball
  dropouts for up to 8 s -> per-frame `possession_timeline.parquet`.
  Turnovers shorter than the hysteresis are absorbed into longer phases.
- **Physical** (`compute_physical.py`): 1 s smoothing; reports the raw max
  speed AND `top_speed_p99` - on broadcast footage the raw max sits at the
  12 m/s glitch cap (tracking jumps), the 99th percentile gives realistic
  21-37 km/h.
- **Phase clips** (`cut_phase_clips.py`): runs of the opponent's possession,
  merged across gaps < 2 s, dropped if < 3 s, padded 1.5 s / 1 s. `--shape`
  draws the back line (4 deepest visible outfielders) and midfield line (next
  3) per frame by depth from their own goal - deliberately NOT from
  identities, which swap among rotating players - and only when the line is
  coherent (<= 12 m deep) with >= 7 outfielders on screen. That per-frame
  guess flickers (line membership changed on ~23% of frames in the City
  out-of-possession compilation). For a steady version, pick the players by
  hand once per phase with `select_shape_players.py` (window; pre-filled with
  the guess; saves `config/<id>_<team>_<phase>_shape_selection.json`, keyed by
  phase start frame, so keep the same --merge-s/--min-s/--pre-s/--post-s),
  then `cut_phase_clips.py --shape-selection`. Each pick is a *chain*: the
  picked track id, handed to a *fresh* track nearby (<= 3 s, <= 15 m) if it
  fragments - fresh only, so a lost pick can't jump onto an unpicked
  neighbour. Gaps <= 4 s between sightings are filled by interpolating in
  pitch space and projecting back through that frame's homography (pixel
  interpolation is wrong on a panning camera); longer gaps stay open. The
  selector's bottom strip shows each chain per frame (green detected, amber
  interpolated, red lost, white tick = tracker hand-off); click it to seek,
  right-click the right player after a gap to re-link the chain. On the 14
  phases picked so far: 91% of chain-frames detected, 6% filled, 3% open.
  Phases without a pick draw nothing (`--unselected auto` = old guess).
  Output: `<id>_<team>_<phase>_shape_selected.mp4`. `--auto-fill` writes the
  guess for every phase without a window (smoke test; membership changed on
  ~4% of frames instead of ~23%).
  **Clip export for PowerPoint (2026-11-04, #37):** in the Lines tab, tick
  the phases to export (Keep all / Drop all to start) and trim each with
  *Start here* `[` / *End here* `]`. Trims and keep/drop choices are saved as
  you go to `config/<id>_<team>_<phase>_export.json`, keyed by phase start
  frame like the picks. A phase missing from the file is kept untrimmed.
  *Export clips (zip)* runs `--shape-selection --export-selection
  --separate`: one H.264 MP4 per kept phase (`phase_NN_MMmSSs.mp4`, where NN is
  the picker's phase number), plus `phases.csv`, in
  `outputs/videos/<id>_<team>_<phase>_shape_selected_clips.zip`. The match
  page offers the zip for download. Each export replaces the last one, so
  dropped phases disappear. The per-clip folder is deleted once zipped, to
  stay inside the 15 GB data disk. Trims only change what is rendered: chains
  are still followed over the whole phase, so a pick anchored outside the
  trim still draws inside it. On pphub a 2-clip, 18 s export took 7.5 s.
  **Web picker (2026-11-04):** the Stopframe app's Lines tab
  (`tools/serve_app.py`, see `app/README.md`) does the same job in a browser,
  so no desktop window is needed. It reads and writes the same selection
  file. **Fix swap** (`x`) repairs a chain that jumped to the wrong player
  at a collision. Tap the chain, then the right player. The cut goes in at
  the two players' closest approach in the previous 6 s, not where you
  tapped, and a picked partner chain swaps back too. These are saved as cut
  anchors `[frame, track_id, 1]` ("from this frame on"); plain anchors still
  split halfway.
  **Chains are exclusive:** one track belongs to at most one chain per frame.
  Before this fix, a chain that lost its player could hand off onto a track
  another picked chain was already following, and two markers sat on one
  player. That happened on 26% of the picked frames (2604 of 10162). With
  exclusivity the 14 picked phases read 83% detected, 5% filled and 12% lost.
  The difference was fake "detected" frames that are now honest gaps to
  re-link.
- **Gap status** (`gap_status.py`, issue #47 item 2): every identity row
  in `tracks.parquet` has a `status`. `detected` is a real box.
  `interpolated` fills a gap of <= 4 s that the player could have run
  (distance <= 3 m + 8.5 m/s x gap), with a straight line in pitch space.
  `absorbed` is the same fill, but the player was last seen within 1.5 m of
  another player and came back within 2.5 m of that same player (a duel).
  `unknown` means the gap is too long, not runnable, or before the first or
  after the last sighting. Unknown frames have no row. `gaps.parquet` has one
  row per gap (kind, reason, partner track). Fill rows have no box (x1..y2
  NaN, conf 0); their foot pixels are projected back through the frame's
  homography. `track_identities.py` labels as it writes tracks.parquet
  (`--no-gap-status` turns it off). `python tools/gap_status.py --match-id
  <id>` re-labels in place in ~10 s, so you can tune `--max-fill-s` and
  `--duel-m` without re-linking. Heatmaps and average positions
  (`compute_positions.py`) weight detected 1, interpolated 0.5, absorbed 0.5
  (`--w-interpolated` / `--w-absorbed`). `compute_physical.py` counts filled
  steps towards distance and time (`distance_filled`), but speed peaks only
  come from detected-to-detected steps. Possession, contact swaps,
  verification, renders and the Lines chains read detected rows only
  (`gap_status.observed()`). New fillers (duel filler, smoother) plug in
  through `label_tracks(..., classify=, fillers={kind: fn})`.
- **Duel gaps** (`duel_gaps.py`, issue #47 item 3): a player who vanishes
  within 1.5 m of another tracked person, and comes back within 2.5 m of
  that same person, was hidden behind them in a merged box. The gap is
  filled with the partner's path plus an offset blended from the entry
  offset to the exit offset. The partner must be one tracklet through the
  gap (a partner chain re-linked mid-gap once jumped onto the referee). At
  the split the fill is checked: kit colour while the two stand > 1.2 m
  apart (opponents can't swap teams), and for teammates exit motion and
  side. A failed check goes to `duel_gaps.json` for review and is not
  written. Filled rows have `interp=True`, `conf` 0, `absorbed_by` = the
  partner's track id ("k+P together"), and show amber ("filled") in the
  Lines tab. Re-runs replace the previous fill. On the 2026-11-17 reference
  run: 2,023 duel gaps, 1,536 filled (6,488 frames), 69 flagged, 418 with no
  usable partner. Most duel gaps are under 1 s, and only 40 last 3-10 s
  (4 filled). In longer duel gaps the player usually comes back far from
  the partner. Spot check: 9 of 10 fills on the right player (two about
  1 m off inside the merged box). All 3 flagged fills were real problems.
  `fill(gap, ctx)` is the plug-in for the gap-status labeller (status
  `absorbed`).
- **Contact swaps** (`contact_swaps.py`, issue #34): this is a proxy for
  identities that swap when two teammates collide. Take each identity's
  average place in the team shape for 3 s before and 3 s after every contact
  (< 1 m). If "A before ≈ B after" fits better than "A before ≈ A after" by
  more than 2 m, flag the contact as a swap. On the 2026-11-04 baseline,
  City had 11 of 87 evaluable contacts flagged and Spurs 29 of 129. The
  check can't see a swap during a full crossing, because the swap looks
  like a bounce. Use it to compare variants, not as a true error rate.
  `track_identities.py --contact-m` (off by default) cuts both tracklets at
  same-team contacts so the global linker decides who is who. `--w-vel` adds
  velocity continuity to short links. Experiments, and the standing
  improvement loop, are tracked in issue #35.
- **Short-gap fill** (`interp_gaps.py`, issue #47 item 4): `track_identities.py`
  fills identity gaps of up to `--interp-max-s` (default 3 s, 0 = off) before
  writing tracks.parquet. Each gap is filled with a constant-velocity Kalman
  filter and RTS smoother run over the player's own sightings 1 s either side.
  The curve is shifted so it joins both sightings exactly. The fill keeps the
  player's velocity through the gap, where a straight line has a kink at each
  end. Peak speed is capped at 10 m/s and acceleration at 8 m/s²: if the curve
  breaks either cap, the straight line is used instead. If even the straight
  line needs more than 10 m/s, the gap stays empty, because that is a tracking
  jump rather than a hidden player. Gaps whose last sighting was within 1.5 m
  of another identity are skipped (duels belong to the duel filler). Fill rows
  carry `interp=True`, `status="interpolated"`, `conf=0`, a foot point
  projected through that frame's homography, and a box. Lines chains and
  `contact_swaps.py` drop them (`strip_fill`). `python tools/interp_gaps.py
  --match-id <id>` re-applies the fill to an existing tracks.parquet, and
  `--holdout` prints the error table.
  **Hold-out test (2026-11-17):** fully seen stretches were blanked and
  refilled, and the fill was compared with the hidden detections. Parameters
  were tuned on even match minutes and the table below is from the odd
  minutes, 600 stretches per length. Values are mean / p95 of each stretch's
  mean error in metres:

  | gap | straight line | RTS |
  |---|---|---|
  | 0.5 s | 0.15 / 0.32 | 0.15 / 0.32 |
  | 1 s | 0.20 / 0.44 | 0.18 / 0.37 |
  | 2 s | 0.39 / 0.87 | 0.27 / 0.56 |
  | 3 s | 0.71 / 1.76 | 0.41 / 0.96 |
  | 5 s | 1.30 / 3.01 | 0.72 / 1.83 |
  | 8 s | 2.48 / 5.55 | 1.47 / 3.22 |
  | 10 s | 3.21 / 7.10 | 2.07 / 4.78 |

  RTS at 5 s is about as accurate as the straight line at 3 s. A cubic
  Hermite spline came close to RTS but was slightly worse at every length.
  These stretches were on camera, while real gaps often happen off camera, so
  treat the table as a lower bound on the error. On the reference half the
  fill covered 12,219 of 15,322 gaps (44k frames). Missing frames in gaps
  under 3 s fell from 55k to 12k. It skipped 1,774 duel-like gaps and 958
  implausible jumps. Team distance rose about 1%, and the median p99 top speed
  fell from about 7.1 to 6.6 m/s. The filled stretches are smooth, so count
  them for distance but not for speed peaks.
- **Verification** (`verify_tracking.py`): see "Reading the verification".

## Reading the verification ("does it match?")
`verify_tracking.py` prints and writes (`outputs/verification/<id>/`):
- **link stability** - share of identity links that survive three perturbed
  re-solves. < ~85% for a team = treat per-player numbers as approximate.
- **implausible hand-offs** - links implying > 8 m/s; each one is in
  `uncertain_moments.csv` with video time + match clock.
- **kit consistency** - share of an identity's readable detections in its own
  team's colours (keepers excluded). Low = the chain picked up opponents.
- **role separability** - how distinct each identity's place in the team
  shape is (mean distance between identities' average offsets / their
  spread). Below 1 = identities are mixing; it catches swaps that no single
  link check can see.
- **Opta events** (`event_checks.csv`) - possession at the moment of each of
  the 23 timestamped first-half events, and whether the named player
  (Cancelo's shots, Rodri's fouls won...) is the event team's player on the
  ball at the last ball sighting before the event (offsides skipped). This
  tests identity persistence AND the name mapping at once.
- **contact sheets** - one row per player, crops spread through the half in
  time order. An identity swap shows as a different-looking person mid-row.
- **the tracking video** - `--minimap` keeps off-screen players as hollow
  markers at their last known spot; watch a player leave and re-enter.
To fix a bad stretch: scrub to the moments in `uncertain_moments.csv`; tune
`track_identities.py` (e.g. `--w-role`, `--role-frame lane`, `--vmax`,
`--max-gap-s`, with `--reuse-tracklets --no-stability` for quick
experiments) and re-run steps 3-10, or add manual roster anchors
(curate_and_identify_players.md - not yet wired into the global linker).

## Outputs
Committed (the repo is public; nothing cut from footage):
- `outputs/reports/<id>_report.pdf` - possession, positions, shape in/out of
  possession, physical, identity verification, Opta event checks, comparison
  with published data, data-quality appendix.
- `outputs/stats/<id>_stats.csv` (tidy metrics with confidence flags),
  `<id>_vs_reference.csv`, `<id>_team_a_out_of_possession_phases.csv`.
- `outputs/verification/<id>/` - `identity_report.csv`, `event_checks.csv`,
  `uncertain_moments.csv`.
Local only (`.gitignore`: footage and anything cut from it):
- `outputs/videos/<id>_team_a_out_of_possession.mp4` and `..._shape.mp4`.
- `outputs/videos/<id>_team_a_out_of_possession_shape_selected_clips.zip`
  (clip export for PowerPoint, see Phase clips).
- `outputs/videos/<id>_tracking_match00-10min.mp4` (verification sample; the
  full-half render was deferred for disk space - see learnings).
- `outputs/verification/<id>/contact_sheet_team_*.jpg`.
- `.tmp/<id>/` - all intermediates (disposable; `detections.parquet` is the
  one expensive to regenerate, ~55 min).

## Edge cases & learnings (2026-11-03 run)
- **Disk**: the machine's C: was at 99%; background app updates filled it
  mid-run and killed detection (ENOSPC). detect_frames.py now writes chunks
  atomically and resumes; renders use H.264 (~5 MB/min, vs several x for mp4v).
  Check free space before starting; ~1 GB is needed for a half.
- **Page file**: running two full-half renders in parallel (each holds ~2M
  track rows in RAM) with the usual desktop apps pushed Windows' page file
  to 24 GB and took free disk from 5 GB to ~130 MB. Run memory-heavy stages
  one at a time on this machine; a reboot shrinks the page file back. The
  full-half tracking video (~330 MB, ~12 min to render) was deferred for that
  reason - a 10-min sample was rendered instead:
  `render_tracking_video.py ... --start-s 46.04 --end-s 633.36 --output ..._tracking_match00-10min.mp4`.
- **OpenVINO on Intel**: `pip install openvino`; the iGPU ran yolo26n at
  36 ms/frame vs 115 ms CPU torch with identical boxes. The NPU can't compile
  yolo26 (unsupported Floor op on int64). Export dirs must end in
  `_openvino_model` or ultralytics won't recognise them.
- **yolo26 duplicates**: its NMS-free head emits near-identical box pairs for
  ~8% of person boxes; unsuppressed they cut median tracklet length from ~190
  to ~45 frames. `suppress_duplicates()` (IoU 0.6) is mandatory.
- **Kit colour flicker**: distant sky-blue washes towards white at 720p;
  hard thresholds left 22% of tracklets team-mixed. The self-trained colour
  model + flip-cutting brought that to ~5%.
- **Green goalkeeper**: Ederson's all-green kit is masked as grass by the kit
  features (shirt_n = 0). Handled as "unreadable + near own goal". A
  referee in dark green has the same signature - team classes require team
  votes on >= 50% of ALL detections, which keeps him out of the teams.
- **Spare balls**: ~54% of raw ball detections were spare balls on the
  perimeter plinths; dropped by projecting off the pitch (>1 m outside).
- **Reward design**: a per-tracklet (count) reward made the flow chain short
  wrong pieces instead of one long right tracklet; reward per second covered.
- **Distance**: on-camera players skew towards the ball, so "scaled to the
  full half" distance leans high; smoothing (0.5-2 s) changed it < 8%.
- **Clock sync**: the PL scoreboard ticks on a fixed frame phase; its reading
  at the whistle matched Opta's end-of-half second exactly (2824 s).
- **`_common.apply_calibration` was O(frames x rows)** for dynamic
  calibrations (a per-frame loop rescanning every row): fine for 300-frame
  clips, hours for a 70k-frame half (compute_possession stalled). Now a
  chunked per-row einsum - identical output, ~90x faster.
- **Verification design**: score possession at the moment of each event
  (fouls are committed against the team on the ball; corners come from the
  attacking spell) - scoring the dead ball *after* a foul blames the
  tracker for the fouler standing next to the ball. For named players use the
  event team's player on the ball at the last ball sighting before the event
  (the nearest body at a shot is often the blocking defender) and skip
  offsides (the named player is the pass receiver).

## Results of the 2026-11-03 run (first half, 46.9 min tracked)
- Calibration: 70,284/70,284 frames snapped to the lines, 0 lost (spot checks
  every ~5 min sit on the paint). Camera solved at x 52.5 m (halfway line),
  52 m behind the near touchline, 34 m up.
- Identities: 71% of on-pitch person detections identified; outfield
  coverage 78-90% per player (keepers: Lloris 71%, Ederson 13% - the camera
  lived in Spurs' half); kit consistency 0.99; 70/1,793 hand-offs implausible.
- **Limit found: long-term identity.** Links stable under perturbation: City
  70.5%, Spurs 62.2%; role separability 0.82 / 0.69 (< 1 = identities mix);
  same-named-player Opta events land on the same identity 1/5 times. Over
  minutes, chains swap among rotating players - City's front five collapse
  to one average spot in avg_positions_team_a_by_phase.png. Tried and
  rejected: head-region skin/hair tone (at ~30 px the "head" reads as shirt
  colour), lane-based role descriptor, role weights 0 / 0.5 / 2.5 / 4
  (baseline 1.5 was best; the landscape is noisy). **Names are therefore
  role-inferred and indicative; per-player stats are indicative; team-level
  stats (possession, territory, shape by depth, distance totals) don't
  depend on identities.** Next step if per-player identity matters: manual
  anchors (label_roster.py on ~10-20 frames) + an anchor-constrained linker
  (multi-commodity flow per inter-anchor segment, e.g. scipy milp/HiGHS).
- Possession (time held): City 60.7% / Spurs 39.3% (Opta full match 71.2 /
  28.8; Spurs Odyssey: City had 90% in the first 10 min after half-time, so
  a lower first-half share is plausible). 18/23 Opta events agree.
- Field tilt (located-ball time in each attacking third): City 84.1% vs
  Opta final-third entries 82.0% - the cleanest external agreement.
- Distance (tracked time): City 51.2 km, Spurs 54.0 km; scaled to the full
  half 62-64 km - ~10-15% above the typical 52-58 km because a
  ball-following camera over-samples active players.
- City out of possession: 55 phases, 17.8 min (20.0 min compilation).
- Ball tracking: drops at goals (ball in the net / goal mouth not detected,
  a false ball elsewhere wins) - check goal moments in the tracking video.
