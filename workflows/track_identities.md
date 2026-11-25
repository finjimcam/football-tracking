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
- Optional: `config/library.json` (issue #16), the library of teams, players,
  competitions, seasons and squads. Fill it with
  `python tools/library.py import-match --match-id <id>` (or *Import from
  match config* in Stopframe's Library tab) once the two files above exist,
  then add the rest of each squad there. The Lines tab's Identify picker
  then offers the whole squad, not only the starting XI, so substitutes can
  be named. `track_identities.py` still takes lineups and role inference
  from `_reference.json`.

## Tools (in order)
| # | Tool | Cost (Core Ultra 5 135U, 47-min half) | Output |
|---|---|---|---|
| 1 | `tools/detect_frames.py` | **~55 min on the iGPU** (OpenVINO, 21 fps) - the only expensive step | `detections.parquet` (+ resumable chunks) |
| 2 | `tools/calibrate_auto.py` | ~55 min CPU, runs *concurrently* with step 1 (it waits for detection chunks) | `frame_homographies.npz`, calibration -> `"dynamic"` |
| 3 | `tools/track_identities.py` | ~10 min (two tracking passes + 8 flow solves); `--reuse-tracklets` re-links from its cache in ~3 min | `tracks.parquet`, `player_map.json`, `team_map.json`, `identity_links.parquet`, `tracklets_cache.parquet` |
| 4 | `tools/track_ball.py` | < 1 min | cleaned ball rows in `tracks.parquet`, `ball_track.parquet` |
| 5 | `tools/compute_possession.py --include-unidentified` | < 1 min | `possession.json`, `possession_timeline.parquet` |
| 5b | `tools/detect_events.py [--instances]` | seconds | `events.parquet`, `events.json` (passes, turnovers, tackles, shots; scored vs Opta); `--instances` also puts them in the Code tab |
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
python tools/track_ball.py --match-id <id>
python tools/compute_possession.py --match-id <id> --include-unidentified
python tools/detect_events.py --match-id <id>            # add --instances to see them in the Code tab
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
  If the log says "self-calibration on 1 searched views" (or few), the global
  search is locking onto false fits - typically steep, zoomed-out views at the
  top of the tilt range. Read the seed view's tilt / focal length and narrow
  the search around them with `--search-tilt MIN,MAX --search-focal MIN,MAX`
  (Korea v Czechia, seed tilt 20.7 deg / f 2866 px: `14,28` and `1400,5000`
  took the hit rate from 1/16 to ~2/3; the misses are midfield views with
  only the halfway line and circle, which is harmless).
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
  (spare balls), links the rest into ball tracklets, drops **static
  clutter**, keeps the strongest per frame, and interpolates gaps <= 12
  frames. Static clutter (issue #7) means still "balls" that keep coming back
  to one spot: at least 30 s in total, across at least 3 different minutes.
  On Man City v Spurs H1 that found 7 spots:
  - the two painted **penalty spots** (94,34) and (12,34);
  - a spare ball by the goal line at (105,58);
  - objects on the far touchline at (80,0) and (78,0).

  Near goal these used to beat the real ball: in 878 frames the pick moved
  to the real ball (6/6 random frames checked by eye). 4409 frames that had
  only clutter now have no ball, which is right. "Ball located" drops from
  48% to 42% of frames, but ball-near-player frames rise from 16447 to
  16914. Real set pieces don't trip the filter: a corner rests seconds, in
  one or two minutes. `--clutter-min-s 0` turns it off.
- **Events** (`detect_events.py`, issue #7, first step): control spells =
  runs of the possession holder, flicker shorter than 5 frames dropped.
  Consecutive spells give pass (same team, ball moved >= 4 m), pass_lost
  (other team, >= 4 m), tackle (other team, < 4 m). A shot is the ball
  leaving a spell <= 35 m from the attacked goal at >= 8 m/s towards it,
  reaching the goal line between ±16 m of the centre or vanishing on the way.
  Man City v Spurs H1 against Opta:
  - shots: 4 of 13 Opta shots and goals found within 5 s;
  - passes: 185 City / 109 Spurs, so 63% City (Opta full match 71%, 755/308
    over 90 min);
  - turnovers: 72/76 (Opta full match 133/104).

  Passes are undercounted because the ball is seen in only ~42% of frames.
  Shots are the weak spot: near goal the ball is in the air or blurred and
  the detector misses it.
- **Fine-tuned ball detector** (issue #7, no hand labels): the cleaned track
  teaches the detector.
  1. `tools/make_ball_dataset.py` writes a person+ball YOLO dataset:
     - person boxes are detections with conf >= 0.4;
     - ball boxes come from `ball_track.parquet` (clutter already dropped),
       plus interpolated balls in gaps of up to 4 frames;
     - training and validation windows are kept apart.
  2. Fine-tune `yolo26n.pt` at imgsz 1280.
  3. Run `detect_frames.py --weights <best.pt>`. It finds the person and
     ball classes by name, so both COCO and fine-tuned weights work.

  Man City v Spurs, trained on minutes 0-32 of H1 (1695 frames), 8 epochs,
  batch 4 (3.6 GB VRAM on the GTX 1060, 41 min). Evaluated on the held-out
  minutes 35-47, same identities, with `track_ball.py` and
  `compute_possession.py` re-run:

  | | yolo26n | fine-tuned |
  |---|---|---|
  | raw ball detections | 40958 | 19946 (less junk) |
  | ball located (detected) | 44.6% | 48.1% |
  | ball-near-player frames | 4658 | **5441** (+17%) |
  | City possession (Opta 71% full match) | 58% | **65%** |
  | City pass share (Opta 71%) | 58% | 63% |

  Spot-check of picks where the two disagree: the fine-tuned picks were the
  ball (5/5 zoomed). The base picks were grass, a coach's hand, and a ball
  on the touchline. Shots near goal are still missed by both.

  The weights are not in the repo (`*.pt` is gitignored). On pphub they are
  at `runs/ballft/v1/weights/best.pt`. Commands:
  ```
  python tools/make_ball_dataset.py --match-id <id> --out <dir> --train 1151:52000 --val 53000:71434
  python tools/train_yolo.py --data <dir>/dataset.yaml --model yolo26n.pt --epochs 8 --imgsz 1280 --batch 4
  python tools/detect_frames.py --match-id <id> --weights runs/detect/train/weights/best.pt
  ```
  On pphub, batch 8 at 1280 needs 5.9 GB of VRAM, which leaves Kodi nothing.
  Use batch 4, and stop training if Jellyfin/Kodi starts playing (it resumes
  from `last.pt`).
- **Manual anchors** (issue #16): `track_identities.py` applies
  `config/<id>_identity_anchors.json` (made in the Lines tab's Identify
  mode) after gap-filling. An anchor is a frame plus a detection box plus a
  shirt number, and is re-found by box IoU, so it survives a tracklet
  rebuild. The run then does two things:
  - Each chain is named after the player most of its anchors point at; the
    chain that had that name takes the displaced name.
  - Anchored tracklets are moved into their player's chain, and that
    chain's tracklets overlapping them in time are dropped from it.

  `player_map.json` marks those names `name_source: manual (Lines tab
  anchors)` and reports `manual_anchors` (unresolved anchors and
  conflicting votes included). Re-run steps 4-10 afterwards.
  Check the result in Stopframe's **Players** tab (`tools/player_paths.py`,
  which also prints the same stats table from the CLI). It shows each player's
  whole-match path and heatmap. A full-back whose heatmap covers the whole
  pitch, or a chain with many `fragile_links`, is where anchors help most.
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
- **Gap status** (`gap_status.py`, issue #47): every identity row in
  `tracks.parquet` has a `status`. `detected` is a real box.
  `interpolated` fills a gap of <= 3 s (`--fill-max-s`) that the player
  could have run (distance <= 3 m + 8.5 m/s x gap) with the RTS smoother
  (see "Short-gap fill"). `absorbed` is a duel gap of <= 10 s
  (`--absorbed-max-s`): the player was last seen within 1.5 m of another
  player and came back within 2.5 m of that same player. It is filled by the
  duel filler (see "Duel gaps"), whose kit and swap checks can refuse it.
  `unknown` means the gap is too long, not runnable, refused by its filler
  (reason `...+fill_refused`), or before the first or after the last
  sighting. `--linear-fill` restores the old straight line for both kinds. Unknown frames have no row. `gaps.parquet` has one
  row per gap (kind, reason, partner track). Fill rows have no box (x1..y2
  NaN, conf 0); their foot pixels are projected back through the frame's
  homography. `track_identities.py` labels as it writes tracks.parquet
  (`--no-gap-status` turns it off). `python tools/gap_status.py --match-id
  <id>` re-labels in place with the same fillers, so you can tune
  `--max-fill-s`, `--max-absorbed-s` and `--duel-m` without re-linking. Heatmaps and average positions
  (`compute_positions.py`) weight detected 1, interpolated 0.5, absorbed 0.5
  (`--w-interpolated` / `--w-absorbed`). `compute_physical.py` counts filled
  steps towards distance and time (`distance_filled`), but speed peaks only
  come from detected-to-detected steps. Possession, contact swaps,
  verification, renders and the Lines chains read detected rows only
  (`gap_status.observed()`). Fillers plug in through
  `label_tracks(..., classify=, fillers={kind: fn}, context_hook=)`;
  `pipeline_fillers()` is the set the pipeline uses.
- **Duel gaps** (`duel_gaps.py`, issue #47 item 3): a player who vanishes
  within 1.5 m of another tracked person, and comes back within 2.5 m of
  that same person, was hidden behind them in a merged box. The gap is
  filled with the partner's path plus an offset blended from the entry
  offset to the exit offset. The partner must be one tracklet through the
  gap (a partner chain re-linked mid-gap once jumped onto the referee). At
  the split the fill is checked: kit colour while the two stand > 1.2 m
  apart (opponents can't swap teams), and for teammates exit motion and
  side. A failed check leaves the gap `unknown`. In the pipeline this runs
  as gap_status's `absorbed` filler (`duel_gaps.fill`, with the full checks
  via `ctx.duel_index`). `python tools/duel_gaps.py --match-id <id>` is
  the standalone experiment tool: it writes its own rows (`interp=True`,
  `absorbed_by`) and a `duel_gaps.json` review list; don't run it on top of
  a gap-status tracks.parquet. On the 2026-11-17 reference
  run: 2,023 duel gaps, 1,536 filled (6,488 frames), 69 flagged, 418 with no
  usable partner. Most duel gaps are under 1 s, and only 40 last 3-10 s
  (4 filled). In longer duel gaps the player usually comes back far from
  the partner. Spot check: 9 of 10 fills on the right player (two about
  1 m off inside the merged box). All 3 flagged fills were real problems.
  `fill(gap, ctx)` is the plug-in for the gap-status labeller (status
  `absorbed`).
- **Duel-split linking** (issue #48, on by default; `--duel-sigma-m 0` = off): when a
  tracklet ends within `--duel-m` 1.5 m of an opponent (a merged box), the
  linker prices successors by their distance from that partner's position
  (`duel_partners()`), and allows `--duel-reach-m` +2 m and
  `--duel-overlap-frames` 10 for those links. Merged boxes make the foot
  point jump and briefly double-detect, so the hidden player's next piece was
  often blocked by the 3 m reach gate or the 3-frame overlap limit. It is
  skipped if a teammate is within `--duel-crowd-m` 2 m. On the 2026-11-17
  reference half (exp48): City+Spurs tracklets in chains 1397 -> 1418,
  unidentified tracks 794 -> 768, detected coverage 83.5% -> 84.1%, missing
  frames in gaps > 10 s 172k -> 162k. 5 of 6 new links checked on video were
  right; one teammate merge was ambiguous. But Spurs suspected swaps rose
  14/90 -> 19/94. Some of that rise is newly evaluable contacts or shifted
  team-shape averages, not wrong links; the proxy can't separate them. So the
  92 links where the two runs disagree (50 events) were judged on video
  instead: 24 sampled events, the new link right in 11, the old one in 3, and
  10 undecidable from frames. The wins are mostly a white+blue merged box,
  where the new link takes the player who comes out of it and the old one
  jumped to a fragment 10-15 s later. The losses drop a correct immediate
  continuation. Validate future linker changes the same way: render only
  the disagreements and judge them (`/srv/football-data/exp48/scripts/
  spot_dis.py`), rather than tune against the swap proxy.
  Diagnosis: of 222 duel gaps >= 1 s, 84 come back to the right identity, 85
  have no fragment near the partner, and 53 have a same-kit fragment near the
  partner that went unclaimed or to a teammate. About half of those are extra
  bodies, because the identity is seen elsewhere at the same time.
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
- **Short-gap fill** (`interp_gaps.py`, issue #47 item 4): gap_status's
  `interpolated` filler (`make_filler`), up to `--fill-max-s` (default 3 s).
  Each gap is filled with a constant-velocity Kalman
  filter and RTS smoother run over the player's own sightings 1 s either side.
  The curve is shifted so it joins both sightings exactly. The fill keeps the
  player's velocity through the gap, where a straight line has a kink at each
  end. Peak speed is capped at 10 m/s and acceleration at 8 m/s²: if the curve
  breaks either cap, the straight line is used instead. If even the straight
  line needs more than 10 m/s, the gap stays empty, because that is a tracking
  jump rather than a hidden player (the filler refuses it: `unknown`).
  `python tools/interp_gaps.py --match-id <id>` is the standalone
  experiment tool (own `interp=True` rows; not on top of a gap-status
  tracks.parquet), and `--holdout` prints the error table.
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

## Fixing tracking by hand (Stopframe Tracking tab, issue #42)
Use this when the automatic identities are wrong in places you care about. The
tab lists every possible swap (`contact_swaps.json`) and every loss of 3 s or
more in view. Step through them with n/p. Each fix is one line in
`config/<id>_track_edits.json`:
- `swap`: two players trade names from a frame to the end of the half
  (the *Swap labels* tool, `s`: click two boxes).
- `assign`: "this box is X". On a labelled box it is a swap (the whole track
  follows, not just the clicked tracklet). On an unlabelled fragment, that
  tracklet becomes X and every tracklet X had overlapping it is unlabelled in
  full. (Until 2026-11-24 it traded only for the clicked tracklet's life, which
  was 8 frames on Korea v Czechia's kick-off frame, so relabels cascaded.)
- `lock` (`l`): every name on this frame is right.
- `box`: a hand-drawn box, interpolated up to 5 s to the player's nearest
  boxes.
- `split`, `delete`, `not_player`.
- `ok`: the problem was checked and is fine.

The fixes are applied in order on top of `tracks.parquet` by
`tools/track_edits.py` whenever a tool loads tracks. That covers
`_common.load_tracks`, the Lines picker, phase clips, contact swaps and the
reference comparison. Tools that rewrite tracks (`track_ball.py`) and the
tracker's own checks (`verify_tracking.py`) still see the raw output, so the
fixes are never baked in. After fixing, re-run possession → positions /
physical → phase clips (the match page marks them stale).
- The file stores a fingerprint of the player rows it was made against. If
  `track_identities.py` is re-run and the ids move, the fixes are **not
  applied**. Both the CLI (`python tools/track_edits.py --match-id <id>`) and
  the tab warn about it. Check them again rather than trusting old ids. A
  re-run with `--reuse-tracklets` and the same settings keeps the
  fingerprint.
- A Lines-tab *Fix swap* and a Tracking-tab swap at the same collision undo
  each other. Fix identities in the Tracking tab first, then pick lines.

### Labels carry through the half: pins and Re-link
A swap, a rename or a lock also writes an identity anchor for each named
(detected, not hand-drawn) box on that frame into
`config/<id>_identity_anchors.json`, tagged `"source": "tracking", "edit": <id>`
(Undo removes them). The boxes show a ✓. The fix shows at once from that frame
on, but the linker only uses the pins when **Re-link whole half** runs (the
calibration-run chain: identities, ball, possession, physical, positions,
verify). With pins the linker solves one chain per player through every
pinned tracklet, so labelling the lineup on the first frame and pressing Lock,
then Re-link, rebuilds each whole track from it.
- After the re-run `track_identities.py` rebases the edits file (old copy kept
  as `<id>_track_edits.before-rerun-<time>.json`): naming edits are retired
  (they're in the tracking now), hand boxes / deletes / splits follow their
  player (stored as team + number + name) to his new id, and not-player edits
  follow their tracklet.
- Pins need shirt numbers: fill them into `config/<id>_reference.json`
  lineups. Without a number a player can't be pinned (the tab says so).

### Start the half at the kick-off, even on a wide shot
The kick-off is the best frame to name everyone (lined up in their own half),
so don't start the window after it because the camera is zoomed out. On
Korea v Czechia (frames 20-269, players ~30 px):
- detect those frames at `--imgsz 1920` (1280 found ~14 of 25 people on the
  pitch, 1920 found ~21);
- `calibrate_auto.py` locked onto the stands there; chaining frame-to-frame
  image homographies (SIFT, players masked, RANSAC) back from the first
  calibrated tactical frame worked (fixed camera: pan/tilt/zoom only);
- `track_identities.py` treats frames whose median box is < 0.65 x the
  match's as wide view and starts tracklets from conf 0.15 there (else 0.35).

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
- **Which yolo26 head (issue #47, 2026-11-17)**: the cached half detections
  came from the NMS-free one-to-one head (OpenVINO export). On pphub
  (CUDA, ultralytics 8.4) `detect_frames.py` with default flags runs the
  one-to-many head + NMS at IoU 0.7 instead. On a 4-min duel-heavy window
  (frames 65250-71249) that gave 258 instead of 178 player track ids and a
  median team tracklet of 121 instead of 193 frames, at the same coverage
  (86%). `--nms-free` is therefore the default since #47 (`--no-nms-free`
  for the old NMS head). Loosening NMS to
  keep the hidden player in a duel did not help: `--iou 0.8/0.85` alone is
  undone by the duplicate filter; with `track_identities.py --dup-iou` raised
  to match, the duplicates triple to five-fold the tracklets, coverage
  falls to 72% / 68% and suspected swaps rise from 2 to 8 / 17. Soft-NMS
  (sigma 0.5) and `--imgsz 1536` didn't help player coverage either, though
  1536 found the ball (conf >= 0.25) in 63% of frames instead of 54%.
  Kit-aware duplicate suppression (keep overlapping opposite kits) rescued
  < 0.01 boxes per frame: the hidden player's box reads the front player's
  shirt. Treat the duel case downstream (gap filling), not in the detector.
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
