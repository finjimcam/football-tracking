# Workflow: Track Players and Visualize Trajectories

## Objective
Run the trained model over match footage to draw per-team bounding boxes and
(optionally) each player's movement trail across the video.

## Required Inputs
- Trained weights from [train_yolo_model.md](train_yolo_model.md), e.g.
  `runs/detect/train/weights/last.pt`.
- Input video (the notebook keeps these under `./Video/`).
- A team-color mapping from class ID → BGR color. Class IDs are whatever
  your Labellerr annotation template used (the notebook's example maps
  `0` and `4`) — check the dataset's `dataset.yaml` `names:` list if unsure.

## Tools
- `tools/track_players.py` — runs `model.track(..., persist=True,
  stream=True)`, draws boxes for classes present in the color mapping, and
  (with `--trajectories`) accumulates and draws each track ID's movement
  line.

## Steps
1. Run tracking only:
   ```
   python tools/track_players.py --weights runs/detect/train/weights/last.pt \
     --video ./Video/2.mp4 --output ./Video/2_tracked.mp4 \
     --team-colors '{"0": [0,0,255], "4": [255,0,0]}'
   ```
2. Add `--trajectories` to also draw movement trails:
   ```
   python tools/track_players.py --weights runs/detect/train/weights/last.pt \
     --video ./Video/4.mp4 --output ./Video/4_output_with_trajectory.mp4 \
     --team-colors '{"0": [0,0,255], "4": [255,0,0]}' --trajectories
   ```
3. Output videos are intermediates, disposable per `CLAUDE.md`'s File
   Structure convention — re-run rather than treating them as precious.

## Edge Cases
- If no boxes are drawn, double-check the class IDs in `--team-colors`
  actually match the trained model's classes — a common failure is reusing
  the notebook's `0`/`4` example on a dataset with different class IDs.
- `TEAM_COLORS` can be set once in `.env` instead of passed on every call.
- Long videos processed frame-by-frame with `stream=True` can take a while;
  there's no progress bar beyond what `ultralytics` prints natively.
- `--trajectories` trails are drawn in pixel space, so on moving-camera
  footage (panning Veo-style units) they mix camera motion with player
  motion and are misleading. Per-frame boxes are still fine. For true
  movement maps on moving footage, use the pitch-space figures from the
  [generate_match_report.md](generate_match_report.md) pipeline
  (`compute_positions.py` after `estimate_camera_motion.py`).
