# Workflow: Generate a Positional Heatmap

## Objective
Aggregate player positions across an entire video into a per-class heatmap,
overlaid on a sample frame, to see team positioning/field control at a glance.

## Required Inputs
- Trained weights, e.g. `runs/detect/train/weights/last.pt`.
- Input video.
- The list of class IDs to include (must match the trained model's classes).

## Tools
- `tools/generate_heatmap.py` — tracks the full video, accumulates bounding
  box occupancy per class into a float array the size of the frame,
  normalizes, applies `COLORMAP_JET`, and overlays it on a chosen sample frame.

## Steps
1. Run:
   ```
   python tools/generate_heatmap.py --weights runs/detect/train/weights/last.pt \
     --video ./Video/4.mp4 --output .tmp/class_heatmaps_overlay.png \
     --classes "[0, 4]" --sample-frame 100
   ```
2. Inspect the PNG. If the background frame looks wrong (e.g. mid-replay,
   scoreboard overlay), adjust `--sample-frame` to a cleaner frame index.

## Edge Cases
- `--sample-frame` must exist in the video (`cv2` will fail to read past the
  end) — pick something well within the clip's frame count.
- This processes the entire video before producing output; for quick
  iteration, test against a short clip first.
- The output PNG is a disposable intermediate unless the user asks for it as
  a deliverable, per `CLAUDE.md`'s File Structure convention — in that case
  push it to wherever they're consolidating deliverables (e.g. Slides/Sheets),
  don't just leave it in `.tmp/`.
