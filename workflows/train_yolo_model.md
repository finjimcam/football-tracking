# Workflow: Train the YOLO Model

## Objective
Fine-tune a pre-trained YOLO model on the prepared football dataset.

## Required Inputs
- `yolo_format/dataset.yaml` from
  [prepare_training_data.md](prepare_training_data.md).
- A base checkpoint to fine-tune from (the notebook defaults to
  `yolo11x.pt`; ultralytics will download it automatically if not present).
- A GPU-backed environment — this is a real training run, not something to
  attempt on CPU.

## Tools
- `tools/train_yolo.py` — wraps the `ultralytics` CLI
  (`yolo task=detect mode=train ...`).

## Steps
1. Sanity-check the command before committing GPU time:
   ```
   python tools/train_yolo.py --data yolo_format/dataset.yaml --dry-run
   ```
2. Kick off training:
   ```
   python tools/train_yolo.py --data yolo_format/dataset.yaml \
     --model yolo11x.pt --epochs 200 --imgsz 640 --batch 20
   ```
3. Weights land under `runs/detect/train*/weights/` (`last.pt`, `best.pt`).
   These are the inputs to
   [track_and_visualize.md](track_and_visualize.md) and
   [generate_heatmap.md](generate_heatmap.md).

## Edge Cases
- `runs/` and downloaded `.pt` base weights are large, regenerable
  intermediates — don't treat them as deliverables to hand off; if a run
  needs to be preserved, move the specific weights file somewhere durable.
- Lower `--batch` if you hit CUDA out-of-memory errors; there's no
  auto-retry in the wrapper, it's a direct pass-through to `ultralytics`.
- Training is a paid-GPU-time operation on rented hardware — check with the
  user before re-running full training sweeps, per the "paid API calls or
  credits" guidance in `CLAUDE.md`.
