# Workflow: Prepare Training Data (COCO → YOLO)

## Objective
Convert the COCO-format export from Labellerr into the YOLO dataset layout
(`images/`, `labels/`, `dataset.yaml`) that `ultralytics` training expects.

## Required Inputs
- COCO annotations JSON (from
  [annotate_and_export_dataset.md](annotate_and_export_dataset.md)),
  e.g. `.tmp/dataset_annotation.json`.
- The matching images directory.
- A local clone of `yolo_finetune_utils`, since the converter lives there and
  isn't vendored in this repo:
  ```
  git clone https://github.com/Labellerr/yolo_finetune_utils.git
  ```

## Tools
- `tools/coco_to_yolo.py` — thin CLI wrapper around
  `yolo_finetune_utils.coco_yolo_converter.bbox_converter.coco_to_yolo_converter`.

## Steps
1. Confirm `yolo_finetune_utils` is cloned and importable (on `PYTHONPATH`
   or in the same parent directory you run the script from).
2. Run the conversion:
   ```
   python tools/coco_to_yolo.py \
     --json-path .tmp/dataset_annotation.json \
     --images-dir .tmp/dataset \
     --output-dir yolo_format
   ```
3. Verify `yolo_format/dataset.yaml` and the `images/`/`labels/` subfolders
   were created before moving on to
   [train_yolo_model.md](train_yolo_model.md).

## Edge Cases
- If the script exits with an `ImportError` about `yolo_finetune_utils`, it
  means the external repo isn't cloned/importable yet — clone it and retry.
- `yolo_format/` is a regenerable intermediate (per the File Structure
  convention in `CLAUDE.md`) — safe to delete and re-run this step.
- Pass `--use-split` if you want the converter to create train/val splits
  itself instead of using the whole set for training.
