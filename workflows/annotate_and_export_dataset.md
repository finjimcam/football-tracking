# Workflow: Annotate and Export a Dataset

## Objective
Get raw football footage/images labeled on Labellerr and pull the accepted
annotations down locally as COCO JSON, ready for YOLO conversion.

## Required Inputs
- A local folder of images (or video frames) to annotate.
- Labellerr workspace credentials in `.env`: `LABELLERR_API_KEY`,
  `LABELLERR_API_SECRET`, `LABELLERR_CLIENT_ID` (from the workspace's
  API keys page, e.g. `https://<workspace>.labellerr.com/workspace/api-keys`).
- Names to use for the dataset, project, and annotation template.

## Tools
- `tools/labellerr_annotate.py setup` — uploads the folder as a dataset,
  creates a polygon annotation template, waits for dataset processing, and
  creates the annotation project.
- `tools/labellerr_annotate.py export` — exports annotations in a given set
  of statuses as COCO JSON and prints a temporary download URL.

## Steps
1. Run `setup`:
   ```
   python tools/labellerr_annotate.py setup \
     --folder <path-to-images> \
     --dataset-name "<name>" --project-name "<name>" --template-name "<name>" \
     --question Object --data-type image
   ```
   Note the printed project ID.
2. Go to labellerr.com, open the project, and label the files (bounding
   box / polygon per the template's question).
3. Once annotations are accepted, export them:
   ```
   python tools/labellerr_annotate.py export --project-id <id> \
     --export-name "Weekly Export" --export-format coco_json --statuses accepted
   ```
4. Download the JSON from the printed URL into `.tmp/` (e.g.
   `.tmp/dataset_annotation.json`) — this is the input to
   [prepare_training_data.md](prepare_training_data.md).

## Edge Cases
- `setup` will fail fast if any of the three Labellerr env vars are missing —
  check `.env` first.
- Annotations not yet in `accepted` status won't show up in the export;
  widen `--statuses` (e.g. `review r_assigned client_review cr_assigned
  accepted`) if you need in-progress work too.
- The dataset must finish processing before project creation — `setup`
  already waits on this (`dataset.status()`), so a hang here means the
  Labellerr-side processing is stuck, not a bug in the script.
