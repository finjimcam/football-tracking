"""Fine-tune a YOLO model on the converted dataset via the ultralytics CLI.

Thin, deterministic wrapper around:
    yolo task=detect mode=train data=<dataset.yaml> model=<base.pt> \
        epochs=<n> imgsz=<n> batch=<n>

Usage:
    python tools/train_yolo.py --data yolo_format/dataset.yaml --model yolo11x.pt \
        --epochs 200 --imgsz 640 --batch 20
"""
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to dataset.yaml produced by tools/coco_to_yolo.py")
    parser.add_argument("--model", default="yolo11x.pt", help="Base model to fine-tune from")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Print the command instead of running it")
    args = parser.parse_args()

    cmd = [
        "yolo",
        "task=detect",
        "mode=train",
        f"data={args.data}",
        f"model={args.model}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
    ]

    print("Running:", " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
