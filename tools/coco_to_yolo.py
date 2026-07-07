"""Convert COCO-format annotations into the YOLO dataset layout ultralytics expects.

Wraps yolo_finetune_utils.coco_yolo_converter.bbox_converter.coco_to_yolo_converter.
That package is an external repo (not vendored here) — clone it next to this
project before running:
    git clone https://github.com/Labellerr/yolo_finetune_utils.git

Usage:
    python tools/coco_to_yolo.py --json-path dataset_annotation.json \
        --images-dir ./dataset --output-dir yolo_format
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-path", required=True, help="Path to the COCO annotations JSON file")
    parser.add_argument("--images-dir", required=True, help="Path to the directory containing the images")
    parser.add_argument("--output-dir", default="yolo_format", help="Where to write the YOLO-format dataset")
    parser.add_argument("--use-split", action="store_true", help="Let the converter create train/val splits")
    args = parser.parse_args()

    try:
        from yolo_finetune_utils.coco_yolo_converter.bbox_converter import coco_to_yolo_converter
    except ImportError:
        sys.exit(
            "yolo_finetune_utils is not installed.\n"
            "Clone it and make sure it's on PYTHONPATH:\n"
            "  git clone https://github.com/Labellerr/yolo_finetune_utils.git"
        )

    result = coco_to_yolo_converter(
        json_path=args.json_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        use_split=args.use_split,
    )
    print(result)


if __name__ == "__main__":
    main()
