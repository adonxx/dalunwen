"""Evaluate Drone-YOLO checkpoints with COCO small/medium/large area bins.

Ultralytics reports class-wise metrics but does not expose COCO area-stratified
AP for a generic YOLO-format dataset. This script therefore:

1. runs the same validation protocol with ``save_json=True``;
2. converts the read-only YOLO validation labels to COCO ground truth in the
   original image coordinate system; and
3. evaluates the saved predictions with pycocotools for all/small/medium/large
   objects, including per-class area metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ultralytics
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import STAGES, build_model  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
AREA_LABELS = ("all", "small", "medium", "large")
AREA_DEFINITION = {
    "small": "area < 32^2 pixels",
    "medium": "32^2 <= area < 96^2 pixels",
    "large": "area >= 96^2 pixels",
}


def sha256(path: Path) -> str:
    """Return a checkpoint hash for provenance tracking."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_id(path: Path) -> int | str:
    """Match the image-id convention used by Ultralytics JSON export."""
    return int(path.stem) if path.stem.isnumeric() else path.stem


def area_bucket(area: float) -> str:
    """Assign an original-image box area to a COCO-compatible size bucket."""
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def resolve_split_dirs(data_config: dict[str, Any], split: str) -> tuple[Path, Path]:
    """Resolve the image and sibling label directories from a YOLO data file."""
    root = Path(data_config["path"])
    image_dir = root / Path(data_config[split])
    label_dir = image_dir.parent / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"missing dataset split directories: {image_dir}, {label_dir}")
    return image_dir, label_dir


def build_coco_ground_truth(
    data_config: dict[str, Any], split: str, output_path: Path
) -> tuple[dict[str, int], dict[str, int]]:
    """Convert YOLO labels to a deterministic COCO ground-truth JSON file."""
    image_dir, label_dir = resolve_split_dirs(data_config, split)
    names = {int(index): str(name) for index, name in data_config["names"].items()}
    images = []
    annotations = []
    area_counts = {label: 0 for label in AREA_LABELS}
    annotation_id = 1

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for path in image_paths:
        with Image.open(path) as image:
            width, height = image.size
        current_image_id = image_id(path)
        images.append(
            {
                "id": current_image_id,
                "file_name": path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{path.stem}.txt"
        if not label_path.exists():
            continue
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            fields = raw_line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid YOLO label at {label_path}:{line_number}: {raw_line}")
            class_id, x_center, y_center, box_width, box_height = map(float, fields)
            class_id = int(class_id)
            if class_id not in names:
                raise ValueError(f"unknown class {class_id} at {label_path}:{line_number}")

            width_px = max(0.0, box_width * width)
            height_px = max(0.0, box_height * height)
            x = max(0.0, x_center * width - width_px / 2)
            y = max(0.0, y_center * height - height_px / 2)
            width_px = min(width_px, width - x)
            height_px = min(height_px, height - y)
            area = width_px * height_px
            if area <= 0:
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": current_image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, width_px, height_px],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            area_counts["all"] += 1
            area_counts[area_bucket(area)] += 1

    dataset = {
        "info": {
            "description": f"VisDrone2019 {split} converted from YOLO labels for area evaluation",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id + 1, "name": class_name, "supercategory": "object"}
            for class_id, class_name in sorted(names.items())
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations)}, area_counts


def mean_valid(values: np.ndarray) -> float | None:
    """Average valid COCO precision/recall entries, excluding sentinel -1."""
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def metric_slice(
    evaluator: COCOeval,
    area_index: int,
    category_index: int | slice = slice(None),
) -> dict[str, float | None]:
    """Extract AP/AP50/AP75/AR for one area and optional category."""
    precision = evaluator.eval["precision"][:, :, category_index, area_index, -1]
    recall = evaluator.eval["recall"][:, category_index, area_index, -1]
    iou_thresholds = evaluator.params.iouThrs

    def at_iou(target: float) -> float | None:
        indices = np.flatnonzero(np.isclose(iou_thresholds, target))
        return mean_valid(precision[indices]) if indices.size else None

    return {
        "ap": mean_valid(precision),
        "ap50": at_iou(0.50),
        "ap75": at_iou(0.75),
        "ar_max_det": mean_valid(recall),
    }


def evaluate_coco_sizes(
    ground_truth_json: Path,
    predictions_json: Path,
    max_det: int,
) -> tuple[dict[str, dict[str, float | None]], list[dict[str, Any]]]:
    """Run COCO evaluation and return overall and per-class area metrics."""
    coco_gt = COCO(str(ground_truth_json))
    coco_dt = coco_gt.loadRes(str(predictions_json))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()

    area_indices = {label: index for index, label in enumerate(evaluator.params.areaRngLbl)}
    overall = {
        label: metric_slice(evaluator, area_indices[label])
        for label in AREA_LABELS
    }
    print(f"COCO area metrics with max_det={max_det}:")
    for label in AREA_LABELS:
        metrics = overall[label]
        print(
            f"  {label:>6}: AP={metrics['ap']:.4f}, AP50={metrics['ap50']:.4f}, "
            f"AP75={metrics['ap75']:.4f}, AR={metrics['ar_max_det']:.4f}"
        )

    category_ids = evaluator.params.catIds
    category_names = {category["id"]: category["name"] for category in coco_gt.dataset["categories"]}
    per_class = []
    for category_index, category_id in enumerate(category_ids):
        for label in AREA_LABELS:
            per_class.append(
                {
                    "class_id": int(category_id) - 1,
                    "class_name": category_names[category_id],
                    "area": label,
                    **metric_slice(evaluator, area_indices[label], category_index),
                }
            )
    return overall, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="size_evaluation")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--ground-truth-json", type=Path)
    parser.add_argument(
        "--predictions-json",
        type=Path,
        help="Reuse an existing Ultralytics predictions.json instead of running inference.",
    )
    args = parser.parse_args()

    data_path = (ROOT / "configs" / "data" / "visdrone2019.yaml").resolve()
    with data_path.open("r", encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)

    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_json = (
        args.ground_truth_json.resolve()
        if args.ground_truth_json
        else output_json.parent / f"visdrone_{args.split}_coco_ground_truth.json"
    )
    dataset_count, area_counts = build_coco_ground_truth(data_config, args.split, ground_truth_json)

    validation_results = None
    validation_save_dir = None
    if args.predictions_json:
        predictions_json = args.predictions_json.resolve()
    else:
        model = build_model(args.stage, weights=str(args.weights.resolve()))
        model.model.names = data_config["names"]
        validation_results = model.val(
            data=str(data_path),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            half=args.half,
            plots=False,
            save_json=True,
            project=str(args.project.resolve()),
            name=args.name,
            exist_ok=False,
        )
        validation_save_dir = Path(validation_results.save_dir).resolve()
        predictions_json = validation_save_dir / "predictions.json"

    if not predictions_json.is_file():
        raise FileNotFoundError(f"missing predictions JSON: {predictions_json}")
    overall, per_class = evaluate_coco_sizes(ground_truth_json, predictions_json, args.max_det)

    checkpoint = args.weights.resolve()
    record = {
        "schema_version": 1,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": args.stage,
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "protocol": {
            "dataset": "VisDrone2019",
            "data_yaml": str(data_path),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "half": args.half,
            "area_coordinate_system": "original image pixels",
            "area_definition": AREA_DEFINITION,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "pycocotools": __import__("pycocotools").__version__
            if hasattr(__import__("pycocotools"), "__version__")
            else "installed",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "dataset_count": dataset_count,
        "ground_truth_area_counts": area_counts,
        "overall_by_area": overall,
        "per_class_by_area": per_class,
        "ultralytics_overall": (
            {key: float(value) for key, value in validation_results.results_dict.items()}
            if validation_results is not None
            else None
        ),
        "ground_truth_json": str(ground_truth_json),
        "predictions_json": str(predictions_json),
        "validation_save_dir": str(validation_save_dir) if validation_save_dir else None,
    }
    output_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Size-stratified metrics saved to {output_json}")


if __name__ == "__main__":
    main()
