"""Evaluate whole-image and slicing-aided inference on VisDrone2019.

The script keeps one trained checkpoint fixed and changes only the inference
pipeline. Sliced predictions are shifted back to original-image coordinates,
merged with class-aware NMS, capped at the same per-image ``max_det`` as the
whole-image control, and evaluated with the existing COCO area protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import ultralytics
import yaml
from torchvision.ops import batched_nms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import build_model  # noqa: E402
from evaluate_object_sizes import (  # noqa: E402
    AREA_DEFINITION,
    IMAGE_SUFFIXES,
    build_coco_ground_truth,
    evaluate_coco_sizes,
    image_id,
    resolve_split_dirs,
)


def sha256(path: Path) -> str:
    """Return a checkpoint hash for provenance tracking."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_origins(length: int, window: int, overlap: float) -> list[int]:
    """Return deterministic window origins that cover one image axis."""
    if length <= 0 or window <= 0:
        raise ValueError("length and window must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if length <= window:
        return [0]
    step = max(1, int(round(window * (1.0 - overlap))))
    origins = list(range(0, length - window + 1, step))
    final_origin = length - window
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def slice_windows(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return ``(x1, y1, x2, y2)`` windows covering an image."""
    x_origins = axis_origins(width, size, overlap)
    y_origins = axis_origins(height, size, overlap)
    return [
        (x, y, min(x + size, width), min(y + size, height))
        for y in y_origins
        for x in x_origins
    ]


def merge_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
    max_det: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply class-aware global NMS and a common per-image detection cap."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4), scores.reshape(0), classes.reshape(0)
    keep = batched_nms(boxes, scores, classes, iou_threshold)
    keep = keep[:max_det]
    return boxes[keep], scores[keep], classes[keep]


def detections_to_coco(
    path: Path,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
) -> list[dict[str, Any]]:
    """Convert original-coordinate ``xyxy`` predictions to COCO records."""
    boxes = boxes.detach().cpu().float()
    scores = scores.detach().cpu().float()
    classes = classes.detach().cpu().long()
    records = []
    current_image_id = image_id(path)
    for box, score, class_id in zip(boxes, scores, classes, strict=True):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            continue
        records.append(
            {
                "image_id": current_image_id,
                "category_id": int(class_id) + 1,
                "bbox": [x1, y1, width, height],
                "score": float(score),
            }
        )
    return records


def synchronize(device: str) -> None:
    """Synchronize CUDA timings when GPU inference is active."""
    if torch.cuda.is_available() and str(device).lower() not in {"cpu", "mps"}:
        torch.cuda.synchronize()


def prediction_tensors(result: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract box, score, and class tensors from one Ultralytics result."""
    if result.boxes is None or len(result.boxes) == 0:
        device = result.orig_img.device if torch.is_tensor(result.orig_img) else torch.device("cpu")
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0,), device=device, dtype=torch.long),
        )
    return result.boxes.xyxy, result.boxes.conf, result.boxes.cls.long()


def predict_whole(model: Any, image: np.ndarray, args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    """Run the whole-image control once."""
    result = model.predict(
        source=image,
        imgsz=args.input_size,
        rect=False,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )[0]
    return prediction_tensors(result)


def predict_sliced(
    model: Any,
    image: np.ndarray,
    windows: list[tuple[int, int, int, int]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run batched slices and shift every prediction to original coordinates."""
    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_classes: list[torch.Tensor] = []
    for start in range(0, len(windows), args.batch):
        current_windows = windows[start : start + args.batch]
        crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in current_windows]
        results = model.predict(
            source=crops,
            imgsz=args.input_size,
            rect=False,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
        for result, (x1, y1, _, _) in zip(results, current_windows, strict=True):
            boxes, scores, classes = prediction_tensors(result)
            if boxes.numel() == 0:
                continue
            offset = boxes.new_tensor([x1, y1, x1, y1])
            all_boxes.append(boxes + offset)
            all_scores.append(scores)
            all_classes.append(classes)

    if not all_boxes:
        empty_boxes = torch.empty((0, 4))
        return empty_boxes, torch.empty((0,)), torch.empty((0,), dtype=torch.long), 0
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    classes = torch.cat(all_classes)
    raw_count = int(scores.numel())
    boxes, scores, classes = merge_detections(
        boxes,
        scores,
        classes,
        iou_threshold=args.merge_iou,
        max_det=args.max_det,
    )
    return boxes, scores, classes, raw_count


def percentile(values: list[float], q: float) -> float:
    """Return one numeric percentile for JSON serialization."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("whole", "sliced"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--ground-truth-json", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "whole" and args.input_size != 640:
        raise ValueError("the pre-registered whole-image control uses input-size=640")
    if args.mode == "sliced" and args.slice_size != args.input_size:
        raise ValueError("slice-size and input-size must match in this experiment")
    if args.half:
        raise ValueError("the pre-registered experiment is FP32; --half must remain disabled")

    data_path = (ROOT / "configs" / "data" / "visdrone2019.yaml").resolve()
    with data_path.open("r", encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)
    image_dir, _ = resolve_split_dirs(data_config, "val")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    weights = args.weights.resolve()
    model = build_model("full", weights=str(weights))
    model.model.names = data_config["names"]

    warmup_side = args.input_size
    warmup = np.zeros((warmup_side, warmup_side, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(
            source=warmup,
            imgsz=args.input_size,
            rect=False,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
    synchronize(args.device)
    if torch.cuda.is_available() and str(args.device).lower() not in {"cpu", "mps"}:
        torch.cuda.reset_peak_memory_stats()

    predictions: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for index, path in enumerate(image_paths, start=1):
        image_started = time.perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        decoded = time.perf_counter()
        height, width = image.shape[:2]
        windows = (
            [(0, 0, width, height)]
            if args.mode == "whole"
            else slice_windows(width, height, args.slice_size, args.overlap)
        )
        prepared = time.perf_counter()

        synchronize(args.device)
        inference_started = time.perf_counter()
        if args.mode == "whole":
            boxes, scores, classes = predict_whole(model, image, args)
            raw_count = int(scores.numel())
        else:
            boxes, scores, classes, raw_count = predict_sliced(model, image, windows, args)
        synchronize(args.device)
        inference_finished = time.perf_counter()

        current_predictions = detections_to_coco(path, boxes, scores, classes)
        predictions.extend(current_predictions)
        image_finished = time.perf_counter()
        merged_count = len(current_predictions)
        per_image.append(
            {
                "image_id": image_id(path),
                "file_name": path.name,
                "width": width,
                "height": height,
                "tiles": len(windows),
                "raw_detections": raw_count,
                "merged_detections": merged_count,
                "decode_ms": (decoded - image_started) * 1000.0,
                "slice_ms": (prepared - decoded) * 1000.0,
                "inference_and_merge_ms": (inference_finished - inference_started) * 1000.0,
                "serialization_ms": (image_finished - inference_finished) * 1000.0,
                "end_to_end_ms": (image_finished - image_started) * 1000.0,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(image_paths):
            print(f"[{index:>3}/{len(image_paths)}] {path.name}: {len(windows)} tiles, {merged_count} kept")

    total_elapsed = time.perf_counter() - total_started
    predictions_json = args.predictions_json.resolve()
    predictions_json.parent.mkdir(parents=True, exist_ok=True)
    predictions_json.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_json = args.ground_truth_json.resolve()
    if args.limit is None:
        dataset_count, area_counts = build_coco_ground_truth(data_config, "val", ground_truth_json)
        overall, per_class = evaluate_coco_sizes(ground_truth_json, predictions_json, args.max_det)
    else:
        dataset_count = {"images": len(image_paths), "annotations": None}
        area_counts = None
        overall = None
        per_class = None

    tile_counts = [float(row["tiles"]) for row in per_image]
    raw_counts = [float(row["raw_detections"]) for row in per_image]
    merged_counts = [float(row["merged_detections"]) for row in per_image]
    end_to_end = [float(row["end_to_end_ms"]) for row in per_image]
    processing = [
        float(row["slice_ms"] + row["inference_and_merge_ms"] + row["serialization_ms"])
        for row in per_image
    ]
    raw_total = sum(raw_counts)
    merged_total = sum(merged_counts)
    record = {
        "schema_version": 1,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "Whole-640" if args.mode == "whole" else f"SAHI-{args.slice_size}",
        "checkpoint": {
            "path": str(weights),
            "size_bytes": weights.stat().st_size,
            "sha256": sha256(weights),
        },
        "protocol": {
            "dataset": "VisDrone2019",
            "split": "val",
            "mode": args.mode,
            "slice_size": None if args.mode == "whole" else args.slice_size,
            "input_size": args.input_size,
            "overlap": None if args.mode == "whole" else args.overlap,
            "batch": args.batch,
            "device": args.device,
            "conf": args.conf,
            "per_tile_nms_iou": args.iou,
            "global_merge": None if args.mode == "whole" else "class-aware NMS",
            "global_merge_iou": None if args.mode == "whole" else args.merge_iou,
            "max_det_per_tile": args.max_det,
            "max_det_per_image": args.max_det,
            "half": args.half,
            "limit": args.limit,
            "area_coordinate_system": "original image pixels",
            "area_definition": AREA_DEFINITION,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "dataset_count": dataset_count,
        "ground_truth_area_counts": area_counts,
        "overall_by_area": overall,
        "per_class_by_area": per_class,
        "efficiency": {
            "wall_time_s": total_elapsed,
            "mean_tiles_per_image": float(np.mean(tile_counts)),
            "p95_tiles_per_image": percentile(tile_counts, 95),
            "mean_raw_detections": float(np.mean(raw_counts)),
            "mean_merged_detections": float(np.mean(merged_counts)),
            "duplicate_suppression_ratio": 0.0 if raw_total == 0 else 1.0 - merged_total / raw_total,
            "mean_processing_ms": float(np.mean(processing)),
            "median_processing_ms": float(np.median(processing)),
            "p95_processing_ms": percentile(processing, 95),
            "mean_end_to_end_ms": float(np.mean(end_to_end)),
            "median_end_to_end_ms": float(np.median(end_to_end)),
            "p95_end_to_end_ms": percentile(end_to_end, 95),
            "fps_from_mean_end_to_end": 1000.0 / float(np.mean(end_to_end)),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated() / 1024**2)
                if torch.cuda.is_available() and str(args.device).lower() not in {"cpu", "mps"}
                else None
            ),
        },
        "per_image": per_image,
        "ground_truth_json": str(ground_truth_json),
        "predictions_json": str(predictions_json),
    }
    output_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Structured SAHI evaluation saved to {output_json}")


if __name__ == "__main__":
    main()
