"""Calibrate and evaluate prediction-density gated SAHI-960 inference."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drone_yolo import build_model  # noqa: E402
from evaluate_object_sizes import (  # noqa: E402
    IMAGE_SUFFIXES,
    evaluate_coco_sizes,
    image_id,
    resolve_split_dirs,
)
from evaluate_sahi import (  # noqa: E402
    detections_to_coco,
    predict_sliced,
    predict_whole,
    sha256,
    slice_windows,
    synchronize,
)
from evaluate_scale48_joint import filter_scale_tensors, timing_summary  # noqa: E402
from fuse_sahi_predictions import merge_image_predictions  # noqa: E402


CALIBRATION_QUANTILES = (20, 30)


def density_scores(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    windows: list[tuple[int, int, int, int]],
    area_threshold: float = 48.0,
    confidence_threshold: float = 0.05,
) -> list[float]:
    """Score each tile using only Whole-predicted small-box centers."""
    if boxes.numel() == 0:
        return [0.0] * len(windows)
    boxes_cpu = boxes.detach().cpu().float()
    scores_cpu = scores.detach().cpu().float()
    widths = (boxes_cpu[:, 2] - boxes_cpu[:, 0]).clamp(min=0)
    heights = (boxes_cpu[:, 3] - boxes_cpu[:, 1]).clamp(min=0)
    mask = (widths * heights < area_threshold**2) & (scores_cpu >= confidence_threshold)
    if not bool(mask.any()):
        return [0.0] * len(windows)
    centers_x = (boxes_cpu[mask, 0] + boxes_cpu[mask, 2]) / 2
    centers_y = (boxes_cpu[mask, 1] + boxes_cpu[mask, 3]) / 2
    selected_scores = scores_cpu[mask]
    output = []
    for x1, y1, x2, y2 in windows:
        inside = (centers_x >= x1) & (centers_x <= x2) & (centers_y >= y1) & (centers_y <= y2)
        output.append(float(selected_scores[inside].sum()))
    return output


def select_top_window(
    windows: list[tuple[int, int, int, int]], scores: list[float], threshold: float
) -> tuple[list[tuple[int, int, int, int]], float, int | None]:
    """Return at most one deterministic tile when the gate threshold is met."""
    if not windows or not scores:
        return [], 0.0, None
    index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    maximum = float(scores[index])
    if maximum <= 0.0 or maximum < threshold:
        return [], maximum, None
    return [windows[index]], maximum, index


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    """Select a deterministic, distribution-spanning calibration subset."""
    if count <= 0:
        raise ValueError("calibration count must be positive")
    if count >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[int(index)] for index in indices]


def build_runtime(args: argparse.Namespace) -> tuple[Any, SimpleNamespace, SimpleNamespace, dict[str, Any]]:
    data_path = (ROOT / "configs" / "data" / "visdrone2019.yaml").resolve()
    data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    model = build_model("full", weights=str(args.weights.resolve()))
    model.model.names = data_config["names"]
    common = {
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": args.device,
        "batch": args.batch,
        "half": False,
        "merge_iou": args.merge_iou,
    }
    whole_args = SimpleNamespace(**common, input_size=640)
    sahi_args = SimpleNamespace(**common, input_size=960)
    for side in (640, 960):
        warmup = np.zeros((side, side, 3), dtype=np.uint8)
        for _ in range(3):
            model.predict(
                source=warmup,
                imgsz=side,
                rect=False,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )
    synchronize(args.device)
    return model, whole_args, sahi_args, data_config


def calibrate(args: argparse.Namespace) -> None:
    model, whole_args, _, data_config = build_runtime(args)
    image_dir, _ = resolve_split_dirs(data_config, "train")
    all_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    image_paths = evenly_spaced(all_paths, args.calibration_images)
    maximum_scores = []
    rows = []
    started = time.perf_counter()
    for index, path in enumerate(image_paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        boxes, scores, _ = predict_whole(model, image, whole_args)
        height, width = image.shape[:2]
        windows = slice_windows(width, height, 960, 0.20)
        tile_scores = density_scores(boxes, scores, windows, 48.0, 0.05)
        maximum = max(tile_scores, default=0.0)
        maximum_scores.append(maximum)
        rows.append(
            {
                "file_name": path.name,
                "width": width,
                "height": height,
                "tiles": len(windows),
                "max_density_score": maximum,
            }
        )
        if index == 1 or index % 50 == 0 or index == len(image_paths):
            print(f"[{index:>3}/{len(image_paths)}] calibration {path.name}: max density={maximum:.3f}")
    quantiles = {
        f"q{quantile}": float(np.percentile(maximum_scores, quantile))
        for quantile in CALIBRATION_QUANTILES
    }
    output = {
        "schema_version": 1,
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(args.weights.resolve()),
            "sha256": sha256(args.weights.resolve()),
        },
        "protocol": {
            "dataset": "VisDrone2019",
            "split": "train",
            "selection": "512 evenly spaced sorted image paths",
            "labels_read": False,
            "calibration_images": len(image_paths),
            "whole_input": 640,
            "tile_size": 960,
            "overlap": 0.20,
            "candidate_area": "predicted area < 48^2 pixels",
            "candidate_confidence": 0.05,
            "tile_score": "sum of candidate confidence for centers inside tile",
        },
        "quantiles": quantiles,
        "distribution": {
            "minimum": float(np.min(maximum_scores)),
            "mean": float(np.mean(maximum_scores)),
            "median": float(np.median(maximum_scores)),
            "maximum": float(np.max(maximum_scores)),
            "zero_score_images": int(sum(value <= 0.0 for value in maximum_scores)),
        },
        "wall_time_s": time.perf_counter() - started,
        "per_image": rows,
    }
    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Calibration saved to {output_path}: {quantiles}")


def evaluate(args: argparse.Namespace) -> None:
    calibration = json.loads(args.calibration.resolve().read_text(encoding="utf-8"))
    thresholds = {
        "top1": 0.0,
        "gate_q20": float(calibration["quantiles"]["q20"]),
        "gate_q30": float(calibration["quantiles"]["q30"]),
    }
    gate_threshold = thresholds[args.strategy]
    model, whole_args, sahi_args, data_config = build_runtime(args)
    if calibration["checkpoint"]["sha256"] != sha256(args.weights.resolve()):
        raise ValueError("calibration and evaluation checkpoints do not match")
    image_dir, _ = resolve_split_dirs(data_config, "val")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if torch.cuda.is_available() and str(args.device).lower() not in {"cpu", "mps"}:
        torch.cuda.reset_peak_memory_stats()

    predictions: list[dict[str, Any]] = []
    per_image = []
    total_started = time.perf_counter()
    for index, path in enumerate(image_paths, start=1):
        image_started = time.perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        decoded = time.perf_counter()
        synchronize(args.device)
        whole_started = time.perf_counter()
        whole_boxes, whole_scores, whole_classes = predict_whole(model, image, whole_args)
        synchronize(args.device)
        whole_finished = time.perf_counter()

        height, width = image.shape[:2]
        all_windows = slice_windows(width, height, 960, 0.20)
        gate_started = time.perf_counter()
        tile_scores = density_scores(whole_boxes, whole_scores, all_windows, 48.0, 0.05)
        selected_windows, maximum_score, selected_index = select_top_window(
            all_windows, tile_scores, gate_threshold
        )
        gate_finished = time.perf_counter()

        sahi_started = time.perf_counter()
        if selected_windows:
            sahi_boxes, sahi_scores, sahi_classes, raw_sahi_count = predict_sliced(
                model, image, selected_windows, sahi_args
            )
        else:
            sahi_boxes = whole_boxes.new_empty((0, 4))
            sahi_scores = whole_scores.new_empty((0,))
            sahi_classes = whole_classes.new_empty((0,), dtype=torch.long)
            raw_sahi_count = 0
        synchronize(args.device)
        sahi_finished = time.perf_counter()

        fusion_started = time.perf_counter()
        selected_boxes, selected_scores, selected_classes = filter_scale_tensors(
            sahi_boxes, sahi_scores, sahi_classes, 48.0
        )
        whole_records = detections_to_coco(path, whole_boxes, whole_scores, whole_classes)
        sahi_records = detections_to_coco(path, selected_boxes, selected_scores, selected_classes)
        fused_records, source_counts = merge_image_predictions(
            whole_records, sahi_records, args.merge_iou, args.max_det
        )
        fusion_finished = time.perf_counter()
        predictions.extend(fused_records)
        image_finished = time.perf_counter()
        per_image.append(
            {
                "image_id": image_id(path),
                "file_name": path.name,
                "width": width,
                "height": height,
                "available_tiles": len(all_windows),
                "selected_tiles": len(selected_windows),
                "selected_window_index": selected_index,
                "selected_window": list(selected_windows[0]) if selected_windows else None,
                "max_density_score": maximum_score,
                "gate_threshold": gate_threshold,
                "raw_sahi_detections": raw_sahi_count,
                "selected_sahi_detections": len(sahi_records),
                "final_detections": len(fused_records),
                "kept_whole": source_counts["kept_whole"],
                "kept_sahi": source_counts["kept_sahi"],
                "decode_ms": (decoded - image_started) * 1000.0,
                "whole_ms": (whole_finished - whole_started) * 1000.0,
                "slice_window_ms": (gate_finished - gate_started) * 1000.0,
                "sahi_ms": (sahi_finished - sahi_started) * 1000.0,
                "filter_and_fusion_ms": (fusion_finished - fusion_started) * 1000.0,
                "serialization_ms": (image_finished - fusion_finished) * 1000.0,
                "end_to_end_ms": (image_finished - image_started) * 1000.0,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(image_paths):
            print(
                f"[{index:>3}/{len(image_paths)}] {path.name}: score={maximum_score:.3f}, "
                f"selected={len(selected_windows)}/{len(all_windows)}, final={len(fused_records)}"
            )

    wall_time_s = time.perf_counter() - total_started
    predictions_path = args.predictions_json.resolve()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    if args.limit is None:
        overall, per_class = evaluate_coco_sizes(
            args.ground_truth.resolve(), predictions_path, args.max_det
        )
    else:
        overall, per_class = None, None
    timing = timing_summary(per_image)
    timing.update(
        {
            "wall_time_s": wall_time_s,
            "gate_trigger_images": int(sum(row["selected_tiles"] > 0 for row in per_image)),
            "gate_trigger_rate": float(np.mean([row["selected_tiles"] > 0 for row in per_image])),
            "mean_available_tiles": float(np.mean([row["available_tiles"] for row in per_image])),
            "mean_selected_tiles": float(np.mean([row["selected_tiles"] for row in per_image])),
            "tile_reduction_ratio": 1.0
            - sum(row["selected_tiles"] for row in per_image)
            / sum(row["available_tiles"] for row in per_image),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated() / 1024**2)
                if torch.cuda.is_available() and str(args.device).lower() not in {"cpu", "mps"}
                else None
            ),
        }
    )
    output = {
        "schema_version": 1,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "strategy": args.strategy,
        "checkpoint": {
            "path": str(args.weights.resolve()),
            "sha256": sha256(args.weights.resolve()),
            "matches_calibration": sha256(args.weights.resolve()) == calibration["checkpoint"]["sha256"],
        },
        "protocol": {
            "dataset": "VisDrone2019",
            "split": "val",
            "labels_used_for_gate": False,
            "whole_input": 640,
            "slice_input": 960,
            "overlap": 0.20,
            "candidate_area": "predicted area < 48^2 pixels",
            "candidate_confidence": 0.05,
            "density_score": "sum of candidate confidence inside tile",
            "gate_threshold": gate_threshold,
            "maximum_selected_tiles": 1,
            "scale_threshold": 48.0,
            "conf": args.conf,
            "per_route_nms_iou": args.iou,
            "merge_iou": args.merge_iou,
            "max_det": args.max_det,
            "batch": args.batch,
            "device": args.device,
            "precision": "FP32",
            "limit": args.limit,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "dataset_count": {"images": len(image_paths)},
        "overall_by_area": overall,
        "per_class_by_area": per_class,
        "efficiency": timing,
        "calibration_json": str(args.calibration.resolve()),
        "per_image": per_image,
        "ground_truth_json": str(args.ground_truth.resolve()),
        "predictions_json": str(predictions_path),
    }
    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Density-gated evaluation saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--strategy", choices=("top1", "gate_q20", "gate_q30"))
    parser.add_argument("--predictions-json", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--calibration-images", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.mode == "calibrate":
        calibrate(args)
    else:
        if not args.calibration or not args.strategy or not args.predictions_json or not args.ground_truth:
            parser.error("evaluate mode requires --calibration, --strategy, --predictions-json, and --ground-truth")
        evaluate(args)


if __name__ == "__main__":
    main()
