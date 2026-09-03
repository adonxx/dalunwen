"""Run true Whole-640 + SAHI-960 + Scale-48 joint inference."""

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
from fuse_sahi_predictions import merge_image_predictions  # noqa: E402


def filter_scale_tensors(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep predictions whose original-coordinate area is below threshold²."""
    if boxes.numel() == 0:
        return boxes, scores, classes
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    keep = widths * heights < threshold**2
    return boxes[keep], scores[keep], classes[keep]


def timing_summary(per_image: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate detailed per-image timings into stable JSON statistics."""
    fields = (
        "decode_ms",
        "whole_ms",
        "slice_window_ms",
        "sahi_ms",
        "filter_and_fusion_ms",
        "serialization_ms",
        "end_to_end_ms",
    )
    summary: dict[str, float] = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in per_image], dtype=np.float64)
        stem = field.removesuffix("_ms")
        summary[f"mean_{stem}_ms"] = float(values.mean())
        summary[f"median_{stem}_ms"] = float(np.median(values))
        summary[f"p95_{stem}_ms"] = float(np.percentile(values, 95))
    summary["fps_from_mean_end_to_end"] = 1000.0 / summary["mean_end_to_end_ms"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--offline-record", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--slice-size", type=int, default=960)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--scale-threshold", type=float, default=48.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.slice_size != 960 or args.scale_threshold != 48.0 or not np.isclose(args.overlap, 0.20):
        raise ValueError("the pre-registered joint experiment requires slice=960, scale=48, overlap=0.20")

    data_path = (ROOT / "configs" / "data" / "visdrone2019.yaml").resolve()
    data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    image_dir, _ = resolve_split_dirs(data_config, "val")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    weights = args.weights.resolve()
    model = build_model("full", weights=str(weights))
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
    sahi_args = SimpleNamespace(**common, input_size=args.slice_size)

    for side in (640, args.slice_size):
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

        synchronize(args.device)
        whole_started = time.perf_counter()
        whole_boxes, whole_scores, whole_classes = predict_whole(model, image, whole_args)
        synchronize(args.device)
        whole_finished = time.perf_counter()

        height, width = image.shape[:2]
        windows = slice_windows(width, height, args.slice_size, args.overlap)
        windows_finished = time.perf_counter()
        synchronize(args.device)
        sahi_started = time.perf_counter()
        sahi_boxes, sahi_scores, sahi_classes, raw_sahi_count = predict_sliced(
            model, image, windows, sahi_args
        )
        synchronize(args.device)
        sahi_finished = time.perf_counter()

        fusion_started = time.perf_counter()
        selected_boxes, selected_scores, selected_classes = filter_scale_tensors(
            sahi_boxes, sahi_scores, sahi_classes, args.scale_threshold
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
                "image_id": fused_records[0]["image_id"] if fused_records else path.stem,
                "file_name": path.name,
                "width": width,
                "height": height,
                "tiles": len(windows),
                "whole_detections": len(whole_records),
                "raw_sahi_detections": raw_sahi_count,
                "merged_sahi_detections": int(sahi_scores.numel()),
                "selected_sahi_detections": len(sahi_records),
                "final_detections": len(fused_records),
                "kept_whole": source_counts["kept_whole"],
                "kept_sahi": source_counts["kept_sahi"],
                "decode_ms": (decoded - image_started) * 1000.0,
                "whole_ms": (whole_finished - whole_started) * 1000.0,
                "slice_window_ms": (windows_finished - whole_finished) * 1000.0,
                "sahi_ms": (sahi_finished - sahi_started) * 1000.0,
                "filter_and_fusion_ms": (fusion_finished - fusion_started) * 1000.0,
                "serialization_ms": (image_finished - fusion_finished) * 1000.0,
                "end_to_end_ms": (image_finished - image_started) * 1000.0,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(image_paths):
            print(
                f"[{index:>3}/{len(image_paths)}] {path.name}: {len(windows)} tiles, "
                f"{len(sahi_records)} SAHI supplements, {len(fused_records)} final"
            )

    wall_time_s = time.perf_counter() - total_started
    predictions_json = args.predictions_json.resolve()
    predictions_json.parent.mkdir(parents=True, exist_ok=True)
    predictions_json.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    if args.limit is None:
        overall, per_class = evaluate_coco_sizes(
            args.ground_truth.resolve(), predictions_json, args.max_det
        )
    else:
        overall, per_class = None, None
    offline = json.loads(args.offline_record.resolve().read_text(encoding="utf-8"))
    differences = None
    if overall is not None:
        differences = {
            area: {
                metric: overall[area][metric] - offline["overall_by_area"][area][metric]
                for metric in ("ap", "ap50", "ap75", "ar_max_det")
            }
            for area in ("all", "small", "medium", "large")
        }
    timing = timing_summary(per_image)
    timing["wall_time_s"] = wall_time_s
    timing["mean_tiles_per_image"] = float(np.mean([row["tiles"] for row in per_image]))
    timing["p95_tiles_per_image"] = float(np.percentile([row["tiles"] for row in per_image], 95))
    timing["mean_selected_sahi_detections"] = float(
        np.mean([row["selected_sahi_detections"] for row in per_image])
    )
    timing["mean_final_detections"] = float(np.mean([row["final_detections"] for row in per_image]))
    timing["peak_gpu_memory_mb"] = (
        float(torch.cuda.max_memory_allocated() / 1024**2)
        if torch.cuda.is_available() and str(args.device).lower() not in {"cpu", "mps"}
        else None
    )

    current_hash = sha256(weights)
    output = {
        "schema_version": 1,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "Scale-48 joint inference",
        "checkpoint": {
            "path": str(weights),
            "size_bytes": weights.stat().st_size,
            "sha256": current_hash,
            "matches_offline": current_hash == offline["checkpoint"]["sha256"],
        },
        "protocol": {
            "dataset": "VisDrone2019",
            "split": "val",
            "whole_input": 640,
            "slice_input": args.slice_size,
            "overlap": args.overlap,
            "scale_threshold_px": args.scale_threshold,
            "batch": args.batch,
            "conf": args.conf,
            "per_route_nms_iou": args.iou,
            "sahi_and_final_merge_iou": args.merge_iou,
            "max_det": args.max_det,
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
        "offline_record": str(args.offline_record.resolve()),
        "differences_vs_offline": differences,
        "efficiency": timing,
        "per_image": per_image,
        "ground_truth_json": str(args.ground_truth.resolve()),
        "predictions_json": str(predictions_json),
    }
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Joint Scale-48 record saved to {output_json}")


if __name__ == "__main__":
    main()
