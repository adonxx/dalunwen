"""Fuse cached Whole-640 and SAHI-960 predictions with fixed scale rules."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.ops import batched_nms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_object_sizes import evaluate_coco_sizes  # noqa: E402


STRATEGIES: dict[str, float | None] = {
    "union_all": None,
    "scale_32": 32.0,
    "scale_48": 48.0,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def box_area(record: dict[str, Any]) -> float:
    """Return one COCO ``xywh`` prediction area in original pixels."""
    return max(0.0, float(record["bbox"][2])) * max(0.0, float(record["bbox"][3]))


def select_sahi(records: list[dict[str, Any]], threshold: float | None) -> list[dict[str, Any]]:
    """Select all SAHI boxes or boxes below one pre-registered scale."""
    if threshold is None:
        return list(records)
    maximum_area = threshold**2
    return [record for record in records if box_area(record) < maximum_area]


def merge_image_predictions(
    whole: list[dict[str, Any]],
    sahi: list[dict[str, Any]],
    iou_threshold: float,
    max_det: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge one image with class-aware NMS and retain source statistics."""
    candidates = list(whole) + list(sahi)
    if not candidates:
        return [], {"raw": 0, "kept": 0, "kept_whole": 0, "kept_sahi": 0}
    boxes = []
    scores = []
    classes = []
    for record in candidates:
        x, y, width, height = [float(value) for value in record["bbox"]]
        boxes.append([x, y, x + width, y + height])
        scores.append(float(record["score"]))
        classes.append(int(record["category_id"]))
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    class_tensor = torch.tensor(classes, dtype=torch.long)
    keep = batched_nms(box_tensor, score_tensor, class_tensor, iou_threshold)[:max_det].tolist()
    output = [candidates[index] for index in keep]
    whole_count = len(whole)
    return output, {
        "raw": len(candidates),
        "kept": len(output),
        "kept_whole": sum(index < whole_count for index in keep),
        "kept_sahi": sum(index >= whole_count for index in keep),
    }


def group_predictions(records: list[dict[str, Any]]) -> dict[int | str, list[dict[str, Any]]]:
    grouped: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["image_id"]].append(record)
    return grouped


def run_strategy(
    key: str,
    threshold: float | None,
    whole_by_image: dict[int | str, list[dict[str, Any]]],
    sahi_by_image: dict[int | str, list[dict[str, Any]]],
    image_ids: list[int | str],
    iou_threshold: float,
    max_det: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fuse all validation images for one fixed rule."""
    output: list[dict[str, Any]] = []
    per_image = []
    times_ms = []
    for current_id in image_ids:
        selected_sahi = select_sahi(sahi_by_image.get(current_id, []), threshold)
        started = time.perf_counter()
        merged, counts = merge_image_predictions(
            whole_by_image.get(current_id, []), selected_sahi, iou_threshold, max_det
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output.extend(merged)
        times_ms.append(elapsed_ms)
        per_image.append({"image_id": current_id, "fusion_ms": elapsed_ms, **counts})
    raw_total = sum(row["raw"] for row in per_image)
    kept_total = sum(row["kept"] for row in per_image)
    stats = {
        "strategy": key,
        "scale_threshold_px": threshold,
        "mean_fusion_ms": float(np.mean(times_ms)),
        "median_fusion_ms": float(np.median(times_ms)),
        "p95_fusion_ms": float(np.percentile(times_ms, 95)),
        "raw_candidates": raw_total,
        "kept_predictions": kept_total,
        "suppression_ratio": 0.0 if raw_total == 0 else 1.0 - kept_total / raw_total,
        "kept_whole": sum(row["kept_whole"] for row in per_image),
        "kept_sahi": sum(row["kept_sahi"] for row in per_image),
        "per_image": per_image,
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole-predictions", type=Path, required=True)
    parser.add_argument("--sahi-predictions", type=Path, required=True)
    parser.add_argument("--whole-record", type=Path, required=True)
    parser.add_argument("--sahi-record", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    whole_predictions = load_json(args.whole_predictions.resolve())
    sahi_predictions = load_json(args.sahi_predictions.resolve())
    whole_record = load_json(args.whole_record.resolve())
    sahi_record = load_json(args.sahi_record.resolve())
    ground_truth = load_json(args.ground_truth.resolve())
    if whole_record["checkpoint"]["sha256"] != sahi_record["checkpoint"]["sha256"]:
        raise ValueError("Whole and SAHI predictions must use the same checkpoint")
    image_ids = [image["id"] for image in ground_truth["images"]]
    if len(image_ids) != 548:
        raise ValueError(f"expected 548 validation images, found {len(image_ids)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    whole_by_image = group_predictions(whole_predictions)
    sahi_by_image = group_predictions(sahi_predictions)
    for key, threshold in STRATEGIES.items():
        print(f"Running {key} with scale threshold {threshold}")
        predictions, stats = run_strategy(
            key,
            threshold,
            whole_by_image,
            sahi_by_image,
            image_ids,
            args.merge_iou,
            args.max_det,
        )
        predictions_path = output_dir / f"{key}_predictions.json"
        predictions_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
        overall, per_class = evaluate_coco_sizes(
            args.ground_truth.resolve(), predictions_path, args.max_det
        )
        conservative_latency = (
            whole_record["efficiency"]["mean_end_to_end_ms"]
            + sahi_record["efficiency"]["mean_end_to_end_ms"]
            + stats["mean_fusion_ms"]
        )
        record = {
            "schema_version": 1,
            "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "strategy": key,
            "checkpoint": whole_record["checkpoint"],
            "protocol": {
                "dataset": "VisDrone2019",
                "split": "val",
                "whole_source": str(args.whole_predictions.resolve()),
                "sahi_source": str(args.sahi_predictions.resolve()),
                "scale_threshold_px": threshold,
                "scale_rule": "predicted width*height below threshold^2" if threshold else "all SAHI boxes",
                "merge": "class-aware NMS",
                "merge_iou": args.merge_iou,
                "max_det_per_image": args.max_det,
                "score_recalibration": None,
            },
            "dataset_count": {"images": len(image_ids)},
            "overall_by_area": overall,
            "per_class_by_area": per_class,
            "fusion_efficiency": {
                **stats,
                "conservative_sequential_end_to_end_ms": conservative_latency,
                "conservative_sequential_fps": 1000.0 / conservative_latency,
                "latency_note": "Whole and SAHI measured end-to-end means summed with cached CPU fusion time; not an integrated deployment benchmark.",
            },
            "ground_truth_json": str(args.ground_truth.resolve()),
            "predictions_json": str(predictions_path),
        }
        (output_dir / f"{key}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved {key} metrics to {output_dir / f'{key}.json'}")


if __name__ == "__main__":
    main()
