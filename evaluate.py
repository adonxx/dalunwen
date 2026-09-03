"""Evaluate a saved checkpoint with the matching local Drone-YOLO stage.

Besides Ultralytics' standard plots, this entry point can emit a structured JSON
record.  The JSON is used by the thesis report generator so every table and
figure can be traced back to one fixed evaluation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import STAGES, build_model  # noqa: E402


def sha256(path: Path) -> str:
    """Return a checkpoint hash for provenance tracking."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_count(data_config: dict, split: str) -> dict[str, int]:
    """Count images and labels without modifying the dataset."""
    root = Path(data_config["path"])
    image_dir = root / Path(data_config[split])
    label_dir = image_dir.parent / "labels"
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return {
        "images": sum(path.suffix.lower() in image_suffixes for path in image_dir.iterdir()),
        "labels": sum(1 for _ in label_dir.glob("*.txt")),
    }


def model_structure(model) -> dict:
    """Measure the unfused model graph used to define the paper architecture."""
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    return {
        "parameters": parameters,
        "parameters_m": parameters / 1e6,
        "gflops_640": float(get_flops(model.model, imgsz=640)),
        "strides": [float(value) for value in model.model.stride],
    }


def build_record(
    args: argparse.Namespace,
    metrics,
    model,
    unfused_structure: dict,
    data_config: dict,
    save_dir: Path,
) -> dict:
    """Convert an Ultralytics DetMetrics object into stable JSON data."""
    names = {int(index): str(name) for index, name in data_config["names"].items()}
    class_rows = []
    for result_index, class_index in enumerate(metrics.box.ap_class_index):
        class_index = int(class_index)
        precision, recall, ap50, ap = metrics.box.class_result(result_index)
        class_rows.append(
            {
                "class_id": class_index,
                "class_name": names[class_index],
                "precision": float(precision),
                "recall": float(recall),
                "ap50": float(ap50),
                "ap50_95": float(ap),
            }
        )

    checkpoint = args.weights.resolve()
    fused_structure = model_structure(model)
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
            "data_yaml": str((ROOT / "configs" / "data" / "visdrone2019.yaml").resolve()),
            "split": args.split,
            "dataset_count": dataset_count(data_config, args.split),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "half": args.half,
            "plots": args.plots,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model": {
            **unfused_structure,
            "post_validation_fused_parameters": fused_structure["parameters"],
            "post_validation_fused_parameters_m": fused_structure["parameters_m"],
            "post_validation_fused_gflops_640": fused_structure["gflops_640"],
        },
        "overall": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class": class_rows,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "save_dir": str(save_dir.resolve()),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    data_path = (ROOT / "configs" / "data" / "visdrone2019.yaml").resolve()
    with data_path.open("r", encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)
    model = build_model(args.stage, weights=str(args.weights.resolve()))
    model.model.names = data_config["names"]
    unfused_structure = model_structure(model)
    metrics = model.val(
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
        plots=args.plots,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
    )
    if args.output_json:
        output_json = args.output_json.resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        record = build_record(args, metrics, model, unfused_structure, data_config, Path(metrics.save_dir))
        output_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Structured metrics saved to {output_json}")


if __name__ == "__main__":
    main()
