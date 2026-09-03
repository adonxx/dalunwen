"""Audit dataset availability and model structure before expensive training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import STAGES, build_model  # noqa: E402


PAPER_TARGETS = {
    "baseline": {"params_m": 9.416, "gflops": 21.3},
    "mffpn": {"params_m": 3.181, "gflops": 24.5},
    "lscd": {"params_m": 3.060, "gflops": 19.6},
    "full": {"params_m": 3.060, "gflops": 19.6},
}


def dataset_counts() -> dict[str, tuple[int, int]]:
    with (ROOT / "configs" / "data" / "visdrone2019.yaml").open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    dataset_root = Path(data["path"])
    counts = {}
    for split in ("train", "val", "test"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        images = sum(1 for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        labels = sum(1 for p in label_dir.glob("*.txt"))
        counts[split] = (images, labels)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = parser.parse_args()

    print("Dataset:")
    for split, (images, labels) in dataset_counts().items():
        print(f"  {split:5s} images={images:4d} labels={labels:4d}")
    print(f"CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    selected = STAGES if args.stage == "all" else (args.stage,)
    print("Models:")
    for stage_name in selected:
        model = build_model(stage_name).model
        params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        gflops = get_flops(model, imgsz=640)
        strides = ",".join(str(int(value)) for value in model.stride)
        target = PAPER_TARGETS.get(stage_name)
        target_text = (
            f" (paper {target['params_m']:.3f}M), GFLOPs={gflops:.2f} (paper {target['gflops']:.1f})"
            if target
            else f", GFLOPs={gflops:.2f} (experimental ablation)"
        )
        print(f"  {stage_name:12s} params={params_m:.3f}M{target_text} strides=[{strides}]")


if __name__ == "__main__":
    main()
