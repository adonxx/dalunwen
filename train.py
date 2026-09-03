"""Train one Drone-YOLO ablation stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import STAGES, build_model  # noqa: E402
from drone_yolo.overlap_tiles import OverlapTileTrainer  # noqa: E402
from drone_yolo.selective_kd import install_scale_selective_kd  # noqa: E402
from drone_yolo.small_crop import SmallObjectCropTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="baseline")
    parser.add_argument("--weights", default="yolo11s.pt", help="Use 'none' to train from scratch")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train" / "paper.yaml")
    parser.add_argument("--epochs", type=int, help="Override epochs, useful for smoke tests")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--device", help="Override CUDA device, e.g. 0 or cpu")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--workers", type=int, help="Override data-loader workers")
    parser.add_argument("--seed", type=int, help="Override the training random seed")
    parser.add_argument("--fraction", type=float, help="Use a fraction of training data, e.g. 0.01 for smoke test")
    parser.add_argument("--name", help="Output run name; defaults to the stage name")
    parser.add_argument("--resume", type=Path, help="Resume from a local last.pt checkpoint")
    parser.add_argument(
        "--small-object-crop",
        action="store_true",
        help="Use the pre-registered small-object-centred crop on training data only",
    )
    parser.add_argument(
        "--overlap-tile-training",
        action="store_true",
        help="Use pre-registered 640px overlapping small-object tile sampling during training only",
    )
    parser.add_argument(
        "--selective-kd-teacher",
        type=Path,
        help="Frozen baseline teacher checkpoint for P3/P4 medium-large selective feature distillation",
    )
    parser.add_argument(
        "--selective-kd-weight",
        type=float,
        default=0.5,
        help="Training-only selective feature-distillation coefficient (default: 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.small_object_crop and args.overlap_tile_training:
        raise ValueError("Choose at most one custom training-data strategy")
    if args.selective_kd_teacher and (args.small_object_crop or args.overlap_tile_training):
        raise ValueError("Selective KD is pre-registered with the standard training data pipeline only")
    with args.config.open("r", encoding="utf-8") as stream:
        train_args = yaml.safe_load(stream)

    train_args["data"] = str((ROOT / train_args["data"]).resolve())
    train_args["project"] = str((ROOT / train_args["project"]).resolve())
    train_args["name"] = args.name or args.stage
    if args.epochs is not None:
        train_args["epochs"] = args.epochs
    if args.batch is not None:
        train_args["batch"] = args.batch
    if args.device is not None:
        train_args["device"] = args.device
    if args.imgsz is not None:
        train_args["imgsz"] = args.imgsz
    if args.workers is not None:
        train_args["workers"] = args.workers
    if args.seed is not None:
        train_args["seed"] = args.seed
    if args.fraction is not None:
        train_args["fraction"] = args.fraction
    if args.selective_kd_teacher:
        teacher_path = args.selective_kd_teacher.resolve()
        if not teacher_path.is_file():
            raise FileNotFoundError(f"Selective-KD teacher checkpoint not found: {teacher_path}")
        install_scale_selective_kd(weight=args.selective_kd_weight)
        # This activates the project-local wrapper through the standard trainer.
        train_args["distill_model"] = str(teacher_path)

    weights = None if args.weights.lower() == "none" else args.weights
    train_args["pretrained"] = weights is not None
    model = build_model(args.stage, weights=weights)
    if args.resume:
        train_args["resume"] = str(args.resume.resolve())
    trainer = (
        OverlapTileTrainer
        if args.overlap_tile_training
        else SmallObjectCropTrainer
        if args.small_object_crop
        else None
    )
    model.train(trainer=trainer, **train_args)


if __name__ == "__main__":
    main()
