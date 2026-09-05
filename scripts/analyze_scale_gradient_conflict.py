"""Measure scale-conditioned gradient conflicts in the shared Drone-YOLO parameters."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import build_model  # noqa: E402


SCALES = ("small", "medium", "large")
PAIRS = (("small", "medium"), ("small", "large"), ("medium", "large"))
OBJECTIVES = ("localization", "total")
VARIANTS = {
    "full_control": ("full", ROOT / "runs" / "loss_inner_wiou_50e" / "weights" / "best.pt"),
    "p4only_candidate": (
        "full_p5lite_p4only",
        ROOT / "runs" / "p5lite_p4only_seed0_50e" / "weights" / "best.pt",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "scale_gradient_conflict")
    return parser.parse_args()


def load_data_config() -> tuple[dict, Path]:
    path = ROOT / "configs" / "data" / "visdrone2019.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    return data, root / data["train"]


def make_loader(data: dict, image_path: Path, args: argparse.Namespace):
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "mode": "val",
            "imgsz": args.imgsz,
            "batch": args.batch_size,
            "workers": args.workers,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
        }
    )
    dataset = build_yolo_dataset(
        cfg,
        str(image_path),
        args.batch_size,
        data,
        mode="val",
        rect=False,
        stride=32,
    )
    return build_dataloader(dataset, batch=args.batch_size, workers=args.workers, shuffle=False, rank=-1)


def scale_masks(batch: dict, imgsz: int) -> dict[str, torch.Tensor]:
    boxes = batch["bboxes"]
    area = boxes[:, 2] * imgsz * boxes[:, 3] * imgsz
    return {
        "small": area < 32.0**2,
        "medium": (area >= 32.0**2) & (area < 96.0**2),
        "large": area >= 96.0**2,
    }


def filtered_batch(batch: dict, mask: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "img": batch["img"].to(device, non_blocking=True).float() / 255.0,
        "batch_idx": batch["batch_idx"][mask].to(device, non_blocking=True),
        "cls": batch["cls"][mask].to(device, non_blocking=True),
        "bboxes": batch["bboxes"][mask].to(device, non_blocking=True),
    }


def parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    head = model.model[-1]
    groups = {
        "shared_features": list(head.shared_conv.parameters()),
        "shared_predictors": list(head.cv2.parameters()) + list(head.cv3.parameters()),
        "deep_backbone": [parameter for layer in model.model[5:9] for parameter in layer.parameters()],
        "neck": [parameter for layer in model.model[9:-1] for parameter in layer.parameters()],
    }
    for name, parameters in groups.items():
        if not parameters:
            raise RuntimeError(f"parameter group {name} is empty")
    return groups


def flattened_gradients(
    objective: torch.Tensor,
    groups: dict[str, list[torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
    parameters: list[torch.nn.Parameter] = []
    offsets: dict[str, tuple[int, int]] = {}
    for name, group in groups.items():
        start = len(parameters)
        parameters.extend(group)
        offsets[name] = (start, len(parameters))
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    vectors = {}
    for name, (start, end) in offsets.items():
        chunks = []
        for parameter, gradient in zip(parameters[start:end], gradients[start:end]):
            chunks.append(
                torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.float32)
                if gradient is None
                else gradient.detach().reshape(-1).float()
            )
        vectors[name] = torch.cat(chunks).cpu()
    return vectors


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.norm() == 0 or right.norm() == 0:
        return math.nan
    return float(F.cosine_similarity(left, right, dim=0).item())


def analyze_variant(
    variant: str,
    stage: str,
    checkpoint: Path,
    data: dict,
    image_path: Path,
    args: argparse.Namespace,
    manifest_reference: list[dict] | None,
) -> tuple[list[dict], list[dict]]:
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    yolo = build_model(stage, weights=str(checkpoint))
    model = yolo.model.to(device).eval()
    model.args = get_cfg(overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5})
    criterion = model.init_criterion()
    # Ultralytics detection criteria are callable objects rather than
    # nn.Module subclasses.  Freeze the stateful Inner-WIoU running mean by
    # putting only its bbox-loss module into evaluation mode.
    criterion.bbox_loss.eval()
    groups = parameter_groups(model)
    loader = make_loader(data, image_path, args)

    rows: list[dict] = []
    manifest: list[dict] = []
    selected = 0
    for source_index, batch in enumerate(loader):
        masks = scale_masks(batch, args.imgsz)
        counts = {name: int(mask.sum().item()) for name, mask in masks.items()}
        if any(counts[name] == 0 for name in SCALES):
            continue
        files = [str(Path(path).resolve()) for path in batch["im_file"]]
        signature = {"batch_index": selected, "source_index": source_index, "files": files, "targets": counts}
        if manifest_reference is not None and signature != manifest_reference[selected]:
            raise RuntimeError(f"sample mismatch at selected batch {selected}")
        manifest.append(signature)

        scale_gradients: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        scale_losses: dict[str, dict[str, float]] = {}
        for scale in SCALES:
            batch_scale = filtered_batch(batch, masks[scale], device)
            predictions = model(batch_scale["img"])
            losses, _ = criterion(predictions, batch_scale)
            scale_losses[scale] = {
                "box": float(losses[0].detach().item()),
                "cls": float(losses[1].detach().item()),
                "dfl": float(losses[2].detach().item()),
            }
            scale_gradients[scale] = {
                "total": flattened_gradients(losses.sum(), groups, retain_graph=True),
                "localization": flattened_gradients(losses[0] + losses[2], groups, retain_graph=False),
            }

        for objective in OBJECTIVES:
            for group in groups:
                for left, right in PAIRS:
                    left_gradient = scale_gradients[left][objective][group]
                    right_gradient = scale_gradients[right][objective][group]
                    rows.append(
                        {
                            "variant": variant,
                            "batch_index": selected,
                            "source_index": source_index,
                            "objective": objective,
                            "parameter_group": group,
                            "pair": f"{left}-{right}",
                            "cosine": cosine(left_gradient, right_gradient),
                            "left_norm": float(left_gradient.norm().item()),
                            "right_norm": float(right_gradient.norm().item()),
                            "left_targets": counts[left],
                            "right_targets": counts[right],
                            "left_loss": sum(scale_losses[left].values()),
                            "right_loss": sum(scale_losses[right].values()),
                        }
                    )
        selected += 1
        print(f"{variant}: analyzed batch {selected}/{args.batches}")
        if selected >= args.batches:
            break

    if selected < args.batches:
        raise RuntimeError(f"only found {selected} batches containing all three target scales")
    del model, yolo, criterion
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, manifest


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if math.isfinite(row["cosine"]):
            grouped[(row["variant"], row["objective"], row["parameter_group"], row["pair"])].append(row["cosine"])
    summary = []
    for (variant, objective, group, pair), values_ in sorted(grouped.items()):
        summary.append(
            {
                "variant": variant,
                "objective": objective,
                "parameter_group": group,
                "pair": pair,
                "n": len(values_),
                "mean_cosine": statistics.fmean(values_),
                "median_cosine": statistics.median(values_),
                "std_cosine": statistics.stdev(values_) if len(values_) > 1 else 0.0,
                "negative_fraction": sum(value < 0 for value in values_) / len(values_),
            }
        )
    return summary


def summary_lookup(summary: list[dict], variant: str, objective: str, group: str, pair: str) -> dict:
    return next(
        row
        for row in summary
        if row["variant"] == variant
        and row["objective"] == objective
        and row["parameter_group"] == group
        and row["pair"] == pair
    )


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_figures(summary: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    groups = ("shared_features", "shared_predictors", "deep_backbone", "neck")
    pairs = tuple(f"{left}-{right}" for left, right in PAIRS)
    variants = tuple(VARIANTS)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, variant in zip(axes, variants):
        matrix = [
            [summary_lookup(summary, variant, "localization", group, pair)["mean_cosine"] for pair in pairs]
            for group in groups
        ]
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(len(pairs)), pairs, rotation=20)
        axis.set_yticks(range(len(groups)), groups)
        axis.set_title(variant)
        for row_index, values_ in enumerate(matrix):
            for column_index, value in enumerate(values_):
                axis.text(column_index, row_index, f"{value:+.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes, label="Mean localization-gradient cosine")
    figure.savefig(output / "localization_gradient_cosine_heatmap.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = list(range(len(groups)))
    width = 0.36
    for offset, variant in zip((-width / 2, width / 2), variants):
        values_ = [
            100
            * summary_lookup(summary, variant, "localization", group, "small-large")["negative_fraction"]
            for group in groups
        ]
        axis.bar([index + offset for index in x], values_, width, label=variant)
    axis.axhline(60, color="black", linestyle="--", linewidth=1, label="60% gate")
    axis.set_xticks(x, groups, rotation=15)
    axis.set_ylabel("Negative small-large gradients (%)")
    axis.set_title("Scale-conflict frequency across analyzed batches")
    axis.legend()
    figure.savefig(output / "small_large_negative_fraction.png", dpi=180)
    plt.close(figure)


def write_report(summary: list[dict], decision: dict, output: Path) -> None:
    rows = []
    for variant in VARIANTS:
        for group in ("shared_features", "shared_predictors", "deep_backbone", "neck"):
            record = summary_lookup(summary, variant, "localization", group, "small-large")
            rows.append(
                f"| {variant} | {group} | {record['mean_cosine']:+.3f} | "
                f"{record['median_cosine']:+.3f} | {100 * record['negative_fraction']:.1f}% | {record['n']} |"
            )
    next_step = (
        "共享头达到预注册冲突标准，可进入P4检测头解耦的50轮单因素实验。"
        if decision["conflict_supported"]
        else "共享头未达到预注册冲突标准，不实施P4头解耦，停止当前P5修复路线。"
    )
    report = f"""# 尺度梯度冲突诊断报告

> 生成时间：{decision['generated_at']}  
> 固定使用相同的16个训练集batch；表中主结果为box+DFL定位损失梯度。

## Small–Large主结果

| 模型 | 参数组 | 平均余弦 | 中位数 | 负梯度比例 | batch数 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

![定位梯度余弦](figures/localization_gradient_cosine_heatmap.png)

![负梯度比例](figures/small_large_negative_fraction.png)

## 预注册判断

- 候选共享特征层：平均余弦 {decision['candidate_shared_features']['mean_cosine']:+.3f}，负梯度比例 {100 * decision['candidate_shared_features']['negative_fraction']:.1f}%。
- 候选共享预测器：平均余弦 {decision['candidate_shared_predictors']['mean_cosine']:+.3f}，负梯度比例 {100 * decision['candidate_shared_predictors']['negative_fraction']:.1f}%。
- 判定条件：任一共享参数组平均余弦 < -0.05、负梯度比例 ≥ 60%、n ≥ 16。

**结论：{next_step}**

## 解释边界

该诊断使用关闭增强的固定训练样本，并通过只保留单一尺度标注构造尺度条件损失。定位损失是主判据；总损失包含被移除目标在分类项中形成背景的影响，仅作为敏感性分析。梯度冲突可以支持“共享优化存在竞争”，但不能单独证明它是AP变化的唯一原因，仍需通过检测头解耦消融验证因果性。

## 可追溯产物

- `metrics/gradient_cosines.csv`：逐batch余弦、梯度范数、目标数和损失。
- `metrics/gradient_summary.csv`：均值、中位数、标准差和负值比例。
- `metrics/sample_manifest.json`：固定图像与尺度目标计数。
- `metrics/decision.json`：机器可读判定。
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batches < 16:
        raise ValueError("the pre-registered diagnostic requires at least 16 batches")
    for _, checkpoint in VARIANTS.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    output = args.output_dir.resolve()
    metrics = output / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    data, image_path = load_data_config()

    all_rows: list[dict] = []
    reference_manifest = None
    for variant, (stage, checkpoint) in VARIANTS.items():
        rows, manifest = analyze_variant(
            variant,
            stage,
            checkpoint,
            data,
            image_path,
            args,
            reference_manifest,
        )
        if reference_manifest is None:
            reference_manifest = manifest
        all_rows.extend(rows)

    summary = summarize(all_rows)
    save_csv(metrics / "gradient_cosines.csv", all_rows)
    save_csv(metrics / "gradient_summary.csv", summary)
    (metrics / "sample_manifest.json").write_text(
        json.dumps(reference_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    candidate_features = summary_lookup(
        summary, "p4only_candidate", "localization", "shared_features", "small-large"
    )
    candidate_predictors = summary_lookup(
        summary, "p4only_candidate", "localization", "shared_predictors", "small-large"
    )
    conflict_supported = any(
        record["mean_cosine"] < -0.05
        and record["negative_fraction"] >= 0.60
        and record["n"] >= 16
        for record in (candidate_features, candidate_predictors)
    )
    decision = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "conflict_supported": conflict_supported,
        "criterion": "candidate shared_features or shared_predictors: mean localization cosine < -0.05, negative fraction >= 0.60, n >= 16",
        "candidate_shared_features": candidate_features,
        "candidate_shared_predictors": candidate_predictors,
    }
    (metrics / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    create_figures(summary, output / "figures")
    write_report(summary, decision, output)
    print(f"Gradient-conflict report generated; conflict_supported={conflict_supported}")


if __name__ == "__main__":
    main()
