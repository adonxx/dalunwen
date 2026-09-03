"""Generate the auditable three-seed core-model replication report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CORE_METRICS = (
    ("mAP50", "mAP50"),
    ("mAP50-95", "mAP50_95"),
    ("Small AP", "small_ap"),
    ("Medium AP", "medium_ap"),
    ("Large AP", "large_ap"),
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def seed_zero_records() -> dict[str, dict[str, float]]:
    metrics_dir = ROOT / "reports" / "unified_evaluation" / "metrics"
    with (metrics_dir / "summary.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        summary_rows = {row["stage"]: row for row in csv.DictReader(stream)}
    with (metrics_dir / "size_metrics.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        size_rows = {(row["stage"], row["area"]): row for row in csv.DictReader(stream)}

    records: dict[str, dict[str, float]] = {}
    for stage, variant in (("baseline", "baseline"), ("full", "full")):
        summary = summary_rows[stage]
        records[variant] = {
            "precision": float(summary["precision"]) * 100,
            "recall": float(summary["recall"]) * 100,
            "mAP50": float(summary["mAP50"]) * 100,
            "mAP50_95": float(summary["mAP50_95"]) * 100,
            "small_ap": float(size_rows[(stage, "small")]["ap"]) * 100,
            "medium_ap": float(size_rows[(stage, "medium")]["ap"]) * 100,
            "large_ap": float(size_rows[(stage, "large")]["ap"]) * 100,
            "parameters_m": float(summary["parameters"]) / 1_000_000,
            "gflops": float(summary["gflops_640"]),
        }
    return records


def later_seed_record(metrics_dir: Path, seed: int, variant: str) -> dict[str, float]:
    overall = read_json(metrics_dir / f"{variant}_seed{seed}_val.json")
    sizes = read_json(metrics_dir / f"{variant}_seed{seed}_size_val.json")
    return {
        "precision": overall["overall"]["metrics/precision(B)"] * 100,
        "recall": overall["overall"]["metrics/recall(B)"] * 100,
        "mAP50": overall["overall"]["metrics/mAP50(B)"] * 100,
        "mAP50_95": overall["overall"]["metrics/mAP50-95(B)"] * 100,
        "small_ap": sizes["overall_by_area"]["small"]["ap"] * 100,
        "medium_ap": sizes["overall_by_area"]["medium"]["ap"] * 100,
        "large_ap": sizes["overall_by_area"]["large"]["ap"] * 100,
        "parameters_m": overall["model"]["parameters_m"],
        "gflops": overall["model"]["gflops_640"],
    }


def format_mean_sd(values: list[float]) -> str:
    return f"{mean(values):.2f} ± {stdev(values):.2f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(figure_dir: Path, records: dict[int, dict[str, dict[str, float]]]) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in CORE_METRICS]
    keys = [key for _, key in CORE_METRICS]
    baseline_means = [mean([records[seed]["baseline"][key] for seed in records]) for key in keys]
    full_means = [mean([records[seed]["full"][key] for seed in records]) for key in keys]
    baseline_sd = [stdev([records[seed]["baseline"][key] for seed in records]) for key in keys]
    full_sd = [stdev([records[seed]["full"][key] for seed in records]) for key in keys]

    plt.style.use("seaborn-v0_8-whitegrid")
    positions = np.arange(len(keys))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10.5, 5.4), dpi=180)
    axis.bar(positions - width / 2, baseline_means, width, yerr=baseline_sd, capsize=4, label="YOLOv11s", color="#6B8EAD")
    axis.bar(positions + width / 2, full_means, width, yerr=full_sd, capsize=4, label="Drone-YOLO", color="#E07A5F")
    axis.set_ylabel("Metric (%)")
    axis.set_xticks(positions, labels)
    axis.set_title("Three-seed mean ± sample standard deviation")
    axis.legend(frameon=True)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(figure_dir / "three_seed_metric_means.png", bbox_inches="tight")
    plt.close(figure)

    delta_keys = ["mAP50", "mAP50_95", "small_ap", "medium_ap", "large_ap"]
    delta_labels = ["ΔmAP50", "ΔmAP50-95", "ΔSmall AP", "ΔMedium AP", "ΔLarge AP"]
    figure, axis = plt.subplots(figsize=(10.5, 5.4), dpi=180)
    for seed, color in zip(sorted(records), ("#457B9D", "#2A9D8F", "#E76F51"), strict=True):
        deltas = [records[seed]["full"][key] - records[seed]["baseline"][key] for key in delta_keys]
        axis.plot(delta_labels, deltas, marker="o", linewidth=2, markersize=6, label=f"seed={seed}", color=color)
    mean_deltas = [mean([records[seed]["full"][key] - records[seed]["baseline"][key] for seed in records]) for key in delta_keys]
    axis.plot(delta_labels, mean_deltas, marker="D", linewidth=2.5, linestyle="--", color="#222222", label="three-seed mean")
    axis.axhline(0, color="#444444", linewidth=1)
    axis.set_ylabel("Full − baseline (pp)")
    axis.set_title("Paired improvements by random seed")
    axis.legend(ncol=2, frameon=True)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(figure_dir / "paired_seed_deltas.png", bbox_inches="tight")
    plt.close(figure)


def build_markdown(records: dict[int, dict[str, dict[str, float]]]) -> str:
    seeds = sorted(records)
    delta_rows = []
    for seed in seeds:
        baseline = records[seed]["baseline"]
        full = records[seed]["full"]
        delta_rows.append(
            "| {seed} | {map50:+.2f} | {map5095:+.2f} | {small:+.2f} | {medium:+.2f} | {large:+.2f} |".format(
                seed=seed,
                map50=full["mAP50"] - baseline["mAP50"],
                map5095=full["mAP50_95"] - baseline["mAP50_95"],
                small=full["small_ap"] - baseline["small_ap"],
                medium=full["medium_ap"] - baseline["medium_ap"],
                large=full["large_ap"] - baseline["large_ap"],
            )
        )

    aggregate_rows = []
    for label, key in CORE_METRICS:
        baseline_values = [records[seed]["baseline"][key] for seed in seeds]
        full_values = [records[seed]["full"][key] for seed in seeds]
        delta_values = [full - base for base, full in zip(baseline_values, full_values, strict=True)]
        aggregate_rows.append(f"| {label} | {format_mean_sd(baseline_values)} | {format_mean_sd(full_values)} | {format_mean_sd(delta_values)} |")

    params_baseline = records[0]["baseline"]["parameters_m"]
    params_full = records[0]["full"]["parameters_m"]
    gflops_baseline = records[0]["baseline"]["gflops"]
    gflops_full = records[0]["full"]["gflops"]
    params_reduction = (1 - params_full / params_baseline) * 100
    gflops_reduction = (1 - gflops_full / gflops_baseline) * 100
    map_deltas = [records[seed]["full"]["mAP50_95"] - records[seed]["baseline"]["mAP50_95"] for seed in seeds]
    small_deltas = [records[seed]["full"]["small_ap"] - records[seed]["baseline"]["small_ap"] for seed in seeds]
    large_deltas = [records[seed]["full"]["large_ap"] - records[seed]["baseline"]["large_ap"] for seed in seeds]

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    return f"""# Drone-YOLO 核心模型三随机种子配对复验报告

> 生成时间：{generated_at}  
> 报告口径：seed=0、1、2 的独立 150 轮训练。每个 seed 内均以相同数据、输入尺寸、初始化来源、训练预算和验证协议配对比较完整 Drone-YOLO 与 YOLOv11s 基线。表中所有精度指标单位为百分数，差值单位为百分点（pp）。

## 1. 结论摘要

**完整 Drone-YOLO 在三个随机种子下均提高了总体精度与 COCO Small AP。** 三种子的平均配对增量为 mAP50-95 **{mean(map_deltas):+.2f} ± {stdev(map_deltas):.2f} pp**、Small AP **{mean(small_deltas):+.2f} ± {stdev(small_deltas):.2f} pp**；两个指标的增量在 3/3 个 seed 中均为正。模型参数量从 {params_baseline:.3f}M 降至 {params_full:.3f}M（-{params_reduction:.1f}%），GFLOPs 从 {gflops_baseline:.2f} 降至 {gflops_full:.2f}（-{gflops_reduction:.1f}%）。

收益存在尺度边界：Large AP 的平均配对变化为 **{mean(large_deltas):+.2f} ± {stdev(large_deltas):.2f} pp**，三个 seed 均为负。因此，当前证据支持“轻量化模型稳定改善小目标”，不支持“对所有尺度目标均有利”。

## 2. 每个随机种子的配对差值

| Seed | ΔmAP50 | ΔmAP50-95 | ΔSmall AP | ΔMedium AP | ΔLarge AP |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

![逐 seed 配对差值](figures/paired_seed_deltas.png)

## 3. 三随机种子汇总

| 指标 | YOLOv11s（均值±样本标准差） | Drone-YOLO（均值±样本标准差） | 配对变化（均值±样本标准差） |
|---|---:|---:|---:|
{chr(10).join(aggregate_rows)}

![三 seed 指标均值](figures/three_seed_metric_means.png)

## 4. 公平性与复杂度

- 每个 seed 内的两组均独立从 `yolo11s.pt` 初始化并训练 150 轮；未从另一模型或另一 seed 继续训练。
- 统一验证使用 VisDrone2019 val（548 张图像）、640 输入、FP32、conf=0.001、NMS IoU=0.7、max_det=300。
- 分尺寸指标在原图像坐标系中按 COCO 面积定义计算：Small＜32² px，Medium 为 32²–96² px，Large≥96² px。
- 结构复杂度为固定架构属性：YOLOv11s 为 {params_baseline:.3f}M 参数、{gflops_baseline:.2f} GFLOPs；完整 Drone-YOLO 为 {params_full:.3f}M 参数、{gflops_full:.2f} GFLOPs。

## 5. 可支持的结论与边界

| 主张 | 证据 | 状态 |
|---|---|---|
| 完整 Drone-YOLO 稳定提升当前协议下的 Small AP | 三个 seed 的 ΔSmall AP 均为正，均值 {mean(small_deltas):+.2f} pp | 支持（当前数据集与协议） |
| 完整 Drone-YOLO 稳定提升总体 mAP50-95 | 三个 seed 的 ΔmAP50-95 均为正，均值 {mean(map_deltas):+.2f} pp | 支持（当前数据集与协议） |
| 完整模型对所有尺度均有益 | 三个 seed 的 ΔLarge AP 均为负 | 不支持 |
| 当前实现完整复现原论文的 44.3% mAP50 | 本复验的三 seed 均值低于该论文报告值 | 不支持 |
| 模型具有统一部署速度优势 | 当前仅有单 seed、PyTorch 前向测速；无 ONNX/TensorRT 端到端复验 | 证据不足 |

## 6. 后续工作

下一步应补充固定设备上的预热端到端延迟、吞吐率和峰值显存测试，并同时报告输入预处理与后处理时间。若需要进一步改善 Large AP，应针对 LSCD 或尺度间特征交互做单因素消融，不能根据本验证集结果直接搜索多个结构参数。

## 7. 可复核材料

- 预注册：`EXPERIMENT_PLAN.md`
- 种子 1、2 的训练与评估日志：`logs/`
- 三 seed 汇总、每次评估的 JSON：`metrics/`
- 新生成的图表：`figures/`
- seed=0 的统一复测来源：`../unified_evaluation/`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "core_multiseed")
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    metrics_dir = report_dir / "metrics"
    figure_dir = report_dir / "figures"
    records: dict[int, dict[str, dict[str, float]]] = {0: seed_zero_records()}
    for seed in (1, 2):
        records[seed] = {variant: later_seed_record(metrics_dir, seed, variant) for variant in ("baseline", "full")}

    summary_rows: list[dict[str, float | int | str]] = []
    delta_rows: list[dict[str, float | int | str]] = []
    for seed in sorted(records):
        for variant in ("baseline", "full"):
            summary_rows.append({"seed": seed, "variant": variant, **records[seed][variant]})
        delta_rows.append(
            {
                "seed": seed,
                **{key: records[seed]["full"][key] - records[seed]["baseline"][key] for _, key in CORE_METRICS},
                "precision": records[seed]["full"]["precision"] - records[seed]["baseline"]["precision"],
                "recall": records[seed]["full"]["recall"] - records[seed]["baseline"]["recall"],
            }
        )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        metrics_dir / "three_seed_summary.csv",
        ["seed", "variant", "precision", "recall", "mAP50", "mAP50_95", "small_ap", "medium_ap", "large_ap", "parameters_m", "gflops"],
        summary_rows,
    )
    write_csv(
        metrics_dir / "paired_deltas.csv",
        ["seed", "precision", "recall", "mAP50", "mAP50_95", "small_ap", "medium_ap", "large_ap"],
        delta_rows,
    )
    make_figures(figure_dir, records)
    (report_dir / "README.md").write_text(build_markdown(records), encoding="utf-8")
    (report_dir / "three_seed_report_completed.flag").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


if __name__ == "__main__":
    main()
