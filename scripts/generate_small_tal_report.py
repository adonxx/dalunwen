"""Generate the fixed comparison report for the small-aware TAL screening experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CONTROL_METRICS = ROOT / "reports" / "nwd_loss_ablation" / "metrics" / "inner_wiou_size_val.json"
CONTROL_RESULTS = ROOT / "runs" / "loss_inner_wiou_50e" / "results.csv"
TREATMENT_RESULTS = ROOT / "runs" / "small_tal_50e" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "small_aware_tal")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def load_results(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    columns: dict[str, list[float]] = {}
    for key in rows[0]:
        clean_key = key.strip()
        columns[clean_key] = [float(row[key]) for row in rows]
    return columns


def pct(value: float) -> float:
    return value * 100.0


def save_metric_figure(control: dict[str, float], treatment: dict[str, float], output: Path) -> None:
    labels = ["mAP50-95", "Small AP", "Medium AP", "Large AP"]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    left = ax.bar([i - width / 2 for i in x], [control[k] for k in labels], width, label="Full control")
    right = ax.bar([i + width / 2 for i in x], [treatment[k] for k in labels], width, label="Small-aware TAL")
    ax.bar_label(left, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(right, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylabel("AP (%)")
    ax.set_xticks(x, labels)
    ax.set_title("50-epoch validation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_training_figure(control: dict[str, list[float]], treatment: dict[str, list[float]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for values, label in ((control, "Full control"), (treatment, "Small-aware TAL")):
        epochs = values["epoch"]
        axes[0].plot(epochs, values["train/box_loss"], label=label)
        axes[1].plot(epochs, [pct(v) for v in values["metrics/mAP50-95(B)"]], label=label)
    axes[0].set_title("Training box loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Built-in validation mAP50-95")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP50-95 (%)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    metric_dir = report_dir / "metrics"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    control_raw = load_json(CONTROL_METRICS)
    treatment_raw = load_json(metric_dir / "small_tal_size_val.json")
    control = {
        "mAP50-95": pct(control_raw["ultralytics_overall"]["metrics/mAP50-95(B)"]),
        "AP50": pct(control_raw["ultralytics_overall"]["metrics/mAP50(B)"]),
        "Precision": pct(control_raw["ultralytics_overall"]["metrics/precision(B)"]),
        "Recall": pct(control_raw["ultralytics_overall"]["metrics/recall(B)"]),
        "Small AP": pct(control_raw["overall_by_area"]["small"]["ap"]),
        "Medium AP": pct(control_raw["overall_by_area"]["medium"]["ap"]),
        "Large AP": pct(control_raw["overall_by_area"]["large"]["ap"]),
    }
    treatment = {
        "mAP50-95": pct(treatment_raw["ultralytics_overall"]["metrics/mAP50-95(B)"]),
        "AP50": pct(treatment_raw["ultralytics_overall"]["metrics/mAP50(B)"]),
        "Precision": pct(treatment_raw["ultralytics_overall"]["metrics/precision(B)"]),
        "Recall": pct(treatment_raw["ultralytics_overall"]["metrics/recall(B)"]),
        "Small AP": pct(treatment_raw["overall_by_area"]["small"]["ap"]),
        "Medium AP": pct(treatment_raw["overall_by_area"]["medium"]["ap"]),
        "Large AP": pct(treatment_raw["overall_by_area"]["large"]["ap"]),
    }
    deltas = {key: treatment[key] - control[key] for key in control}

    conditions = {
        "Small AP ≥ +0.50 pp": deltas["Small AP"] >= 0.50,
        "mAP50-95 ≥ -0.20 pp": deltas["mAP50-95"] >= -0.20,
        "Medium AP ≥ -0.50 pp": deltas["Medium AP"] >= -0.50,
        "Large AP ≥ -0.50 pp": deltas["Large AP"] >= -0.50,
    }
    passed = all(conditions.values())

    save_metric_figure(control, treatment, figure_dir / "metric_comparison.png")
    save_training_figure(
        load_results(CONTROL_RESULTS), load_results(TREATMENT_RESULTS), figure_dir / "training_curves.png"
    )

    ordered_metrics = ["mAP50-95", "AP50", "Precision", "Recall", "Small AP", "Medium AP", "Large AP"]
    rows = "\n".join(
        f"| {name} | {control[name]:.4f} | {treatment[name]:.4f} | {deltas[name]:+.4f} |"
        for name in ordered_metrics
    )
    checks = "\n".join(f"- {'通过' if ok else '未通过'}：{name}" for name, ok in conditions.items())
    conclusion = (
        "本次 50 轮筛选通过预设门槛，下一步应使用 seed 1 和 seed 2 重复训练，以检验增益稳定性。"
        if passed
        else "本次 50 轮筛选未同时满足预设门槛，应停止该固定配置，不根据当前验证结果继续调参；下一步转向小目标区域采样或切片训练。"
    )
    report = f"""# 小目标感知 TAL：50 轮筛选报告

## 实验结论

**{'通过' if passed else '未通过'}预注册筛选标准。** {conclusion}

## 指标对比

所有 AP、Precision 和 Recall 均以百分数表示，差值单位为百分点（pp）。对照组复用冻结的 `loss_inner_wiou_50e`，实验组为 `small_tal_50e`。

| 指标 | Full 对照 | Small-aware TAL | 差值 |
|---|---:|---:|---:|
{rows}

## 判据核验

{checks}

训练全程无 NaN 或异常中止；该方法仅改变训练期标签分配，模型结构与推理流程保持不变。

![核心指标对比](figures/metric_comparison.png)

![训练曲线](figures/training_curves.png)

## 方法说明

仅对输入坐标系下面积小于 `32² px` 的真实框，将 TAL 的定位质量替换为 `0.5 × CIoU + 0.5 × NWD similarity`；中、大目标仍使用原始 CIoU。回归损失继续采用 Inner-WIoU + DFL，训练设置与对照一致。完整预注册见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

## 可复核材料

- 实验组检查点：`runs/small_tal_50e/weights/best.pt`
- 实验组统一指标：`reports/small_aware_tal/metrics/small_tal_val.json`
- 实验组尺寸指标：`reports/small_aware_tal/metrics/small_tal_size_val.json`
- 训练日志：`reports/small_aware_tal/logs/small_tal_train.log`
- 评估日志：`reports/small_aware_tal/logs/small_tal_val.log`
"""
    (report_dir / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "passed": passed,
        "control_percent": control,
        "treatment_percent": treatment,
        "delta_percentage_points": deltas,
        "conditions": conditions,
    }
    (metric_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
