"""Generate the comparison report for the small-object crop screening experiment."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CONTROL_METRICS = ROOT / "reports" / "nwd_loss_ablation" / "metrics" / "inner_wiou_size_val.json"
CONTROL_RESULTS = ROOT / "runs" / "loss_inner_wiou_50e" / "results.csv"
TREATMENT_RESULTS = ROOT / "runs" / "small_crop_50e" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "small_object_crop")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def load_results(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {key.strip(): [float(row[key]) for row in rows] for key in rows[0]}


def pct(value: float) -> float:
    return value * 100.0


def save_figures(
    control: dict[str, float],
    treatment: dict[str, float],
    control_curve: dict[str, list[float]],
    treatment_curve: dict[str, list[float]],
    figure_dir: Path,
) -> None:
    labels = ["mAP50-95", "Small AP", "Medium AP", "Large AP"]
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    left = ax.bar([i - width / 2 for i in x], [control[k] for k in labels], width, label="Full control")
    right = ax.bar(
        [i + width / 2 for i in x], [treatment[k] for k in labels], width, label="Small-object crop"
    )
    ax.bar_label(left, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(right, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylabel("AP (%)")
    ax.set_xticks(x, labels)
    ax.set_title("50-epoch validation comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for curve, label in ((control_curve, "Full control"), (treatment_curve, "Small-object crop")):
        axes[0].plot(curve["epoch"], curve["train/box_loss"], label=label)
        axes[1].plot(curve["epoch"], [pct(v) for v in curve["metrics/mAP50-95(B)"]], label=label)
    axes[0].set_title("Training box loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Built-in validation mAP50-95")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP50-95 (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    metric_dir = report_dir / "metrics"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    control_raw = load_json(CONTROL_METRICS)
    treatment_raw = load_json(metric_dir / "small_crop_size_val.json")

    def extract(raw: dict) -> dict[str, float]:
        return {
            "mAP50-95": pct(raw["ultralytics_overall"]["metrics/mAP50-95(B)"]),
            "AP50": pct(raw["ultralytics_overall"]["metrics/mAP50(B)"]),
            "Precision": pct(raw["ultralytics_overall"]["metrics/precision(B)"]),
            "Recall": pct(raw["ultralytics_overall"]["metrics/recall(B)"]),
            "Small AP": pct(raw["overall_by_area"]["small"]["ap"]),
            "Medium AP": pct(raw["overall_by_area"]["medium"]["ap"]),
            "Large AP": pct(raw["overall_by_area"]["large"]["ap"]),
        }

    control, treatment = extract(control_raw), extract(treatment_raw)
    deltas = {key: treatment[key] - control[key] for key in control}
    conditions = {
        "Small AP ≥ +0.50 pp": deltas["Small AP"] >= 0.50,
        "mAP50-95 ≥ -0.20 pp": deltas["mAP50-95"] >= -0.20,
        "Medium AP ≥ -0.50 pp": deltas["Medium AP"] >= -0.50,
        "Large AP ≥ -0.50 pp": deltas["Large AP"] >= -0.50,
    }
    passed = all(conditions.values())

    save_figures(control, treatment, load_results(CONTROL_RESULTS), load_results(TREATMENT_RESULTS), figure_dir)
    smoke_batch = ROOT / "runs" / "smoke_small_crop_1e" / "train_batch0.jpg"
    if smoke_batch.exists():
        shutil.copy2(smoke_batch, figure_dir / "crop_training_batch_example.jpg")

    metric_order = ["mAP50-95", "AP50", "Precision", "Recall", "Small AP", "Medium AP", "Large AP"]
    rows = "\n".join(
        f"| {name} | {control[name]:.4f} | {treatment[name]:.4f} | {deltas[name]:+.4f} |"
        for name in metric_order
    )
    checks = "\n".join(f"- {'通过' if ok else '未通过'}：{name}" for name, ok in conditions.items())
    conclusion = (
        "本配置通过筛选，下一步使用 seed 1/2 重复 50 轮实验。"
        if passed
        else "本配置未同时满足筛选门槛，应停止该固定配置，下一步转向固定训练步数的重叠切片训练。"
    )
    report = f"""# 小目标中心裁剪训练：50 轮筛选报告

## 实验结论

**{'通过' if passed else '未通过'}预注册筛选标准。** {conclusion}

## 指标对比

所有指标均为百分数，差值单位为百分点（pp）。两组采用相同 Full 模型、初始化、训练步数与验证协议，唯一差异是实验组训练读取时有 50% 概率执行小目标中心裁剪。

| 指标 | Full 对照 | 小目标中心裁剪 | 差值 |
|---|---:|---:|---:|
{rows}

## 判据核验

{checks}

![核心指标对比](figures/metric_comparison.png)

![训练曲线](figures/training_curves.png)

## 方法与可视化

裁剪宽高均为原图的 50%，保证选中的输入空间小目标完整保留，再等比例放大回原尺寸。该操作只用于训练，不改变推理图。

![训练批次示例](figures/crop_training_batch_example.jpg)

## 可复核材料

- 预注册：`reports/small_object_crop/EXPERIMENT_PLAN.md`
- 实验组检查点：`runs/small_crop_50e/weights/best.pt`
- 总体指标：`reports/small_object_crop/metrics/small_crop_val.json`
- 尺寸指标：`reports/small_object_crop/metrics/small_crop_size_val.json`
- 训练日志：`reports/small_object_crop/logs/small_crop_train.log`
"""
    (report_dir / "REPORT.md").write_text(report, encoding="utf-8")
    (metric_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "control_percent": control,
                "treatment_percent": treatment,
                "delta_percentage_points": deltas,
                "conditions": conditions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
