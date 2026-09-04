"""Generate an auditable decision report for the P5-lite gated-fusion screen."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p5lite_gated_fusion"
CONTROL = ROOT / "reports" / "nwd_loss_ablation" / "metrics"
METRICS = REPORT / "metrics"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def values(standard: dict, size: dict) -> dict[str, float]:
    return {
        "mAP50": standard["overall"]["metrics/mAP50(B)"] * 100,
        "mAP50-95": standard["overall"]["metrics/mAP50-95(B)"] * 100,
        "Small AP": size["overall_by_area"]["small"]["ap"] * 100,
        "Medium AP": size["overall_by_area"]["medium"]["ap"] * 100,
        "Large AP": size["overall_by_area"]["large"]["ap"] * 100,
        "Parameters (M)": standard["model"]["parameters_m"],
        "GFLOPs": standard["model"]["gflops_640"],
    }


def main() -> None:
    control = values(load(CONTROL / "inner_wiou_val.json"), load(CONTROL / "inner_wiou_size_val.json"))
    candidate = values(load(METRICS / "p5lite_gated_val.json"), load(METRICS / "p5lite_gated_size_val.json"))
    metric_names = ["mAP50", "mAP50-95", "Small AP", "Medium AP", "Large AP"]
    delta = {name: candidate[name] - control[name] for name in metric_names}
    gate_checks = {
        "Large AP Δ ≥ +1.00 pp": delta["Large AP"] >= 1.0,
        "Small AP Δ ≥ -0.30 pp": delta["Small AP"] >= -0.3,
        "mAP50-95 Δ ≥ -0.10 pp": delta["mAP50-95"] >= -0.1,
        "Parameters ≤ 3.50M": candidate["Parameters (M)"] <= 3.5,
        "GFLOPs ≤ 22.00": candidate["GFLOPs"] <= 22.0,
    }
    passed = all(gate_checks.values())

    METRICS.mkdir(parents=True, exist_ok=True)
    figures = REPORT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (METRICS / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "full_50e_control", "p5lite_gated", "delta_pp"])
        for name in metric_names:
            writer.writerow([name, f"{control[name]:.4f}", f"{candidate[name]:.4f}", f"{delta[name]:+.4f}"])
        writer.writerow(["Parameters (M)", f"{control['Parameters (M)']:.6f}", f"{candidate['Parameters (M)']:.6f}", f"{candidate['Parameters (M)'] - control['Parameters (M)']:+.6f}"])
        writer.writerow(["GFLOPs", f"{control['GFLOPs']:.6f}", f"{candidate['GFLOPs']:.6f}", f"{candidate['GFLOPs'] - control['GFLOPs']:+.6f}"])

    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    indexes = list(range(len(metric_names)))
    width = 0.36
    axis.bar([index - width / 2 for index in indexes], [control[name] for name in metric_names], width, label="Full control")
    axis.bar([index + width / 2 for index in indexes], [candidate[name] for name in metric_names], width, label="P5-lite + gate")
    axis.set_xticks(indexes, metric_names, rotation=18)
    axis.set_ylabel("Score (%)")
    axis.set_title("P5-lite gated-fusion: fixed 50-epoch screen")
    axis.legend()
    figure.savefig(figures / "p5lite_gated_metrics.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    colours = ["#2f855a" if delta[name] >= 0 else "#c53030" for name in metric_names]
    bars = axis.bar(metric_names, [delta[name] for name in metric_names], color=colours)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Candidate - Full control (pp)")
    axis.set_title("P5-lite gated-fusion metric changes")
    axis.tick_params(axis="x", rotation=18)
    for bar, name in zip(bars, metric_names):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{delta[name]:+.2f}",
            ha="center",
            va="bottom" if delta[name] >= 0 else "top",
        )
    figure.savefig(figures / "p5lite_gated_deltas.png", dpi=180)
    plt.close(figure)

    rows = "\n".join(
        f"| {name} | {control[name]:.2f}% | {candidate[name]:.2f}% | {delta[name]:+.2f} |" for name in metric_names
    )
    check_rows = "\n".join(
        f"| {name} | {'通过' if value else '未通过'} |" for name, value in gate_checks.items()
    )
    decision = (
        "通过筛选：进入150轮、三随机种子复验，并分解 P5-lite 与空间门控的贡献。"
        if passed
        else "未通过筛选：停止此精确配置；不将个别指标改善解释为整体有效。"
    )
    report = f"""# P5-lite 语义反馈与空间门控：50轮筛选报告

> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  
> 候选：P5-lite 内部语义反馈 + 像素级空间门控；检测层仍为 P2/P3/P4。

## 1. 固定协议比较

| 指标 | Full 50轮对照 | P5-lite + gate | Δ（百分点） |
|---|---:|---:|---:|
{rows}

| 推理结构 | Full 50轮对照 | P5-lite + gate | 差值 |
|---|---:|---:|---:|
| 参数量（M） | {control['Parameters (M)']:.3f} | {candidate['Parameters (M)']:.3f} | {candidate['Parameters (M)'] - control['Parameters (M)']:+.3f} |
| GFLOPs | {control['GFLOPs']:.2f} | {candidate['GFLOPs']:.2f} | {candidate['GFLOPs'] - control['GFLOPs']:+.2f} |

![主要指标](figures/p5lite_gated_metrics.png)

![相对变化](figures/p5lite_gated_deltas.png)

## 2. 预注册判定

| 条件 | 结果 |
|---|---|
{check_rows}

**结论：{decision}**

## 3. 解释边界与下一步

本结果只检验 seed=0、50轮、128通道 P5-lite、初始 gate bias=-2 的组合。通过门槛后仍需使用更长训练和三个随机种子验证稳定性；未通过只能说明该精确的反馈路径与门控设置未同时满足“补大目标且不伤小目标”的目标，不能外推为所有跨尺度语义反馈都无效。

## 4. 可追溯产物

- 预注册方案：`EXPERIMENT_PLAN.md`
- 训练、验证、报告日志：`logs/`
- 结构化指标及对照表：`metrics/`
- 学生检查点：`../../runs/p5lite_gated_seed0_50e/weights/best.pt`
"""
    (REPORT / "README.md").write_text(report, encoding="utf-8")
    print(f"P5-lite gated-fusion report generated; passed={passed}")


if __name__ == "__main__":
    main()
