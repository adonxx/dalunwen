"""Generate the fixed-protocol report for the 150-epoch P5-lite run."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p5lite_gated_fusion"
METRICS = REPORT / "metrics"
CONTROL_METRICS = ROOT / "reports" / "unified_evaluation" / "metrics"
CONTROL_RUN = ROOT / "runs" / "full_paper" / "results.csv"
CANDIDATE_RUN = ROOT / "runs" / "p5lite_gated_seed0_150e" / "results.csv"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(standard: dict, size: dict) -> dict[str, float]:
    return {
        "mAP50": standard["overall"]["metrics/mAP50(B)"] * 100,
        "mAP50-95": standard["overall"]["metrics/mAP50-95(B)"] * 100,
        "Small AP": size["overall_by_area"]["small"]["ap"] * 100,
        "Medium AP": size["overall_by_area"]["medium"]["ap"] * 100,
        "Large AP": size["overall_by_area"]["large"]["ap"] * 100,
        "Parameters (M)": standard["model"]["parameters_m"],
        "GFLOPs": standard["model"]["gflops_640"],
    }


def load_curve(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = []
        for raw in csv.DictReader(stream):
            row = {key.strip(): float(value) for key, value in raw.items() if key is not None and value != ""}
            rows.append(row)
    return rows


def main() -> None:
    control = metric_values(
        load_json(CONTROL_METRICS / "full_val.json"),
        load_json(CONTROL_METRICS / "full_size_val.json"),
    )
    candidate = metric_values(
        load_json(METRICS / "p5lite_gated_150e_val.json"),
        load_json(METRICS / "p5lite_gated_150e_size_val.json"),
    )
    names = ["mAP50", "mAP50-95", "Small AP", "Medium AP", "Large AP"]
    delta = {name: candidate[name] - control[name] for name in names}
    checks = {
        "Large AP Δ ≥ +1.00 pp": delta["Large AP"] >= 1.0,
        "Small AP Δ ≥ -0.30 pp": delta["Small AP"] >= -0.3,
        "mAP50-95 Δ ≥ -0.10 pp": delta["mAP50-95"] >= -0.1,
        "Parameters ≤ 3.50M": candidate["Parameters (M)"] <= 3.5,
        "GFLOPs ≤ 22.00": candidate["GFLOPs"] <= 22.0,
    }
    passed = all(checks.values())

    figures = REPORT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    with (METRICS / "comparison_150e.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "full_seed0_150e", "p5lite_gated_seed0_150e", "delta"])
        for name in names:
            writer.writerow([name, f"{control[name]:.4f}", f"{candidate[name]:.4f}", f"{delta[name]:+.4f}"])
        for name in ("Parameters (M)", "GFLOPs"):
            writer.writerow([name, f"{control[name]:.6f}", f"{candidate[name]:.6f}", f"{candidate[name] - control[name]:+.6f}"])

    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    indexes = list(range(len(names)))
    width = 0.36
    axis.bar([index - width / 2 for index in indexes], [control[name] for name in names], width, label="Full 150e")
    axis.bar([index + width / 2 for index in indexes], [candidate[name] for name in names], width, label="P5-lite + gate 150e")
    axis.set_xticks(indexes, names, rotation=18)
    axis.set_ylabel("Score (%)")
    axis.set_title("Matched seed-0, 150-epoch comparison")
    axis.legend()
    figure.savefig(figures / "p5lite_gated_150e_metrics.png", dpi=180)
    plt.close(figure)

    control_curve = load_curve(CONTROL_RUN)
    candidate_curve = load_curve(CANDIDATE_RUN)
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    axis.plot([row["epoch"] for row in control_curve], [100 * row["metrics/mAP50-95(B)"] for row in control_curve], label="Full 150e")
    axis.plot([row["epoch"] for row in candidate_curve], [100 * row["metrics/mAP50-95(B)"] for row in candidate_curve], label="P5-lite + gate 150e")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation mAP50-95 (%)")
    axis.set_title("Convergence under the matched training budget")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(figures / "p5lite_gated_150e_convergence.png", dpi=180)
    plt.close(figure)

    best_row = max(candidate_curve, key=lambda row: row["metrics/mAP50-95(B)"])
    rows = "\n".join(
        f"| {name} | {control[name]:.2f}% | {candidate[name]:.2f}% | {delta[name]:+.2f} |" for name in names
    )
    check_rows = "\n".join(f"| {name} | {'通过' if value else '未通过'} |" for name, value in checks.items())
    decision = (
        "通过：补做 seed=1、2，并在三随机种子上验证总体与尺度收益。"
        if passed
        else "未通过：停止当前全尺度反馈配置，转入仅增强最终 P4 分支的结构实验。"
    )
    report = f"""# P5-lite 空间门控：150轮探索性复验报告

> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  
> 两组均为 seed=0、从 yolo11s.pt 独立初始化、训练150轮；本实验不改变50轮筛选未通过的历史结论。

## 1. 统一验证结果

| 指标 | Full 150轮对照 | P5-lite + gate 150轮 | Δ（百分点） |
|---|---:|---:|---:|
{rows}

| 推理结构 | Full | P5-lite + gate | 差值 |
|---|---:|---:|---:|
| 参数量（M） | {control['Parameters (M)']:.3f} | {candidate['Parameters (M)']:.3f} | {candidate['Parameters (M)'] - control['Parameters (M)']:+.3f} |
| GFLOPs | {control['GFLOPs']:.2f} | {candidate['GFLOPs']:.2f} | {candidate['GFLOPs'] - control['GFLOPs']:+.2f} |

候选训练日志中的最高 mAP50-95 出现在第 {int(best_row['epoch']) + 1} 轮，为 {100 * best_row['metrics/mAP50-95(B)']:.2f}%。最终结论使用统一加载 `best.pt` 后的独立验证结果。

![150轮指标比较](figures/p5lite_gated_150e_metrics.png)

![150轮收敛曲线](figures/p5lite_gated_150e_convergence.png)

## 2. 预注册判定

| 条件 | 结果 |
|---|---|
{check_rows}

**结论：{decision}**

## 3. 证据边界

本报告是针对50轮临界候选追加的探索性长训，仅包含一个随机种子。即使通过，也只能说明该候选值得进入多随机种子复验；在 seed=1、2 完成前，不能声称性能提升稳定。若未通过，只否定当前128通道、全尺度传播和固定门控初始化的精确组合。

## 4. 可追溯产物

- 长训预注册：`LONGRUN_PLAN.md`
- 训练、验证、报告日志：`logs/`
- 结构化结果和对比表：`metrics/`
- 候选检查点：`../../runs/p5lite_gated_seed0_150e/weights/best.pt`
"""
    (REPORT / "LONGRUN_REPORT.md").write_text(report, encoding="utf-8")
    print(f"P5-lite 150-epoch report generated; passed={passed}")


if __name__ == "__main__":
    main()
