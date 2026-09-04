"""Generate a fixed-protocol report for a P4-only P5-lite experiment."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p5lite_p4only"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, choices=(50, 150), required=True)
    args = parser.parse_args()
    if args.epochs == 50:
        control_root = ROOT / "reports" / "nwd_loss_ablation" / "metrics"
        control_standard = control_root / "inner_wiou_val.json"
        control_size = control_root / "inner_wiou_size_val.json"
        report_name = "SCREEN_REPORT.md"
    else:
        control_root = ROOT / "reports" / "unified_evaluation" / "metrics"
        control_standard = control_root / "full_val.json"
        control_size = control_root / "full_size_val.json"
        report_name = "LONGRUN_REPORT.md"

    tag = f"{args.epochs}e"
    control = values(load(control_standard), load(control_size))
    candidate = values(
        load(METRICS / f"p5lite_p4only_{tag}_val.json"),
        load(METRICS / f"p5lite_p4only_{tag}_size_val.json"),
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
    with (METRICS / f"comparison_{tag}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", f"full_{tag}_control", f"p5lite_p4only_{tag}", "delta"])
        for name in names:
            writer.writerow([name, f"{control[name]:.4f}", f"{candidate[name]:.4f}", f"{delta[name]:+.4f}"])
        for name in ("Parameters (M)", "GFLOPs"):
            writer.writerow([name, f"{control[name]:.6f}", f"{candidate[name]:.6f}", f"{candidate[name] - control[name]:+.6f}"])

    decision = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "epochs": args.epochs,
        "passed": passed,
        "checks": checks,
        "control": control,
        "candidate": candidate,
        "delta": delta,
    }
    (METRICS / f"decision_{tag}.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.rcParams["axes.unicode_minus"] = False
    indexes = list(range(len(names)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    axis.bar([i - width / 2 for i in indexes], [control[name] for name in names], width, label=f"Full {tag}")
    axis.bar([i + width / 2 for i in indexes], [candidate[name] for name in names], width, label=f"P4-only {tag}")
    axis.set_xticks(indexes, names, rotation=18)
    axis.set_ylabel("Score (%)")
    axis.set_title(f"P5-lite P4-only: matched {args.epochs}-epoch comparison")
    axis.legend()
    figure.savefig(figures / f"p5lite_p4only_metrics_{tag}.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    colours = ["#2f855a" if delta[name] >= 0 else "#c53030" for name in names]
    bars = axis.bar(names, [delta[name] for name in names], color=colours)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("P4-only - Full control (pp)")
    axis.set_title(f"Metric changes after {args.epochs} epochs")
    axis.tick_params(axis="x", rotation=18)
    for bar, name in zip(bars, names):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{delta[name]:+.2f}", ha="center", va="bottom" if delta[name] >= 0 else "top")
    figure.savefig(figures / f"p5lite_p4only_deltas_{tag}.png", dpi=180)
    plt.close(figure)

    rows = "\n".join(
        f"| {name} | {control[name]:.2f}% | {candidate[name]:.2f}% | {delta[name]:+.2f} |" for name in names
    )
    check_rows = "\n".join(f"| {name} | {'通过' if result else '未通过'} |" for name, result in checks.items())
    next_step = (
        "50轮通过，按预注册方案启动同结构150轮独立复验。"
        if passed and args.epochs == 50
        else "150轮通过，可进入 seed=1、2 稳定性复验。"
        if passed
        else "未通过，停止当前精确配置，不进入下一阶段。"
    )
    report = f"""# P5-lite 仅增强 P4：{args.epochs}轮实验报告

> 生成时间：{decision['generated_at']}  
> 候选仅改变最终P4检测输入；P2/P3直接特征路径保持Full模型不变。

## 统一指标

| 指标 | Full {args.epochs}轮对照 | P4-only候选 | Δ（百分点） |
|---|---:|---:|---:|
{rows}

| 结构指标 | Full | P4-only | 差值 |
|---|---:|---:|---:|
| 参数量（M） | {control['Parameters (M)']:.3f} | {candidate['Parameters (M)']:.3f} | {candidate['Parameters (M)'] - control['Parameters (M)']:+.3f} |
| GFLOPs | {control['GFLOPs']:.2f} | {candidate['GFLOPs']:.2f} | {candidate['GFLOPs'] - control['GFLOPs']:+.2f} |

![指标比较](figures/p5lite_p4only_metrics_{tag}.png)

![指标变化](figures/p5lite_p4only_deltas_{tag}.png)

## 预注册判断

| 条件 | 结果 |
|---|---|
{check_rows}

**结论：{next_step}**

## 解释边界

该结果只检验128隐藏通道、gate bias=-2和当前LSCD共享检测头。P2/P3没有直接接收P5-lite特征，但三个尺度仍通过共享检测头参数间接耦合。单随机种子结果只能用于阶段筛选，不能直接表述为稳定提升。
"""
    (REPORT / report_name).write_text(report, encoding="utf-8")
    print(f"P4-only {tag} report generated; passed={passed}")


if __name__ == "__main__":
    main()
