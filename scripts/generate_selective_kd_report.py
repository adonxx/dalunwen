"""Generate the auditable report for the pre-registered selective-KD screen."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "selective_kd"
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
    candidate = values(load(METRICS / "selective_kd_val.json"), load(METRICS / "selective_kd_size_val.json"))
    metrics = ["mAP50", "mAP50-95", "Small AP", "Medium AP", "Large AP"]
    delta = {key: candidate[key] - control[key] for key in metrics}
    passed = (
        delta["Large AP"] >= 1.0
        and delta["Small AP"] >= -0.3
        and delta["mAP50-95"] >= -0.1
        and abs(candidate["Parameters (M)"] - control["Parameters (M)"]) < 0.001
        and abs(candidate["GFLOPs"] - control["GFLOPs"]) < 0.001
    )

    METRICS.mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    with (METRICS / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "control", "selective_kd", "delta_pp"])
        for key in metrics:
            writer.writerow([key, f"{control[key]:.4f}", f"{candidate[key]:.4f}", f"{delta[key]:+.4f}"])
        writer.writerow(["Parameters (M)", f"{control['Parameters (M)']:.6f}", f"{candidate['Parameters (M)']:.6f}", "0.000000"])
        writer.writerow(["GFLOPs", f"{control['GFLOPs']:.6f}", f"{candidate['GFLOPs']:.6f}", "0.000000"])

    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = list(range(len(metrics)))
    width = 0.36
    axis.bar([i - width / 2 for i in x], [control[key] for key in metrics], width, label="Full control")
    axis.bar([i + width / 2 for i in x], [candidate[key] for key in metrics], width, label="Selective KD")
    axis.set_xticks(x, metrics, rotation=18)
    axis.set_ylabel("Score (%)")
    axis.set_title("50-epoch selective-KD screen")
    axis.legend()
    figure.savefig(REPORT / "figures" / "selective_kd_metrics.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    colours = ["#2f855a" if delta[key] >= 0 else "#c53030" for key in metrics]
    bars = axis.bar(metrics, [delta[key] for key in metrics], color=colours)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Selective KD - control (pp)")
    axis.set_title("Metric changes under the fixed screening protocol")
    axis.tick_params(axis="x", rotation=18)
    for bar, key in zip(bars, metrics):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{delta[key]:+.2f}", ha="center", va="bottom" if delta[key] >= 0 else "top")
    figure.savefig(REPORT / "figures" / "selective_kd_deltas.png", dpi=180)
    plt.close(figure)

    rows = "\n".join(
        f"| {key} | {control[key]:.2f}% | {candidate[key]:.2f}% | {delta[key]:+.2f} |" for key in metrics
    )
    decision = "通过筛选，可进入150轮和三随机种子复验。" if passed else "未通过筛选；停止该精确配置，不进入长训。"
    report = f"""# 尺度选择性知识蒸馏：50轮筛选报告

> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  
> 学生为完整 Drone-YOLO；教师为150轮 YOLOv11s 基线。教师和适配器仅在训练期存在，验证使用保存后的学生模型。

## 1. 统一对比

| 指标 | Full 对照 | Selective KD | Δ（百分点） |
|---|---:|---:|---:|
{rows}

| 推理结构 | Full 对照 | Selective KD |
|---|---:|---:|
| 参数量（M） | {control['Parameters (M)']:.3f} | {candidate['Parameters (M)']:.3f} |
| GFLOPs | {control['GFLOPs']:.2f} | {candidate['GFLOPs']:.2f} |

![主要指标](figures/selective_kd_metrics.png)

![相对变化](figures/selective_kd_deltas.png)

## 2. 预注册判定

| 条件 | 实测 | 是否通过 |
|---|---:|---:|
| Large AP Δ ≥ +1.00 pp | {delta['Large AP']:+.2f} pp | {'是' if delta['Large AP'] >= 1.0 else '否'} |
| Small AP Δ ≥ -0.30 pp | {delta['Small AP']:+.2f} pp | {'是' if delta['Small AP'] >= -0.3 else '否'} |
| mAP50-95 Δ ≥ -0.10 pp | {delta['mAP50-95']:+.2f} pp | {'是' if delta['mAP50-95'] >= -0.1 else '否'} |
| 参数量、GFLOPs不变 | 一致 | {'是' if abs(candidate['Parameters (M)'] - control['Parameters (M)']) < 0.001 and abs(candidate['GFLOPs'] - control['GFLOPs']) < 0.001 else '否'} |

**结论：{decision}**

## 3. 解释边界

该结果仅检验“基线教师、P3/P4对齐、32²面积掩码、权重0.5”的组合。若未通过，不能外推为所有知识蒸馏方法均无效；更可能的原因包括教师总体精度不足、特征约束与当前检测损失竞争，或中大目标区域掩码过于稀疏/粗糙。

## 4. 可追溯产物

- 预注册方案：`EXPERIMENT_PLAN.md`
- 训练、验证和报告日志：`logs/`
- 结构化指标与比较表：`metrics/`
- 学生检查点：`../../runs/selective_kd_p34_seed0_50e/weights/best.pt`
"""
    (REPORT / "README.md").write_text(report, encoding="utf-8")
    print(f"Selective-KD report generated; passed={passed}")


if __name__ == "__main__":
    main()
