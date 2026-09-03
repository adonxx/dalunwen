"""Generate accuracy, efficiency, and gating visualizations for density-gated SAHI."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "density_gated_sahi"
METRICS = REPORT / "metrics"
FIGURES = REPORT / "figures"
GATES = (("top1", "Top-1"), ("gate_q20", "Gate-Q20"), ("gate_q30", "Gate-Q30"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def fmt_delta(value: float) -> str:
    return f"{100 * value:+.2f}"


def audit(record: dict[str, Any]) -> dict[str, int]:
    gt = load_json(Path(record["ground_truth_json"]))
    predictions = load_json(Path(record["predictions_json"]))
    shapes = {image["id"]: (image["width"], image["height"]) for image in gt["images"]}
    counts: dict[int | str, int] = {}
    invalid = 0
    for prediction in predictions:
        current_id = prediction["image_id"]
        counts[current_id] = counts.get(current_id, 0) + 1
        x, y, width, height = [float(value) for value in prediction["bbox"]]
        image_width, image_height = shapes[current_id]
        if x < -1e-3 or y < -1e-3 or width <= 0 or height <= 0 or x + width > image_width + 1e-3 or y + height > image_height + 1e-3:
            invalid += 1
    return {
        "images": len(counts),
        "predictions": len(predictions),
        "max_per_image": max(counts.values()),
        "invalid": invalid,
    }


def make_row(key: str, label: str, record: dict[str, Any], source: str) -> dict[str, Any]:
    areas = record["overall_by_area"]
    efficiency = record["efficiency"]
    return {
        "key": key,
        "method": label,
        "source": source,
        "all_ap": areas["all"]["ap"],
        "all_ap50": areas["all"]["ap50"],
        "small_ap": areas["small"]["ap"],
        "small_ap50": areas["small"]["ap50"],
        "small_ar": areas["small"]["ar_max_det"],
        "medium_ap": areas["medium"]["ap"],
        "large_ap": areas["large"]["ap"],
        "latency_ms": efficiency["mean_end_to_end_ms"],
        "p95_latency_ms": efficiency["p95_end_to_end_ms"],
        "fps": efficiency["fps_from_mean_end_to_end"],
        "trigger_rate": efficiency.get("gate_trigger_rate", 1.0 if key == "full_scale48" else 0.0),
        "mean_selected_tiles": efficiency.get("mean_selected_tiles", efficiency.get("mean_tiles_per_image", 0.0)),
        "tile_reduction_ratio": efficiency.get("tile_reduction_ratio", 0.0),
        "peak_gpu_memory_mb": efficiency["peak_gpu_memory_mb"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_latency(rows: list[dict[str, Any]], full: dict[str, Any]) -> None:
    colors = ("#4263EB", "#F59F00", "#E64980", "#12B886", "#9775FA")
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for row, color in zip(rows, colors, strict=True):
        ax.scatter(row["latency_ms"], 100 * row["small_ap"], s=115, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate(row["method"], (row["latency_ms"], 100 * row["small_ap"]), xytext=(6, 5), textcoords="offset points")
    ax.axvline(0.70 * full["latency_ms"], color="#C92A2A", linestyle="--", linewidth=1.2, label="30% latency reduction target")
    ax.axhline(100 * (full["small_ap"] - 0.005), color="#495057", linestyle="--", linewidth=1.2, label="Small AP retention target")
    ax.set_xlabel("Mean end-to-end latency (ms/image; lower is better)")
    ax.set_ylabel("Small AP (%; higher is better)")
    ax.set_title("Density-gated SAHI accuracy–latency trade-off")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "gate_accuracy_latency.png", dpi=180)
    plt.close(fig)


def plot_metrics(rows: list[dict[str, Any]]) -> None:
    labels = ("AP50-95", "Small AP", "Medium AP", "Large AP")
    x = np.arange(len(labels))
    width = 0.15
    colors = ("#4263EB", "#F59F00", "#E64980", "#12B886", "#9775FA")
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    for index, (row, color) in enumerate(zip(rows, colors, strict=True)):
        values = 100 * np.asarray([row["all_ap"], row["small_ap"], row["medium_ap"], row["large_ap"]])
        bars = ax.bar(x + (index - 2) * width, values, width, label=row["method"], color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("COCO metric (%)")
    ax.set_title("Full Scale-48 and prediction-density gate ablation")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate_metrics.png", dpi=180)
    plt.close(fig)


def plot_efficiency(rows: list[dict[str, Any]]) -> None:
    gate_rows = rows[1:]
    labels = [row["method"] for row in gate_rows]
    trigger = [100 * row["trigger_rate"] for row in gate_rows]
    reduction = [100 * row["tile_reduction_ratio"] for row in gate_rows]
    latency = [row["latency_ms"] for row in gate_rows]
    x = np.arange(len(gate_rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    first = ax.bar(x - width, trigger, width, label="Gate trigger rate (%)", color="#4263EB")
    second = ax.bar(x, reduction, width, label="Tile reduction (%)", color="#12B886")
    third = ax.bar(x + width, latency, width, label="Latency (ms/image)", color="#F59F00")
    for bars in (first, second, third):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent or milliseconds")
    ax.set_title("Gate coverage, tile reduction, and measured latency")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate_efficiency.png", dpi=180)
    plt.close(fig)


def plot_calibration(calibration: dict[str, Any]) -> None:
    values = [float(row["max_density_score"]) for row in calibration["per_image"]]
    q20 = calibration["quantiles"]["q20"]
    q30 = calibration["quantiles"]["q30"]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.hist(values, bins=32, color="#74C0FC", edgecolor="white")
    ax.axvline(q20, color="#E64980", linestyle="--", linewidth=1.5, label=f"Q20={q20:.2f}")
    ax.axvline(q30, color="#F59F00", linestyle="--", linewidth=1.5, label=f"Q30={q30:.2f}")
    ax.set_xlabel("Maximum tile density score per training image")
    ax.set_ylabel("Calibration images")
    ax.set_title("Unlabeled 512-image training calibration distribution")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate_calibration_distribution.png", dpi=180)
    plt.close(fig)


def gate_examples(best_record: dict[str, Any]) -> None:
    config = yaml.safe_load((ROOT / "configs" / "data" / "visdrone2019.yaml").read_text(encoding="utf-8"))
    image_dir = Path(config["path"]) / config["val"]
    candidates = [row for row in best_record["per_image"] if row["selected_window"] and row["available_tiles"] > 1]
    candidates.sort(key=lambda row: row["max_density_score"], reverse=True)
    selected = candidates[:3]
    fig, axes = plt.subplots(1, len(selected), figsize=(15.6, 4.8), squeeze=False)
    for index, row in enumerate(selected):
        image = cv2.imread(str(image_dir / row["file_name"]), cv2.IMREAD_COLOR)
        ax = axes[0, index]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        from evaluate_sahi import slice_windows

        for x1, y1, x2, y2 in slice_windows(row["width"], row["height"], 960, 0.20):
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#ADB5BD", linewidth=1.0, linestyle="--"))
        x1, y1, x2, y2 = row["selected_window"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#FA5252", linewidth=2.5))
        ax.set_title(f"score={row['max_density_score']:.1f}, selected 1/{row['available_tiles']}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Prediction-density gate: all candidate tiles (gray) and selected tile (red)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "gate_selected_tiles.jpg", dpi=170, pil_kwargs={"quality": 92})
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    whole_record = load_json(ROOT / "reports" / "sahi_inference" / "metrics" / "whole.json")
    full_record = load_json(ROOT / "reports" / "scale48_joint_inference" / "metrics" / "joint_scale48.json")
    calibration = load_json(METRICS / "calibration.json")
    gate_records = {key: load_json(METRICS / f"{key}.json") for key, _ in GATES}
    rows = [
        make_row("whole", "Whole-640", whole_record, "baseline"),
        make_row("full_scale48", "Full Scale-48", full_record, "upper_bound"),
        *[make_row(key, label, gate_records[key], "gate") for key, label in GATES],
    ]
    write_csv(METRICS / "gate_summary.csv", rows)
    whole = rows[0]
    full = rows[1]
    deltas = []
    audits = {}
    for row in rows[2:]:
        record = gate_records[row["key"]]
        audits[row["key"]] = audit(record)
        deltas.append(
            {
                "key": row["key"],
                "method": row["method"],
                "delta_all_ap_vs_full": row["all_ap"] - full["all_ap"],
                "delta_small_ap_vs_full": row["small_ap"] - full["small_ap"],
                "delta_small_ar_vs_full": row["small_ar"] - full["small_ar"],
                "delta_medium_ap_vs_full": row["medium_ap"] - full["medium_ap"],
                "delta_large_ap_vs_whole": row["large_ap"] - whole["large_ap"],
                "latency_reduction_vs_full": 1.0 - row["latency_ms"] / full["latency_ms"],
            }
        )
    write_csv(METRICS / "gate_deltas.csv", deltas)
    plot_accuracy_latency(rows, full)
    plot_metrics(rows)
    plot_efficiency(rows[1:])
    plot_calibration(calibration)

    criteria = []
    delta_by_key = {row["key"]: row for row in deltas}
    for row in rows[2:]:
        delta = delta_by_key[row["key"]]
        current_audit = audits[row["key"]]
        checks = {
            "small": delta["delta_small_ap_vs_full"] >= -0.005,
            "all": delta["delta_all_ap_vs_full"] >= -0.002,
            "large": delta["delta_large_ap_vs_whole"] >= -0.02,
            "latency": delta["latency_reduction_vs_full"] >= 0.30,
            "complete": current_audit["images"] == 548 and current_audit["invalid"] == 0 and current_audit["max_per_image"] <= 300,
            "checkpoint": gate_records[row["key"]]["checkpoint"]["matches_calibration"],
        }
        criteria.append((row, delta, checks, all(checks.values())))
    passed = [item for item in criteria if item[3]]
    if passed:
        passed.sort(key=lambda item: (item[0]["latency_ms"], -item[0]["small_ap"]))
        best = passed[0][0]
        decision = f"{best['method']}通过全部预注册条件，建议进行独立复验并作为密度自适应候选。"
    else:
        best = min(rows[2:], key=lambda row: row["latency_ms"])
        decision = "三种密度门控均未通过全部预注册条件，不继续在验证集搜索阈值；保留Full Scale-48作为精度上界。"
    gate_examples(gate_records[best["key"]])

    result_rows = [
        f"| {row['method']} | {fmt_pct(row['all_ap'])} | {fmt_pct(row['small_ap'])} | {fmt_pct(row['small_ar'])} | "
        f"{fmt_pct(row['medium_ap'])} | {fmt_pct(row['large_ap'])} | {row['latency_ms']:.2f} | {row['fps']:.2f} |"
        for row in rows
    ]
    delta_rows = [
        f"| {row['method']} | {fmt_delta(row['delta_all_ap_vs_full'])} | {fmt_delta(row['delta_small_ap_vs_full'])} | "
        f"{fmt_delta(row['delta_small_ar_vs_full'])} | {fmt_delta(row['delta_medium_ap_vs_full'])} | "
        f"{fmt_delta(row['delta_large_ap_vs_whole'])} | {100*row['latency_reduction_vs_full']:+.1f}% |"
        for row in deltas
    ]
    criteria_rows = [
        f"| {row['method']} | {'是' if checks['small'] else '否'} | {'是' if checks['all'] else '否'} | "
        f"{'是' if checks['large'] else '否'} | {'是' if checks['latency'] else '否'} | "
        f"{'是' if checks['complete'] and checks['checkpoint'] else '否'} | {'通过' if is_passed else '未通过'} |"
        for row, _, checks, is_passed in criteria
    ]
    report = [
        "# 密度门控SAHI-960实验报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 门控阈值来自512张训练图像的无标签校准；验证阶段未修改阈值。",
        "",
        "## 1. 结论摘要",
        "",
        f"**预注册决策：{decision}**",
        "",
        "## 2. 无标签校准",
        "",
        f"训练校准集Q20阈值为 **{calibration['quantiles']['q20']:.3f}**，Q30阈值为 **{calibration['quantiles']['q30']:.3f}**；校准过程读取标签：否。",
        "",
        "![校准分布](figures/gate_calibration_distribution.png)",
        "",
        "## 3. 统一结果",
        "",
        "| 方法 | AP50-95 | Small AP | Small AR | Medium AP | Large AP | 延迟(ms/图) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *result_rows,
        "",
        "![指标对比](figures/gate_metrics.png)",
        "",
        "## 4. 相对基准变化",
        "",
        "| 门控 | ΔAP vs Full | ΔSmall AP vs Full | ΔSmall AR vs Full | ΔMedium AP vs Full | ΔLarge AP vs Whole | 延迟降低 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *delta_rows,
        "",
        "![精度延迟权衡](figures/gate_accuracy_latency.png)",
        "",
        "## 5. 门控效率",
        "",
        "| 方法 | 触发图像 | 触发率 | 平均选择切片 | 切片减少 | P95延迟(ms) | 峰值显存(MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['method']} | {gate_records[row['key']]['efficiency']['gate_trigger_images']} | {fmt_pct(row['trigger_rate'])} | "
            f"{row['mean_selected_tiles']:.2f} | {fmt_pct(row['tile_reduction_ratio'])} | {row['p95_latency_ms']:.2f} | {row['peak_gpu_memory_mb']:.0f} |"
            for row in rows[2:]
        ],
        "",
        "![门控效率](figures/gate_efficiency.png)",
        "",
        "![门控切片示意](figures/gate_selected_tiles.jpg)",
        "",
        "## 6. 预注册判定",
        "",
        "| 候选 | Small AP下降≤0.50 pp | AP下降≤0.20 pp | Large AP≥Whole-2.00 pp | 延迟降低≥30% | 完整性 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *criteria_rows,
        "",
        "## 7. 证据边界",
        "",
        "- 密度门控只使用Whole预测，不使用真实标注，具备部署可实现性。",
        "- 阈值由训练图像无标签分布确定，但仍只在当前数据集验证，跨数据集稳定性未知。",
        "- Whole完全漏检的小目标无法为门控提供信号，这是该方法的固有召回上限。",
        "- 即使候选通过，仍需独立重复运行确认耗时与精度，然后再讨论轻量化贡献。",
        "",
        "## 8. 主张—证据映射",
        "",
        "| 主张 | 证据 | 状态 |",
        "|---|---|---|",
        f"| 无标签密度门控满足预注册精度—延迟目标 | {decision} | {'支持' if passed else '不支持'} |",
        "| 门控不依赖真实标注 | 校准与验证程序均只读取图像和Whole预测 | 支持 |",
        "| 门控具有跨数据集泛化能力 | 尚未在其他数据集验证 | 证据不足 |",
        "",
        "## 9. 可追溯产物",
        "",
        "- 预注册方案：`EXPERIMENT_PLAN.md`",
        "- 无标签校准、三组指标、预测与CSV：`metrics/`",
        "- 运行日志：`logs/`",
        "- 图表与门控示意：`figures/`",
    ]
    (REPORT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (METRICS / "gate_audits.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Density gate report generated at {REPORT / 'README.md'}")


if __name__ == "__main__":
    main()
