"""Generate accuracy-reproduction and latency report for joint Scale-48 inference."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "scale48_joint_inference"
METRICS = REPORT / "metrics"
FIGURES = REPORT / "figures"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_delta(value: float) -> str:
    return f"{100.0 * value:+.3f}"


def audit_predictions(joint: dict[str, Any]) -> dict[str, int]:
    ground_truth = load_json(Path(joint["ground_truth_json"]))
    predictions = load_json(Path(joint["predictions_json"]))
    shapes = {image["id"]: (image["width"], image["height"]) for image in ground_truth["images"]}
    per_image: dict[int | str, int] = {}
    invalid = 0
    for prediction in predictions:
        current_id = prediction["image_id"]
        per_image[current_id] = per_image.get(current_id, 0) + 1
        x, y, width, height = [float(value) for value in prediction["bbox"]]
        image_width, image_height = shapes[current_id]
        if (
            x < -1e-3
            or y < -1e-3
            or width <= 0
            or height <= 0
            or x + width > image_width + 1e-3
            or y + height > image_height + 1e-3
        ):
            invalid += 1
    return {
        "images_with_predictions": len(per_image),
        "predictions": len(predictions),
        "max_detections_per_image": max(per_image.values()),
        "out_of_bounds_or_invalid": invalid,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(offline: dict[str, Any], joint: dict[str, Any]) -> None:
    labels = ("AP50-95", "Small AP", "Medium AP", "Large AP")
    offline_values = [
        offline["overall_by_area"]["all"]["ap"],
        offline["overall_by_area"]["small"]["ap"],
        offline["overall_by_area"]["medium"]["ap"],
        offline["overall_by_area"]["large"]["ap"],
    ]
    joint_values = [
        joint["overall_by_area"]["all"]["ap"],
        joint["overall_by_area"]["small"]["ap"],
        joint["overall_by_area"]["medium"]["ap"],
        joint["overall_by_area"]["large"]["ap"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    first = ax.bar(x - width / 2, 100 * np.asarray(offline_values), width, label="Cached offline fusion", color="#868E96")
    second = ax.bar(x + width / 2, 100 * np.asarray(joint_values), width, label="True joint inference", color="#12B886")
    ax.bar_label(first, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(second, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("COCO metric (%)")
    ax.set_title("Accuracy reproduction: cached fusion versus true joint inference")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "joint_accuracy_reproduction.png", dpi=180)
    plt.close(fig)


def plot_timing_breakdown(joint: dict[str, Any]) -> None:
    efficiency = joint["efficiency"]
    components = (
        ("Decode", efficiency["mean_decode_ms"]),
        ("Whole-640", efficiency["mean_whole_ms"]),
        ("Slice setup", efficiency["mean_slice_window_ms"]),
        ("SAHI-960", efficiency["mean_sahi_ms"]),
        ("Scale-48 fusion", efficiency["mean_filter_and_fusion_ms"]),
        ("Serialization", efficiency["mean_serialization_ms"]),
    )
    labels = [label for label, _ in components]
    values = [value for _, value in components]
    colors = ("#ADB5BD", "#4263EB", "#CED4DA", "#F59F00", "#12B886", "#9775FA")
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_ylabel("Mean time (ms/image)")
    ax.set_title("Measured Scale-48 joint inference timing breakdown")
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIGURES / "joint_timing_breakdown.png", dpi=180)
    plt.close(fig)


def plot_latency_by_tiles(joint: dict[str, Any]) -> None:
    rows = joint["per_image"]
    tile_counts = sorted({int(row["tiles"]) for row in rows})
    grouped = [[float(row["end_to_end_ms"]) for row in rows if int(row["tiles"]) == count] for count in tile_counts]
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    box = ax.boxplot(grouped, tick_labels=[str(count) for count in tile_counts], patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#74C0FC")
        patch.set_alpha(0.8)
    ax.axhline(65.0, color="#C92A2A", linestyle="--", linewidth=1.2, label="Pre-registered mean limit")
    ax.set_xlabel("Number of 960×960 slices per image")
    ax.set_ylabel("End-to-end latency (ms/image)")
    ax.set_title("Latency distribution by slice count")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "joint_latency_by_tiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    joint = load_json(METRICS / "joint_scale48.json")
    offline = load_json(Path(joint["offline_record"]))
    whole = load_json(ROOT / "reports" / "sahi_inference" / "metrics" / "whole.json")
    audit = audit_predictions(joint)
    areas = ("all", "small", "medium", "large")
    comparison = []
    for area in areas:
        comparison.append(
            {
                "area": area,
                "offline_ap": offline["overall_by_area"][area]["ap"],
                "joint_ap": joint["overall_by_area"][area]["ap"],
                "delta_ap": joint["overall_by_area"][area]["ap"] - offline["overall_by_area"][area]["ap"],
                "offline_ap50": offline["overall_by_area"][area]["ap50"],
                "joint_ap50": joint["overall_by_area"][area]["ap50"],
                "delta_ap50": joint["overall_by_area"][area]["ap50"] - offline["overall_by_area"][area]["ap50"],
            }
        )
    write_csv(METRICS / "joint_accuracy_comparison.csv", comparison)

    timing_rows = []
    for row in joint["per_image"]:
        timing_rows.append(
            {
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "tiles": row["tiles"],
                "decode_ms": row["decode_ms"],
                "whole_ms": row["whole_ms"],
                "sahi_ms": row["sahi_ms"],
                "fusion_ms": row["filter_and_fusion_ms"],
                "end_to_end_ms": row["end_to_end_ms"],
            }
        )
    write_csv(METRICS / "joint_per_image_timing.csv", timing_rows)
    plot_accuracy(offline, joint)
    plot_timing_breakdown(joint)
    plot_latency_by_tiles(joint)

    diffs = {row["area"]: row["delta_ap"] for row in comparison}
    checks = {
        "all_ap": abs(diffs["all"]) <= 0.001,
        "small_ap": abs(diffs["small"]) <= 0.001,
        "medium_ap": abs(diffs["medium"]) <= 0.001,
        "large_ap": abs(diffs["large"]) <= 0.001,
        "latency": joint["efficiency"]["mean_end_to_end_ms"] <= 65.0,
        "images": joint["dataset_count"]["images"] == 548,
        "bounds": audit["out_of_bounds_or_invalid"] == 0 and audit["max_detections_per_image"] <= 300,
        "checkpoint": joint["checkpoint"]["matches_offline"],
    }
    passed = all(checks.values())
    decision = (
        "真实联合推理复现成功，可进入密度自适应切片实验。"
        if passed
        else "真实联合推理未通过全部复现条件，应先定位差异，暂不进入密度自适应切片。"
    )
    efficiency = joint["efficiency"]
    whole_latency = whole["efficiency"]["mean_end_to_end_ms"]
    conservative_latency = offline["fusion_efficiency"]["conservative_sequential_end_to_end_ms"]
    report = [
        "# Scale-48真实联合推理复验报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 每张图像单次解码，在同一模型进程中顺序执行Whole-640、SAHI-960与Scale-48融合。",
        "",
        "## 1. 结论摘要",
        "",
        f"**预注册决策：{decision}**",
        "",
        "## 2. 精度复现",
        "",
        "| 尺度 | 离线AP | 联合AP | 差值(pp) | 离线AP50 | 联合AP50 | 差值(pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row['area'].title()} | {fmt_pct(row['offline_ap'])} | {fmt_pct(row['joint_ap'])} | {fmt_delta(row['delta_ap'])} | "
            f"{fmt_pct(row['offline_ap50'])} | {fmt_pct(row['joint_ap50'])} | {fmt_delta(row['delta_ap50'])} |"
            for row in comparison
        ],
        "",
        "![精度复现](figures/joint_accuracy_reproduction.png)",
        "",
        "## 3. 真实效率",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 平均端到端延迟 | {efficiency['mean_end_to_end_ms']:.2f} ms/图 |",
        f"| 中位数延迟 | {efficiency['median_end_to_end_ms']:.2f} ms/图 |",
        f"| P95延迟 | {efficiency['p95_end_to_end_ms']:.2f} ms/图 |",
        f"| FPS | {efficiency['fps_from_mean_end_to_end']:.2f} |",
        f"| Whole-640平均延迟 | {whole_latency:.2f} ms/图 |",
        f"| 相对Whole-640延迟倍数 | {efficiency['mean_end_to_end_ms'] / whole_latency:.2f}× |",
        f"| 离线保守延迟估计 | {conservative_latency:.2f} ms/图 |",
        f"| 相对保守估计降低 | {100 * (1 - efficiency['mean_end_to_end_ms'] / conservative_latency):.2f}% |",
        f"| 平均切片数 | {efficiency['mean_tiles_per_image']:.2f} |",
        f"| 峰值GPU显存 | {efficiency['peak_gpu_memory_mb']:.0f} MB |",
        f"| 全部548张墙钟时间 | {efficiency['wall_time_s']:.2f} s |",
        "",
        "![耗时分解](figures/joint_timing_breakdown.png)",
        "",
        "![切片数与延迟](figures/joint_latency_by_tiles.png)",
        "",
        "## 4. 完整性审计",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        f"| 图像数 | {joint['dataset_count']['images']} |",
        f"| 输出预测框 | {audit['predictions']} |",
        f"| 单图最大预测框 | {audit['max_detections_per_image']} |",
        f"| 越界或无效框 | {audit['out_of_bounds_or_invalid']} |",
        f"| 检查点SHA匹配 | {'是' if joint['checkpoint']['matches_offline'] else '否'} |",
        "",
        "## 5. 预注册判定",
        "",
        "| 条件 | 是否通过 |",
        "|---|---:|",
        f"| AP50-95差异≤0.10 pp | {'是' if checks['all_ap'] else '否'} |",
        f"| Small AP差异≤0.10 pp | {'是' if checks['small_ap'] else '否'} |",
        f"| Medium/Large AP差异均≤0.10 pp | {'是' if checks['medium_ap'] and checks['large_ap'] else '否'} |",
        f"| 平均延迟≤65 ms/图 | {'是' if checks['latency'] else '否'} |",
        f"| 548张完整、零越界 | {'是' if checks['images'] and checks['bounds'] else '否'} |",
        f"| 同一检查点 | {'是' if checks['checkpoint'] else '否'} |",
        "",
        "## 6. 证据边界与下一步",
        "",
        "本实验使用同一Python进程顺序执行两路推理，精度和延迟可以作为后续密度门控的真实基线；但结果仅适用于当前RTX 4080 SUPER与PyTorch环境，不能替代ONNX/TensorRT部署测试。Scale-48仍需两路推理，因此不构成轻量化结论。",
        "",
        ("下一步可固定本结果为全量切片上界，设计候选区域门控，只对小目标密集区域运行SAHI-960，并以Small AP下降≤0.50 pp、总体AP下降≤0.20 pp、延迟降低≥30%作为筛选条件。" if passed else "下一步应检查同进程交替640/960输入、预处理参数和NMS执行路径，不应直接优化门控。"),
        "",
        "## 7. 主张—证据映射",
        "",
        "| 主张 | 证据 | 状态 |",
        "|---|---|---|",
        f"| 离线Scale-48精度可由真实联合程序复现 | 最大尺度AP差异 {100*max(abs(value) for value in diffs.values()):.3f} pp | {'支持' if checks['all_ap'] and checks['small_ap'] and checks['medium_ap'] and checks['large_ap'] else '不支持'} |",
        f"| 当前联合程序满足65 ms延迟目标 | 平均 {efficiency['mean_end_to_end_ms']:.2f} ms/图 | {'支持' if checks['latency'] else '不支持'} |",
        "| 当前方法已经轻量化 | 仍执行Whole与SAHI两路推理 | 不支持 |",
        "",
        "## 8. 可追溯产物",
        "",
        "- 预注册方案：`EXPERIMENT_PLAN.md`",
        "- 联合预测、结构化指标及逐图耗时：`metrics/`",
        "- 运行日志：`logs/`",
        "- 精度与效率图：`figures/`",
    ]
    (REPORT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (METRICS / "joint_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Joint Scale-48 report generated at {REPORT / 'README.md'}")


if __name__ == "__main__":
    main()
