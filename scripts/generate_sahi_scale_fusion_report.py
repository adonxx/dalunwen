"""Generate the scale-aware Whole+SAHI fusion experiment report."""

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
SOURCE = ROOT / "reports" / "sahi_inference"
REPORT = ROOT / "reports" / "sahi_scale_fusion"
METRICS = REPORT / "metrics"
FIGURES = REPORT / "figures"
FUSION_METHODS = (
    ("union_all", "Union-All"),
    ("scale_32", "Scale-32"),
    ("scale_48", "Scale-48"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_delta(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def source_row(key: str, label: str, record: dict[str, Any]) -> dict[str, Any]:
    areas = record["overall_by_area"]
    efficiency = record["efficiency"]
    return {
        "key": key,
        "method": label,
        "all_ap": areas["all"]["ap"],
        "all_ap50": areas["all"]["ap50"],
        "small_ap": areas["small"]["ap"],
        "small_ap50": areas["small"]["ap50"],
        "small_ar": areas["small"]["ar_max_det"],
        "medium_ap": areas["medium"]["ap"],
        "large_ap": areas["large"]["ap"],
        "estimated_latency_ms": efficiency["mean_end_to_end_ms"],
        "estimated_fps": efficiency["fps_from_mean_end_to_end"],
        "mean_fusion_ms": 0.0,
        "kept_whole": None,
        "kept_sahi": None,
    }


def fusion_row(key: str, label: str, record: dict[str, Any]) -> dict[str, Any]:
    areas = record["overall_by_area"]
    efficiency = record["fusion_efficiency"]
    return {
        "key": key,
        "method": label,
        "all_ap": areas["all"]["ap"],
        "all_ap50": areas["all"]["ap50"],
        "small_ap": areas["small"]["ap"],
        "small_ap50": areas["small"]["ap50"],
        "small_ar": areas["small"]["ar_max_det"],
        "medium_ap": areas["medium"]["ap"],
        "large_ap": areas["large"]["ap"],
        "estimated_latency_ms": efficiency["conservative_sequential_end_to_end_ms"],
        "estimated_fps": efficiency["conservative_sequential_fps"],
        "mean_fusion_ms": efficiency["mean_fusion_ms"],
        "kept_whole": efficiency["kept_whole"],
        "kept_sahi": efficiency["kept_sahi"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, Any]]) -> None:
    x = np.arange(3)
    width = 0.15
    colors = ("#4263EB", "#868E96", "#E64980", "#12B886", "#F59F00")
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    for index, (row, color) in enumerate(zip(rows, colors, strict=True)):
        values = [100 * row["small_ap"], 100 * row["medium_ap"], 100 * row["large_ap"]]
        bars = ax.bar(x + (index - 2) * width, values, width, label=row["method"], color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x, ("Small AP", "Medium AP", "Large AP"))
    ax.set_ylabel("COCO AP (%)")
    ax.set_title("Scale-aware fusion recovers context while retaining sliced small-object cues")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_size_metrics.png", dpi=180)
    plt.close(fig)


def plot_scale_tradeoff(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    colors = ("#868E96", "#E64980", "#12B886", "#F59F00")
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for row, color in zip(rows[1:], colors, strict=True):
        small_delta = 100 * (row["small_ap"] - baseline["small_ap"])
        large_delta = 100 * (row["large_ap"] - baseline["large_ap"])
        ax.scatter(large_delta, small_delta, s=115, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate(row["method"], (large_delta, small_delta), xytext=(6, 5), textcoords="offset points")
    ax.axhline(2.0, color="#495057", linestyle="--", linewidth=1, label="Small AP threshold")
    ax.axvline(-2.0, color="#C92A2A", linestyle="--", linewidth=1, label="Large AP limit")
    ax.set_xlabel("ΔLarge AP vs Whole-640 (pp; right is better)")
    ax.set_ylabel("ΔSmall AP vs Whole-640 (pp; higher is better)")
    ax.set_title("Small-object gain versus large-object retention")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_scale_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_sources(rows: list[dict[str, Any]]) -> None:
    fusion_rows = rows[2:]
    labels = [row["method"] for row in fusion_rows]
    whole = np.asarray([row["kept_whole"] for row in fusion_rows], dtype=float)
    sahi = np.asarray([row["kept_sahi"] for row in fusion_rows], dtype=float)
    totals = whole + sahi
    whole_pct = 100.0 * whole / totals
    sahi_pct = 100.0 * sahi / totals
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.bar(labels, whole_pct, color="#4263EB", label="Kept from Whole-640")
    ax.bar(labels, sahi_pct, bottom=whole_pct, color="#F59F00", label="Kept from SAHI-960")
    for index, (first, second) in enumerate(zip(whole_pct, sahi_pct, strict=True)):
        ax.text(index, first / 2, f"{first:.1f}%", ha="center", va="center", color="white", fontsize=9)
        ax.text(index, first + second / 2, f"{second:.1f}%", ha="center", va="center", color="black", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of retained predictions (%)")
    ax.set_title("Prediction-source composition after global NMS")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_source_composition.png", dpi=180)
    plt.close(fig)


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = box
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    top_left = np.maximum(box_a[:2], box_b[:2])
    bottom_right = np.minimum(box_a[2:], box_b[2:])
    intersection = float(np.prod(np.maximum(0.0, bottom_right - top_left)))
    union = float(np.prod(box_a[2:] - box_a[:2]) + np.prod(box_b[2:] - box_b[:2]) - intersection)
    return 0.0 if union <= 0 else intersection / union


def matched_small_count(gt: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> int:
    targets = [row for row in gt if row["area"] < 32**2]
    candidates = sorted((row for row in predictions if row["score"] >= 0.25), key=lambda row: row["score"], reverse=True)
    used: set[int] = set()
    matches = 0
    for prediction in candidates:
        best_overlap, best_index = 0.0, None
        for index, target in enumerate(targets):
            if index in used or prediction["category_id"] != target["category_id"]:
                continue
            overlap = iou(xywh_to_xyxy(prediction["bbox"]), xywh_to_xyxy(target["bbox"]))
            if overlap > best_overlap:
                best_overlap, best_index = overlap, index
        if best_index is not None and best_overlap >= 0.5:
            used.add(best_index)
            matches += 1
    return matches


def draw(ax: Any, image: np.ndarray, predictions: list[dict[str, Any]], title: str, names: dict[int, str]) -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    shown = sorted((row for row in predictions if row["score"] >= 0.25), key=lambda row: row["score"], reverse=True)[:100]
    cmap = plt.get_cmap("tab10")
    for record in shown:
        x, y, width, height = record["bbox"]
        color = cmap((int(record["category_id"]) - 1) % 10)
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=0.75))
        if width * height >= 32**2:
            label = f"{names[int(record['category_id']) - 1]} {record['score']:.2f}"
            ax.text(x, max(0, y - 2), label, fontsize=5, color="white", bbox={"facecolor": color, "alpha": 0.72, "pad": 1})
    ax.set_title(f"{title} | predictions≥0.25: {len(shown)}", fontsize=8.5)
    ax.axis("off")


def qualitative_figure(best_key: str, best_label: str, source_records: dict[str, Any], fusion_records: dict[str, Any]) -> str:
    gt_dataset = load_json(Path(source_records["whole"]["ground_truth_json"]))
    whole_predictions = load_json(Path(source_records["whole"]["predictions_json"]))
    sahi_predictions = load_json(Path(source_records["sahi960"]["predictions_json"]))
    fusion_predictions = load_json(Path(fusion_records[best_key]["predictions_json"]))
    collections = []
    for records in (gt_dataset["annotations"], whole_predictions, sahi_predictions, fusion_predictions):
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(record["image_id"], []).append(record)
        collections.append(grouped)
    gt_by_image, whole_by_image, sahi_by_image, fusion_by_image = collections
    ranked = []
    for info in gt_dataset["images"]:
        targets = gt_by_image.get(info["id"], [])
        small_count = sum(row["area"] < 32**2 for row in targets)
        context_count = sum(row["area"] >= 32**2 for row in targets)
        if small_count < 5 or context_count < 1:
            continue
        gain = matched_small_count(targets, fusion_by_image.get(info["id"], [])) - matched_small_count(
            targets, whole_by_image.get(info["id"], [])
        )
        ranked.append((gain, context_count, small_count, info))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected = [item[3] for item in ranked[:2]]
    config = yaml.safe_load((ROOT / "configs" / "data" / "visdrone2019.yaml").read_text(encoding="utf-8"))
    image_dir = Path(config["path"]) / config["val"]
    names = {int(key): str(value) for key, value in config["names"].items()}
    fig, axes = plt.subplots(len(selected), 3, figsize=(16, 4.7 * len(selected)), squeeze=False)
    for row_index, info in enumerate(selected):
        image = cv2.imread(str(image_dir / info["file_name"]), cv2.IMREAD_COLOR)
        current_id = info["id"]
        draw(axes[row_index, 0], image, whole_by_image.get(current_id, []), "Whole-640", names)
        draw(axes[row_index, 1], image, sahi_by_image.get(current_id, []), "SAHI-960", names)
        draw(axes[row_index, 2], image, fusion_by_image.get(current_id, []), best_label, names)
    fig.suptitle("Whole, sliced, and scale-aware fused predictions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_name = f"qualitative_{best_key}.jpg"
    fig.savefig(FIGURES / output_name, dpi=170, pil_kwargs={"quality": 92})
    plt.close(fig)
    return output_name


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_records = {
        "whole": load_json(SOURCE / "metrics" / "whole.json"),
        "sahi960": load_json(SOURCE / "metrics" / "sahi960.json"),
    }
    fusion_records = {key: load_json(METRICS / f"{key}.json") for key, _ in FUSION_METHODS}
    rows = [
        source_row("whole", "Whole-640", source_records["whole"]),
        source_row("sahi960", "SAHI-960", source_records["sahi960"]),
        *[fusion_row(key, label, fusion_records[key]) for key, label in FUSION_METHODS],
    ]
    write_csv(METRICS / "fusion_summary.csv", rows)
    baseline = rows[0]
    deltas = []
    for row in rows[1:]:
        deltas.append(
            {
                "key": row["key"],
                "method": row["method"],
                "delta_all_ap": row["all_ap"] - baseline["all_ap"],
                "delta_small_ap": row["small_ap"] - baseline["small_ap"],
                "delta_small_ap50": row["small_ap50"] - baseline["small_ap50"],
                "delta_small_ar": row["small_ar"] - baseline["small_ar"],
                "delta_medium_ap": row["medium_ap"] - baseline["medium_ap"],
                "delta_large_ap": row["large_ap"] - baseline["large_ap"],
            }
        )
    write_csv(METRICS / "fusion_deltas.csv", deltas)
    plot_metrics(rows)
    plot_scale_tradeoff(rows, baseline)
    plot_sources(rows)

    criteria = []
    delta_by_key = {row["key"]: row for row in deltas}
    for row in rows[2:]:
        delta = delta_by_key[row["key"]]
        checks = {
            "small": delta["delta_small_ap"] >= 0.02,
            "all": delta["delta_all_ap"] >= 0.0,
            "medium": delta["delta_medium_ap"] >= -0.01,
            "large": delta["delta_large_ap"] >= -0.02,
            "complete": fusion_records[row["key"]]["dataset_count"]["images"] == 548,
        }
        criteria.append((row, delta, checks, all(checks.values())))
    passed = [item for item in criteria if item[3]]
    if passed:
        passed.sort(key=lambda item: (item[0]["small_ap"], item[0]["all_ap"]), reverse=True)
        best = passed[0][0]
        decision = f"{best['method']}通过全部预注册条件，建议进入实际联合推理与密度自适应切片。"
    else:
        best = max(rows[2:], key=lambda row: row["small_ap"])
        decision = "三种融合均未通过全部预注册条件，应停止面积阈值扩展并转入区域密度预测或RFLA标签分配。"
    qualitative_name = qualitative_figure(best["key"], best["method"], source_records, fusion_records)

    main_rows = [
        f"| {row['method']} | {fmt_pct(row['all_ap'])} | {fmt_pct(row['small_ap'])} | {fmt_pct(row['small_ap50'])} | "
        f"{fmt_pct(row['small_ar'])} | {fmt_pct(row['medium_ap'])} | {fmt_pct(row['large_ap'])} | {row['estimated_latency_ms']:.1f} |"
        for row in rows
    ]
    delta_rows = [
        f"| {row['method']} | {fmt_delta(row['delta_all_ap'])} | {fmt_delta(row['delta_small_ap'])} | "
        f"{fmt_delta(row['delta_small_ar'])} | {fmt_delta(row['delta_medium_ap'])} | {fmt_delta(row['delta_large_ap'])} |"
        for row in deltas
    ]
    criteria_rows = [
        f"| {row['method']} | {'是' if checks['small'] else '否'} | {'是' if checks['all'] else '否'} | "
        f"{'是' if checks['medium'] else '否'} | {'是' if checks['large'] else '否'} | {'是' if checks['complete'] else '否'} | "
        f"{'通过' if is_passed else '未通过'} |"
        for row, _, checks, is_passed in criteria
    ]
    report = [
        "# Whole-640与SAHI-960尺度感知融合报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 本实验复用缓存预测，仅改变预测框选择与全局类别感知NMS；不重新训练或推理。",
        "",
        "## 1. 结论摘要",
        "",
        f"**预注册决策：{decision}**",
        "",
        "## 2. 统一结果",
        "",
        "| 方法 | AP50-95 | Small AP | Small AP50 | Small AR | Medium AP | Large AP | 估计延迟(ms/图) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *main_rows,
        "",
        "融合方法的延迟为Whole与SAHI实测均值之和再加缓存后处理时间，是保守估计，不是真实集成部署测速。",
        "",
        "![分尺寸指标](figures/fusion_size_metrics.png)",
        "",
        "## 3. 相对Whole-640的变化",
        "",
        "| 方法 | ΔAP50-95 | ΔSmall AP | ΔSmall AR | ΔMedium AP | ΔLarge AP |",
        "|---|---:|---:|---:|---:|---:|",
        *delta_rows,
        "",
        "![尺度权衡](figures/fusion_scale_tradeoff.png)",
        "",
        "## 4. 融合来源与效率",
        "",
        "| 融合方法 | CPU融合(ms/图) | 保留Whole框 | 保留SAHI框 | 估计FPS |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['method']} | {row['mean_fusion_ms']:.3f} | {row['kept_whole']} | {row['kept_sahi']} | {row['estimated_fps']:.1f} |"
            for row in rows[2:]
        ],
        "",
        "![来源组成](figures/fusion_source_composition.png)",
        "",
        "## 5. 预注册判定",
        "",
        "| 候选 | Small AP≥+2.00 pp | 总体AP不下降 | Medium AP≥-1.00 pp | Large AP≥-2.00 pp | 548张完整 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *criteria_rows,
        "",
        "## 6. 定性结果",
        "",
        f"![融合定性对照](figures/{qualitative_name})",
        "",
        "定性图根据融合后置信度≥0.25的小目标正确匹配增量选择，同时要求场景包含中或大目标；定量结论以全部548张评估为准。",
        "",
        "## 7. 证据边界与局限",
        "",
        "- 三组融合规则在查看融合结果前固定，未继续搜索尺度阈值或NMS参数。",
        "- 本实验能够验证缓存预测的尺度选择效果，但不能证明真实联合推理延迟。",
        "- 融合需要Whole与SAHI两路预测，不增加参数量但增加总计算，尚不能称为轻量化方法。",
        "- 预测框面积只是目标真实尺度的代理，边界附近可能发生尺度误分。",
        "",
        "## 8. 主张—证据映射",
        "",
        "| 主张 | 证据 | 状态 |",
        "|---|---|---|",
        f"| 尺度感知融合同时保留小目标增益与大目标能力 | {decision} | {'支持' if passed else '不支持'} |",
        "| 融合规则不增加模型参数 | 仅对缓存框执行筛选和NMS | 支持 |",
        "| 融合保持轻量推理 | 需要两路模型推理，当前仅有保守延迟估计 | 不支持 |",
        "",
        "## 9. 可追溯产物",
        "",
        "- 预注册方案：`EXPERIMENT_PLAN.md`",
        "- 三组融合预测、指标与CSV：`metrics/`",
        "- 运行日志：`logs/`",
        "- 图表与定性结果：`figures/`",
    ]
    (REPORT / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Fusion report generated at {REPORT / 'README.md'}")


if __name__ == "__main__":
    main()
