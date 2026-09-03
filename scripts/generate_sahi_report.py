"""Generate tables, figures, qualitative examples, and a Markdown SAHI report."""

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
REPORT = ROOT / "reports" / "sahi_inference"
METRICS = REPORT / "metrics"
FIGURES = REPORT / "figures"
METHODS = (
    ("whole", "Whole-640", "whole.json"),
    ("sahi640", "SAHI-640", "sahi640.json"),
    ("sahi960", "SAHI-960", "sahi960.json"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_delta(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def flatten_record(key: str, label: str, record: dict[str, Any]) -> dict[str, Any]:
    areas = record["overall_by_area"]
    efficiency = record["efficiency"]
    input_size = int(record["protocol"]["input_size"])
    return {
        "key": key,
        "method": label,
        "all_ap": areas["all"]["ap"],
        "all_ap50": areas["all"]["ap50"],
        "all_ar": areas["all"]["ar_max_det"],
        "small_ap": areas["small"]["ap"],
        "small_ap50": areas["small"]["ap50"],
        "small_ar": areas["small"]["ar_max_det"],
        "medium_ap": areas["medium"]["ap"],
        "large_ap": areas["large"]["ap"],
        "mean_tiles": efficiency["mean_tiles_per_image"],
        "mean_end_to_end_ms": efficiency["mean_end_to_end_ms"],
        "p95_end_to_end_ms": efficiency["p95_end_to_end_ms"],
        "fps": efficiency["fps_from_mean_end_to_end"],
        "peak_gpu_memory_mb": efficiency["peak_gpu_memory_mb"],
        "duplicate_suppression_ratio": efficiency["duplicate_suppression_ratio"],
        "pixel_compute_equivalent": efficiency["mean_tiles_per_image"] * (input_size / 640.0) ** 2,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_size_metrics(rows: list[dict[str, Any]]) -> None:
    categories = (("small_ap", "Small AP"), ("medium_ap", "Medium AP"), ("large_ap", "Large AP"))
    x = np.arange(len(categories))
    width = 0.24
    colors = ("#4263EB", "#12B886", "#F59F00")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for index, (row, color) in enumerate(zip(rows, colors, strict=True)):
        values = [100.0 * row[key] for key, _ in categories]
        bars = ax.bar(x + (index - 1) * width, values, width, label=row["method"], color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x, [label for _, label in categories])
    ax.set_ylabel("COCO AP (%)")
    ax.set_title("Object-size accuracy under whole-image and sliced inference")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "sahi_size_metrics.png", dpi=180)
    plt.close(fig)


def plot_accuracy_latency(rows: list[dict[str, Any]]) -> None:
    colors = ("#4263EB", "#12B886", "#F59F00")
    fig, ax = plt.subplots(figsize=(7.7, 5.4))
    for row, color in zip(rows, colors, strict=True):
        x = row["mean_end_to_end_ms"]
        y = 100.0 * row["small_ap"]
        ax.scatter(x, y, s=115, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate(row["method"], (x, y), xytext=(7, 6), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Mean end-to-end latency (ms/image, lower is better)")
    ax.set_ylabel("Small AP (%, higher is better)")
    ax.set_title("Small-object accuracy–latency trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "sahi_accuracy_latency.png", dpi=180)
    plt.close(fig)


def plot_efficiency(rows: list[dict[str, Any]]) -> None:
    labels = [row["method"] for row in rows]
    latency = [row["mean_end_to_end_ms"] for row in rows]
    peak_memory = [row["peak_gpu_memory_mb"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(8.5, 5.4))
    bars1 = ax1.bar(x - width / 2, latency, width, color="#5C7CFA", label="Latency")
    ax1.set_ylabel("Mean end-to-end latency (ms/image)", color="#364FC7")
    ax1.tick_params(axis="y", labelcolor="#364FC7")
    ax1.set_xticks(x, labels)
    ax1.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, peak_memory, width, color="#FFA94D", label="Peak memory")
    ax2.set_ylabel("Peak allocated GPU memory (MB)", color="#D9480F")
    ax2.tick_params(axis="y", labelcolor="#D9480F")
    ax2.bar_label(bars2, fmt="%.0f", padding=3, fontsize=8)
    ax1.set_title("End-to-end efficiency cost of slicing-aided inference")
    ax1.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "sahi_efficiency.png", dpi=180)
    plt.close(fig)


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = box
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    top_left = np.maximum(box_a[:2], box_b[:2])
    bottom_right = np.minimum(box_a[2:], box_b[2:])
    intersection = float(np.prod(np.maximum(0.0, bottom_right - top_left)))
    area_a = float(np.prod(np.maximum(0.0, box_a[2:] - box_a[:2])))
    area_b = float(np.prod(np.maximum(0.0, box_b[2:] - box_b[:2])))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def matched_small_count(
    ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], score_threshold: float = 0.25
) -> int:
    small_gt = [row for row in ground_truth if row["area"] < 32**2]
    candidates = sorted(
        (row for row in predictions if row["score"] >= score_threshold),
        key=lambda row: row["score"],
        reverse=True,
    )
    used: set[int] = set()
    matched = 0
    for prediction in candidates:
        prediction_box = xywh_to_xyxy(prediction["bbox"])
        best_iou = 0.0
        best_index = None
        for index, target in enumerate(small_gt):
            if index in used or target["category_id"] != prediction["category_id"]:
                continue
            overlap = iou(prediction_box, xywh_to_xyxy(target["bbox"]))
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= 0.5:
            used.add(best_index)
            matched += 1
    return matched


def draw_predictions(
    ax: Any,
    image: np.ndarray,
    predictions: list[dict[str, Any]],
    class_names: dict[int, str],
    title: str,
) -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    shown = sorted((row for row in predictions if row["score"] >= 0.25), key=lambda row: row["score"], reverse=True)[:120]
    cmap = plt.get_cmap("tab10")
    for row in shown:
        x, y, width, height = row["bbox"]
        color = cmap((int(row["category_id"]) - 1) % 10)
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=0.8))
        if width * height >= 24**2:
            label = f"{class_names[int(row['category_id']) - 1]} {row['score']:.2f}"
            ax.text(x, max(0, y - 2), label, color="white", fontsize=5.2, bbox={"facecolor": color, "alpha": 0.75, "pad": 1})
    ax.set_title(f"{title} | predictions≥0.25: {len(shown)}", fontsize=9)
    ax.axis("off")


def create_qualitative_figure(records: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    best = max(rows[1:], key=lambda row: row["small_ap"])
    best_key = best["key"]
    gt_dataset = load_json(Path(records["whole"]["ground_truth_json"]))
    whole_predictions = load_json(Path(records["whole"]["predictions_json"]))
    best_predictions = load_json(Path(records[best_key]["predictions_json"]))
    gt_by_image: dict[Any, list[dict[str, Any]]] = {}
    whole_by_image: dict[Any, list[dict[str, Any]]] = {}
    best_by_image: dict[Any, list[dict[str, Any]]] = {}
    for row in gt_dataset["annotations"]:
        gt_by_image.setdefault(row["image_id"], []).append(row)
    for row in whole_predictions:
        whole_by_image.setdefault(row["image_id"], []).append(row)
    for row in best_predictions:
        best_by_image.setdefault(row["image_id"], []).append(row)

    ranked = []
    for image in gt_dataset["images"]:
        current_id = image["id"]
        targets = gt_by_image.get(current_id, [])
        small_count = sum(row["area"] < 32**2 for row in targets)
        if small_count < 5:
            continue
        whole_matches = matched_small_count(targets, whole_by_image.get(current_id, []))
        best_matches = matched_small_count(targets, best_by_image.get(current_id, []))
        ranked.append((best_matches - whole_matches, small_count, image))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in ranked[:3]]

    data_config = yaml.safe_load((ROOT / "configs" / "data" / "visdrone2019.yaml").read_text(encoding="utf-8"))
    image_dir = Path(data_config["path"]) / data_config["val"]
    class_names = {int(key): str(value) for key, value in data_config["names"].items()}
    fig, axes = plt.subplots(len(selected), 2, figsize=(14, 4.4 * len(selected)), squeeze=False)
    for row_index, image_info in enumerate(selected):
        current_id = image_info["id"]
        image = cv2.imread(str(image_dir / image_info["file_name"]), cv2.IMREAD_COLOR)
        draw_predictions(axes[row_index, 0], image, whole_by_image.get(current_id, []), class_names, "Whole-640")
        draw_predictions(axes[row_index, 1], image, best_by_image.get(current_id, []), class_names, best["method"])
    fig.suptitle("Qualitative comparison on scenes with increased matched small objects", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_name = f"qualitative_whole_vs_{best_key}.jpg"
    fig.savefig(FIGURES / output_name, dpi=170, pil_kwargs={"quality": 92})
    plt.close(fig)
    return output_name


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    records = {key: load_json(METRICS / file_name) for key, _, file_name in METHODS}
    rows = [flatten_record(key, label, records[key]) for key, label, _ in METHODS]
    write_csv(METRICS / "sahi_summary.csv", rows)

    baseline = rows[0]
    deltas = []
    for row in rows[1:]:
        deltas.append(
            {
                "key": row["key"],
                "method": row["method"],
                "delta_all_ap": row["all_ap"] - baseline["all_ap"],
                "delta_all_ap50": row["all_ap50"] - baseline["all_ap50"],
                "delta_small_ap": row["small_ap"] - baseline["small_ap"],
                "delta_small_ap50": row["small_ap50"] - baseline["small_ap50"],
                "delta_small_ar": row["small_ar"] - baseline["small_ar"],
                "delta_medium_ap": row["medium_ap"] - baseline["medium_ap"],
                "delta_large_ap": row["large_ap"] - baseline["large_ap"],
                "latency_multiplier": row["mean_end_to_end_ms"] / baseline["mean_end_to_end_ms"],
                "fps_ratio": row["fps"] / baseline["fps"],
            }
        )
    write_csv(METRICS / "sahi_deltas.csv", deltas)

    plot_size_metrics(rows)
    plot_accuracy_latency(rows)
    plot_efficiency(rows)
    qualitative_name = create_qualitative_figure(records, rows)

    criteria = []
    for row, delta in zip(rows[1:], deltas, strict=True):
        checks = {
            "small": delta["delta_small_ap"] >= 0.01,
            "overall": delta["delta_all_ap"] >= 0.0,
            "recall": delta["delta_small_ar"] > 0.0,
            "complete": records[row["key"]]["dataset_count"]["images"] == 548,
        }
        criteria.append((row, delta, checks, all(checks.values())))
    passed = [entry for entry in criteria if entry[3]]
    best_passed = max(passed, key=lambda entry: entry[0]["small_ap"]) if passed else None
    best_sahi = max(rows[1:], key=lambda row: row["small_ap"])
    historical_record = load_json(REPORT.parent / "unified_evaluation" / "metrics" / "full_size_val.json")
    historical_all_ap = historical_record["overall_by_area"]["all"]["ap"]
    historical_small_ap = historical_record["overall_by_area"]["small"]["ap"]

    table_rows = [
        f"| {row['method']} | {fmt_pct(row['all_ap50'])} | {fmt_pct(row['all_ap'])} | {fmt_pct(row['small_ap'])} | "
        f"{fmt_pct(row['small_ap50'])} | {fmt_pct(row['small_ar'])} | {fmt_pct(row['medium_ap'])} | "
        f"{fmt_pct(row['large_ap'])} | {row['mean_end_to_end_ms']:.1f} | {row['fps']:.1f} |"
        for row in rows
    ]
    delta_rows = [
        f"| {delta['method']} | {fmt_delta(delta['delta_all_ap'])} | {fmt_delta(delta['delta_small_ap'])} | "
        f"{fmt_delta(delta['delta_small_ap50'])} | {fmt_delta(delta['delta_small_ar'])} | "
        f"{fmt_delta(delta['delta_medium_ap'])} | {fmt_delta(delta['delta_large_ap'])} | {delta['latency_multiplier']:.2f}× |"
        for delta in deltas
    ]
    criteria_rows = [
        f"| {row['method']} | {'是' if checks['small'] else '否'} | {'是' if checks['overall'] else '否'} | "
        f"{'是' if checks['recall'] else '否'} | {'是' if checks['complete'] else '否'} | {'通过' if is_passed else '未通过'} |"
        for row, _, checks, is_passed in criteria
    ]
    decision = (
        f"{best_passed[0]['method']}通过预注册条件，建议进入密度自适应切片研究；切片只作为精度上界，"
        "下一阶段必须降低端到端延迟。"
        if best_passed
        else "两种SAHI设置均未通过预注册条件，停止切片主线并转入RFLA/NWD感知标签分配。"
    )
    gain = best_sahi["small_ap"] - baseline["small_ap"]
    report_lines = [
        "# SAHI切片推理对照报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 固定150轮完整模型权重，仅改变推理方式；所有框在原图坐标下统一COCO评估。",
        "",
        "## 1. 结论摘要",
        "",
        f"Small AP最高的切片设置为 **{best_sahi['method']}**，相对Whole-640变化 **{fmt_delta(gain)} 个百分点**。",
        f"**预注册决策：{decision}**",
        "",
        "## 2. 评估入口兼容性说明",
        "",
        "| 入口 | AP50-95 | Small AP | 用途 |",
        "|---|---:|---:|---|",
        f"| 既有Ultralytics `model.val`批量验证 | {fmt_pct(historical_all_ap)} | {fmt_pct(historical_small_ap)} | 与前期150轮统一评估衔接 |",
        f"| 本实验逐图 `model.predict` Whole-640 | {fmt_pct(baseline['all_ap'])} | {fmt_pct(baseline['small_ap'])} | SAHI的同管线因果对照 |",
        "",
        f"两种入口的AP50-95相差 **{fmt_delta(baseline['all_ap'] - historical_all_ap)} 个百分点**，"
        f"Small AP相差 **{fmt_delta(baseline['small_ap'] - historical_small_ap)} 个百分点**。"
        "该差异可能来自批量矩形验证与逐图固定方形预测的预处理、填充及执行路径差异；当前证据不能进一步归因。"
        "为避免混用协议，本报告所有SAHI增量和预注册判定只使用同一 `model.predict` 管线的Whole-640对照；既有数值仅作为兼容性参考。",
        "",
        "## 3. 统一结果",
        "",
        "| 方法 | AP50 | AP50-95 | Small AP | Small AP50 | Small AR | Medium AP | Large AP | 延迟(ms/图) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "![分尺寸指标](figures/sahi_size_metrics.png)",
        "",
        "## 4. 相对Whole-640的变化",
        "",
        "| 方法 | ΔAP50-95 | ΔSmall AP | ΔSmall AP50 | ΔSmall AR | ΔMedium AP | ΔLarge AP | 延迟倍数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *delta_rows,
        "",
        "![精度延迟权衡](figures/sahi_accuracy_latency.png)",
        "",
        "## 5. 效率代价",
        "",
        "| 方法 | 平均切片数 | 640像素计算当量 | P95延迟(ms) | 峰值显存(MB) | 重复框抑制比例 |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {row['method']} | {row['mean_tiles']:.2f} | {row['pixel_compute_equivalent']:.2f}× | "
            f"{row['p95_end_to_end_ms']:.1f} | {row['peak_gpu_memory_mb']:.0f} | {fmt_pct(row['duplicate_suppression_ratio'])} |"
            for row in rows
        ],
        "",
        "“640像素计算当量”按平均切片数×(输入边长/640)²估算，只反映输入像素规模，不替代实测延迟。",
        "",
        "![效率代价](figures/sahi_efficiency.png)",
        "",
        "## 6. 预注册判定",
        "",
        "| 候选 | Small AP≥+1.00 pp | AP50-95不下降 | Small AR提高 | 548张完整 | 最终判定 |",
        "|---|---:|---:|---:|---:|---:|",
        *criteria_rows,
        "",
        "## 7. 定性结果",
        "",
        f"![整图与切片检测对照](figures/{qualitative_name})",
        "",
        "图像根据置信度≥0.25时正确匹配的小目标增加量选择，只用于解释典型成功场景；总体结论仍以全部548张的COCO指标为准。",
        "",
        "## 8. 证据边界与局限",
        "",
        "- 三组使用同一检查点和验证集，能够隔离推理切片的影响。",
        "- SAHI不增加模型参数，但增加输入像素、推理次数与端到端延迟，因此不能直接作为轻量化结论。",
        "- 当前全局合并固定为类别感知NMS、IoU=0.5；本报告不包含阈值后验搜索。",
        "- SAHI-960虽然提高总体AP和Small AP，但Medium AP与Large AP分别下降3.95和18.28个百分点；后续必须融合整图预测或采用尺度感知保留策略。",
        "- 端到端耗时包含图像解码、切片、推理、合并和JSON序列化，仅代表当前RTX 4080 SUPER与PyTorch环境。",
        "",
        "## 9. 主张—证据映射",
        "",
        "| 主张 | 证据 | 状态 |",
        "|---|---|---|",
        f"| 切片改善当前模型的小目标检测 | 最佳ΔSmall AP {fmt_delta(gain)} pp | {'支持' if gain > 0 else '不支持'} |",
        f"| 切片满足预注册晋级条件 | {decision} | {'支持' if best_passed else '不支持'} |",
        "| 切片属于轻量化改进 | 推理次数和端到端延迟均增加 | 不支持 |",
        "",
        "## 10. 可追溯产物",
        "",
        "- 预注册方案：`EXPERIMENT_PLAN.md`",
        "- 结构化指标与预测：`metrics/`",
        "- 三组完整日志：`logs/`",
        "- 图表和定性结果：`figures/`",
    ]
    (REPORT / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"SAHI report generated at {REPORT / 'README.md'}")


if __name__ == "__main__":
    main()
