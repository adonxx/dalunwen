"""Generate a thesis-ready Markdown report from unified checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
STAGES = OrderedDict(
    [
        ("baseline", {"label": "YOLOv11s", "run": "baseline_paper"}),
        ("mffpn", {"label": "+ MF-FPN", "run": "mffpn_paper"}),
        ("lscd", {"label": "+ LSCD", "run": "lscd_paper"}),
        ("full", {"label": "+ Inner-WIoU", "run": "full_paper"}),
    ]
)
COLORS = ["#5B6C8F", "#2F80ED", "#27AE60", "#EB5757"]
METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def load_records(report_dir: Path) -> OrderedDict[str, dict]:
    records: OrderedDict[str, dict] = OrderedDict()
    for stage in STAGES:
        path = report_dir / "metrics" / f"{stage}_val.json"
        if not path.exists():
            raise FileNotFoundError(f"missing unified evaluation record: {path}")
        records[stage] = json.loads(path.read_text(encoding="utf-8"))
    return records


def load_size_records(report_dir: Path) -> OrderedDict[str, dict]:
    """Load COCO area-stratified records for all ablation stages."""
    records: OrderedDict[str, dict] = OrderedDict()
    for stage in STAGES:
        path = report_dir / "metrics" / f"{stage}_size_val.json"
        if not path.exists():
            raise FileNotFoundError(f"missing size-stratified evaluation record: {path}")
        records[stage] = json.loads(path.read_text(encoding="utf-8"))
    return records


def load_training_rows() -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    frames: dict[str, pd.DataFrame] = {}
    best_rows: dict[str, pd.Series] = {}
    for stage, meta in STAGES.items():
        frame = pd.read_csv(ROOT / "runs" / meta["run"] / "results.csv")
        frame.columns = [column.strip() for column in frame.columns]
        frames[stage] = frame
        # Ultralytics 8.4.110 selects best.pt by mAP50-95 fitness.
        best_rows[stage] = frame.loc[frame["metrics/mAP50-95(B)"].idxmax()]
    return frames, best_rows


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.18,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def plot_overall(records: OrderedDict[str, dict], output: Path) -> None:
    labels = [STAGES[stage]["label"] for stage in STAGES]
    metrics = ["precision", "recall", "map50", "map50_95"]
    metric_labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for index, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        values = [100 * records[stage]["overall"][METRIC_KEYS[metric]] for stage in STAGES]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=metric_label, alpha=0.9)
        if metric in {"map50", "map50_95"}:
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5, rotation=90)
    ax.set_ylabel("Metric (%)")
    ax.set_title("Unified validation metrics on VisDrone2019 val")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(60, ax.get_ylim()[1] + 4))
    ax.legend(ncols=4, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_deltas(records: OrderedDict[str, dict], output: Path) -> None:
    stages = list(STAGES)
    transitions = [
        "Baseline -> MF-FPN",
        "MF-FPN -> LSCD",
        "LSCD -> Inner-WIoU",
        "Baseline -> Full",
    ]
    pairs = [(stages[0], stages[1]), (stages[1], stages[2]), (stages[2], stages[3]), (stages[0], stages[3])]
    delta50 = [
        100 * (records[right]["overall"][METRIC_KEYS["map50"]] - records[left]["overall"][METRIC_KEYS["map50"]])
        for left, right in pairs
    ]
    delta95 = [
        100
        * (
            records[right]["overall"][METRIC_KEYS["map50_95"]]
            - records[left]["overall"][METRIC_KEYS["map50_95"]]
        )
        for left, right in pairs
    ]
    x = np.arange(len(transitions))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars50 = ax.bar(x - width / 2, delta50, width, label="Delta mAP50", color="#2F80ED")
    bars95 = ax.bar(x + width / 2, delta95, width, label="Delta mAP50-95", color="#EB5757")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.bar_label(bars50, fmt="%+.2f", padding=3, fontsize=8.5)
    ax.bar_label(bars95, fmt="%+.2f", padding=3, fontsize=8.5)
    ax.set_ylabel("Change (percentage points)")
    ax.set_title("Stage-wise ablation deltas under the unified protocol")
    ax.set_xticks(x, transitions, rotation=12, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_training_curves(frames: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for (stage, frame), color in zip(frames.items(), COLORS):
        axes[0].plot(frame["epoch"], 100 * frame["metrics/mAP50(B)"], label=STAGES[stage]["label"], color=color)
        axes[1].plot(
            frame["epoch"],
            100 * frame["metrics/mAP50-95(B)"],
            label=STAGES[stage]["label"],
            color=color,
        )
    axes[0].set_title("Validation mAP50 across training")
    axes[1].set_title("Validation mAP50-95 across training")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric (%)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def per_class_frame(records: OrderedDict[str, dict]) -> pd.DataFrame:
    rows = []
    for stage, record in records.items():
        for class_row in record["per_class"]:
            rows.append({"stage": stage, **class_row})
    return pd.DataFrame(rows).sort_values(["class_id", "stage"])


def plot_per_class(per_class: pd.DataFrame, output: Path) -> None:
    class_names = (
        per_class[per_class["stage"] == "baseline"].sort_values("class_id")["class_name"].tolist()
    )
    x = np.arange(len(class_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 5.8))
    for index, (stage, color) in enumerate(zip(STAGES, COLORS)):
        subset = per_class[per_class["stage"] == stage].sort_values("class_id")
        ax.bar(
            x + (index - 1.5) * width,
            100 * subset["ap50"].to_numpy(),
            width,
            label=STAGES[stage]["label"],
            color=color,
        )
    ax.set_ylabel("AP50 (%)")
    ax.set_title("Per-class AP50 under the unified validation protocol")
    ax.set_xticks(x, class_names, rotation=28, ha="right")
    ax.legend(ncols=4, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_class_delta(per_class: pd.DataFrame, output: Path) -> None:
    baseline = per_class[per_class["stage"] == "baseline"].set_index("class_id")
    full = per_class[per_class["stage"] == "full"].set_index("class_id")
    names = baseline["class_name"].tolist()
    delta50 = 100 * (full["ap50"] - baseline["ap50"])
    delta95 = 100 * (full["ap50_95"] - baseline["ap50_95"])
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - width / 2, delta50, width, label="Delta AP50", color="#2F80ED")
    ax.bar(x + width / 2, delta95, width, label="Delta AP50-95", color="#EB5757")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Full - baseline (percentage points)")
    ax.set_title("Class-wise change from YOLOv11s to the full model")
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def size_frames(size_records: OrderedDict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten overall and per-class COCO area metrics into tabular data."""
    overall_rows = []
    class_rows = []
    for stage, record in size_records.items():
        counts = record["ground_truth_area_counts"]
        for area, metrics in record["overall_by_area"].items():
            overall_rows.append(
                {
                    "stage": stage,
                    "display_name": STAGES[stage]["label"],
                    "area": area,
                    "object_count": counts[area],
                    **metrics,
                }
            )
        for row in record["per_class_by_area"]:
            class_rows.append({"stage": stage, **row})
    return pd.DataFrame(overall_rows), pd.DataFrame(class_rows)


def plot_size_metrics(size_metrics: pd.DataFrame, output: Path) -> None:
    """Plot AP and AP50 for COCO small/medium/large object bins."""
    areas = ["small", "medium", "large"]
    area_labels = ["Small (<32²)", "Medium (32²-96²)", "Large (>=96²)"]
    x = np.arange(len(areas))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for stage_index, (stage, color) in enumerate(zip(STAGES, COLORS)):
        subset = size_metrics[size_metrics["stage"] == stage].set_index("area")
        offset = (stage_index - 1.5) * width
        for axis, metric, title in zip(
            axes,
            ("ap", "ap50"),
            ("COCO AP50-95 by object size", "COCO AP50 by object size"),
        ):
            bars = axis.bar(
                x + offset,
                100 * subset.loc[areas, metric].to_numpy(),
                width,
                label=STAGES[stage]["label"],
                color=color,
            )
            if stage in {"baseline", "full"}:
                axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, rotation=90)
            axis.set_title(title)
            axis.set_ylabel("Metric (%)")
            axis.set_xticks(x, area_labels)
    axes[1].legend(ncols=2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def copy_evaluation_assets(records: OrderedDict[str, dict], figures: Path) -> None:
    for stage in ("baseline", "full"):
        save_dir = Path(records[stage]["save_dir"])
        for source_name, target_name in (
            ("confusion_matrix_normalized.png", f"{stage}_confusion_matrix_normalized.png"),
            ("val_batch0_pred.jpg", f"{stage}_val_batch0_pred.jpg"),
        ):
            source = save_dir / source_name
            if source.exists():
                shutil.copy2(source, figures / target_name)

    training_dashboard = ROOT / "runs" / STAGES["full"]["run"] / "results.png"
    if training_dashboard.exists():
        shutil.copy2(training_dashboard, figures / "full_training_dashboard.png")


def make_qualitative_comparison(figures: Path) -> None:
    left_path = figures / "baseline_val_batch0_pred.jpg"
    right_path = figures / "full_val_batch0_pred.jpg"
    if not left_path.exists() or not right_path.exists():
        return
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    height = min(left.height, right.height)
    left = left.resize((round(left.width * height / left.height), height))
    right = right.resize((round(right.width * height / right.height), height))
    title_height = max(46, height // 18)
    canvas = Image.new("RGB", (left.width + right.width, height + title_height), "white")
    canvas.paste(left, (0, title_height))
    canvas.paste(right, (left.width, title_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(18, title_height // 2))
    draw.text((left.width // 2, title_height // 2), "YOLOv11s baseline", fill="black", anchor="mm", font=font)
    draw.text(
        (left.width + right.width // 2, title_height // 2),
        "Full Drone-YOLO",
        fill="black",
        anchor="mm",
        font=font,
    )
    canvas.save(figures / "qualitative_baseline_vs_full.jpg", quality=94)


def write_csvs(
    records: OrderedDict[str, dict],
    frames: dict[str, pd.DataFrame],
    best_rows: dict[str, pd.Series],
    per_class: pd.DataFrame,
    report_dir: Path,
) -> pd.DataFrame:
    rows = []
    for stage, record in records.items():
        best = best_rows[stage]
        overall = record["overall"]
        rows.append(
            {
                "stage": stage,
                "display_name": STAGES[stage]["label"],
                "best_epoch_from_training": int(best["epoch"]),
                "training_time_hours": float(frames[stage].iloc[-1]["time"]) / 3600,
                "precision": overall[METRIC_KEYS["precision"]],
                "recall": overall[METRIC_KEYS["recall"]],
                "mAP50": overall[METRIC_KEYS["map50"]],
                "mAP50_95": overall[METRIC_KEYS["map50_95"]],
                "parameters": record["model"]["parameters"],
                "gflops_640": record["model"]["gflops_640"],
                "inference_ms_per_image": record["speed_ms_per_image"].get("inference"),
                "checkpoint_sha256": record["checkpoint"]["sha256"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(report_dir / "metrics" / "summary.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(report_dir / "metrics" / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    return summary


def markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "| 阶段 | 最佳轮次 | P | R | mAP50 | mAP50-95 | 参数量(M) | GFLOPs | 推理(ms/图) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.best_epoch_from_training} | {fmt_pct(row.precision)} | "
            f"{fmt_pct(row.recall)} | {fmt_pct(row.mAP50)} | {fmt_pct(row.mAP50_95)} | "
            f"{row.parameters / 1e6:.3f} | {row.gflops_640:.2f} | {row.inference_ms_per_image:.2f} |"
        )
    return "\n".join(lines)


def per_class_markdown(per_class: pd.DataFrame) -> str:
    pivot50 = per_class.pivot(index="class_name", columns="stage", values="ap50")
    pivot95 = per_class.pivot(index="class_name", columns="stage", values="ap50_95")
    class_order = (
        per_class[per_class["stage"] == "baseline"].sort_values("class_id")["class_name"].tolist()
    )
    lines = [
        "| 类别 | 基线 AP50 | MF-FPN AP50 | LSCD AP50 | 完整模型 AP50 | 完整模型 AP50-95 | AP50变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in class_order:
        delta = 100 * (pivot50.loc[name, "full"] - pivot50.loc[name, "baseline"])
        lines.append(
            f"| {name} | {fmt_pct(pivot50.loc[name, 'baseline'])} | {fmt_pct(pivot50.loc[name, 'mffpn'])} | "
            f"{fmt_pct(pivot50.loc[name, 'lscd'])} | {fmt_pct(pivot50.loc[name, 'full'])} | "
            f"{fmt_pct(pivot95.loc[name, 'full'])} | {delta:+.2f} pp |"
        )
    return "\n".join(lines)


def size_markdown(size_metrics: pd.DataFrame) -> str:
    """Render one compact table containing the key area-stratified metrics."""
    lines = [
        "| 阶段 | 全部 AP | Small AP | Small AP50 | Small AR | Medium AP | Large AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        subset = size_metrics[size_metrics["stage"] == stage].set_index("area")
        lines.append(
            f"| {STAGES[stage]['label']} | {fmt_pct(subset.loc['all', 'ap'])} | "
            f"{fmt_pct(subset.loc['small', 'ap'])} | {fmt_pct(subset.loc['small', 'ap50'])} | "
            f"{fmt_pct(subset.loc['small', 'ar_max_det'])} | {fmt_pct(subset.loc['medium', 'ap'])} | "
            f"{fmt_pct(subset.loc['large', 'ap'])} |"
        )
    return "\n".join(lines)


def generate_markdown(
    records: OrderedDict[str, dict],
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    size_records: OrderedDict[str, dict],
    size_metrics: pd.DataFrame,
    report_dir: Path,
) -> None:
    baseline = summary.set_index("stage").loc["baseline"]
    mffpn = summary.set_index("stage").loc["mffpn"]
    lscd = summary.set_index("stage").loc["lscd"]
    full = summary.set_index("stage").loc["full"]
    full_gain50 = 100 * (full.mAP50 - baseline.mAP50)
    full_gain95 = 100 * (full.mAP50_95 - baseline.mAP50_95)
    full_gain_recall = 100 * (full.recall - baseline.recall)
    mffpn_delta = 100 * (mffpn.mAP50 - baseline.mAP50)
    lscd_delta = 100 * (lscd.mAP50 - mffpn.mAP50)
    inner_delta = 100 * (full.mAP50 - lscd.mAP50)
    paper_gap = 44.3 - 100 * full.mAP50
    size_lookup = size_metrics.set_index(["stage", "area"])
    small_ap_gain = 100 * (
        size_lookup.loc[("full", "small"), "ap"]
        - size_lookup.loc[("baseline", "small"), "ap"]
    )
    small_ap50_gain = 100 * (
        size_lookup.loc[("full", "small"), "ap50"]
        - size_lookup.loc[("baseline", "small"), "ap50"]
    )
    large_ap_gain = 100 * (
        size_lookup.loc[("full", "large"), "ap"]
        - size_lookup.loc[("baseline", "large"), "ap"]
    )
    size_counts = size_records["baseline"]["ground_truth_area_counts"]
    protocol = records["baseline"]["protocol"]
    environment = records["baseline"]["environment"]

    figure = lambda name: f"figures/{name}"  # noqa: E731
    lines = [
        "# Drone-YOLO 复现实验统一评估报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 报告性质：四组 `best.pt` 在同一验证协议下的统一复测；训练过程指标仅用于定位最佳轮次和观察收敛。",
        "",
        "## 1. 结论摘要",
        "",
        (
            f"完整模型在统一复测中取得 **mAP50={fmt_pct(full.mAP50)}、mAP50-95={fmt_pct(full.mAP50_95)}、"
            f"Recall={fmt_pct(full.recall)}**。相较基线，mAP50 提升 **{full_gain50:.2f} 个百分点**，"
            f"mAP50-95 提升 **{full_gain95:.2f} 个百分点**，Recall 提升 **{full_gain_recall:.2f} 个百分点**。"
        ),
        "",
        (
            f"按原图目标面积进行 COCO 分桶后，完整模型的 Small AP 相对基线提升 **{small_ap_gain:.2f} 个百分点**，"
            f"Small AP50 提升 **{small_ap50_gain:.2f} 个百分点**；但 Large AP 变化为 **{large_ap_gain:+.2f} 个百分点**。"
            "因此，当前结果支持模型对小目标更有利，但这种收益并非对所有尺度都一致。"
        ),
        "",
        (
            f"分阶段结果表明：MF-FPN 带来 {mffpn_delta:+.2f} 个百分点的 mAP50 变化；在 MF-FPN 基础上加入 LSCD 后"
            f"变化为 {lscd_delta:+.2f} 个百分点；进一步加入 Inner-WIoU 后变化为 {inner_delta:+.2f} 个百分点。"
            "因此，当前复现证据支持 MF-FPN 与 Inner-WIoU 的有效性，但 LSCD 的独立增益尚未复现，需要继续做结构参数对照。"
        ),
        "",
        f"论文报告的完整模型 mAP50 为 44.30%，当前统一复测与其相差 **{paper_gap:.2f} 个百分点**。该差距应被视为复现边界，而不是忽略或用训练过程中的峰值替代。",
        "",
        "## 2. 实验设置与公平性约束",
        "",
        "四个检查点均使用相同数据划分、输入尺寸、置信度阈值、NMS IoU 阈值和最大检测数，并分别在独立 Python 进程中加载，避免自定义 LSCD 检测头与 Inner-WIoU 运行时注册相互影响。",
        "",
        "| 项目 | 统一设置 |",
        "|---|---|",
        f"| 数据集 | VisDrone2019 `{protocol['split']}`（{protocol['dataset_count']['images']} 张图像，{protocol['dataset_count']['labels']} 个标签文件） |",
        f"| 输入尺寸 | {protocol['imgsz']} × {protocol['imgsz']} |",
        f"| Batch size | {protocol['batch']} |",
        f"| Precision mode | {'FP16' if protocol['half'] else 'FP32'} |",
        f"| conf / IoU / max_det | {protocol['conf']} / {protocol['iou']} / {protocol['max_det']} |",
        f"| GPU | {environment['gpu']} |",
        f"| 软件 | Python {environment['python']}；PyTorch {environment['torch']}；Ultralytics {environment['ultralytics']} |",
        "| 随机性 | 推理阶段无数据增强；四组采用相同确定性验证协议 |",
        "",
        "## 3. 统一复测结果",
        "",
        "表中精度来自本次统一复测；“最佳轮次”来自各训练日志中 mAP50-95 最大的轮次。推理耗时包含当前软件环境下的单图前向时间，不等同于端到端部署 FPS。",
        "",
        markdown_table(summary),
        "",
        f"![四组模型统一指标对比]({figure('overall_metrics_comparison.png')})",
        "",
        f"![逐阶段消融增量]({figure('ablation_map_deltas.png')})",
        "",
        "## 4. 训练过程与收敛情况",
        "",
        "四组训练均完成 150 轮。下图直接读取各运行目录中的 `results.csv`，展示验证 mAP 的完整变化轨迹；统一复测结果以 `best.pt` 为准，因此不应使用最后一轮替代最佳检查点。",
        "",
        f"![四组训练mAP曲线]({figure('training_map_curves.png')})",
        "",
        f"![完整模型训练面板]({figure('full_training_dashboard.png')})",
        "",
        "## 5. 分类别结果",
        "",
        "分类别 AP 可以揭示总体均值背后的收益来源。下表给出四阶段 AP50，以及完整模型相对基线的变化。",
        "",
        per_class_markdown(per_class),
        "",
        f"![分类别AP50对比]({figure('per_class_ap50_comparison.png')})",
        "",
        f"![完整模型相对基线的分类别变化]({figure('per_class_full_vs_baseline_delta.png')})",
        "",
        "## 6. 按目标尺寸分桶评估",
        "",
        (
            "本节在原始图像坐标系中采用 COCO 面积定义：Small＜32²像素，Medium为32²–96²像素，"
            f"Large≥96²像素。验证集共包含 {size_counts['small']} 个小目标、{size_counts['medium']} 个中目标和"
            f" {size_counts['large']} 个大目标。所有模型使用同一份转换标注、conf={protocol['conf']}、"
            f"NMS IoU={protocol['iou']} 和 max_det={protocol['max_det']}。"
        ),
        "",
        size_markdown(size_metrics),
        "",
        f"![四组模型按目标尺寸的COCO指标]({figure('size_stratified_metrics.png')})",
        "",
        (
            f"完整模型的 Small AP 从 {fmt_pct(size_lookup.loc[('baseline', 'small'), 'ap'])} 提升至 "
            f"{fmt_pct(size_lookup.loc[('full', 'small'), 'ap'])}，Small AP50 从 "
            f"{fmt_pct(size_lookup.loc[('baseline', 'small'), 'ap50'])} 提升至 "
            f"{fmt_pct(size_lookup.loc[('full', 'small'), 'ap50'])}。MF-FPN 是小目标收益的主要来源；"
            "LSCD 阶段略有回落，Inner-WIoU 随后恢复并进一步提升。Large AP 相对基线下降，说明高分辨率检测层和轻量检测头"
            "可能改变了尺度间的能力分配，后续优化不能只观察总体 mAP。"
        ),
        "",
        "这里的 COCO AP 使用 pycocotools 在原图面积上独立计算，与 Ultralytics 类别均值的实现细节略有不同；四模型之间的尺寸比较使用完全相同的评估器，因此横向增量有效。",
        "",
        "## 7. 定性结果与混淆矩阵",
        "",
        "以下可视化来自同一验证批次，适合用于观察漏检、密集遮挡和相邻类别混淆。图片只能辅助解释，定量结论仍以统一复测表为准。",
        "",
        f"![基线与完整模型预测对照]({figure('qualitative_baseline_vs_full.jpg')})",
        "",
        f"![基线归一化混淆矩阵]({figure('baseline_confusion_matrix_normalized.png')})",
        "",
        f"![完整模型归一化混淆矩阵]({figure('full_confusion_matrix_normalized.png')})",
        "",
        "## 8. 诊断与下一步实验",
        "",
        "1. **LSCD 参数对照。** 当前累计阶段在加入 LSCD 后未获得正向 mAP50 增益。下一步应固定其余条件，对隐藏通道宽度、GroupNorm 分组数和初始化方式做 30–50 轮筛选，再对最佳配置训练 150 轮。",
        "2. **尺度权衡验证。** 当前 Small AP 获得提升而 Large AP 下降。应通过保留一个低分辨率检测层、调整尺度损失权重或增加尺度均衡蒸馏，验证能否保住小目标收益并恢复大目标性能。",
        "3. **重复实验。** 至少对基线和完整模型增加两个随机种子，报告均值与标准差，避免把单次波动写成确定性增益。",
        "4. **部署测量。** 在 PyTorch、ONNX 或 TensorRT 下分别测量预热后的吞吐率、端到端延迟和显存峰值，完善轻量化证据。",
        "",
        "## 9. 可追溯产物",
        "",
        "- 统一指标：`metrics/summary.csv`",
        "- 分类别指标：`metrics/per_class_metrics.csv`",
        "- 分尺寸指标：`metrics/size_metrics.csv`、`metrics/per_class_size_metrics.csv`",
        "- 每组完整结构化记录：`metrics/*_val.json`",
        "- 每组控制台日志：`logs/*_val.log`",
        "- 分尺寸评估记录与日志：`metrics/*_size_val.json`、`logs/*_size_val.log`",
        "- 跨领域小目标方法整理：[`../small_object_detection_methods.md`](../small_object_detection_methods.md)",
        "- Ultralytics 原生评估图片：对应 JSON 中的 `save_dir`，本报告复制了基线与完整模型的代表性图片。",
        "- 检查点完整性：每份 JSON 均记录绝对路径、文件大小和 SHA-256。",
        "",
        "## 10. 主张—证据映射",
        "",
        "| 主张 | 证据 | 状态 |",
        "|---|---|---|",
        f"| 完整模型优于当前基线 | mAP50 {full_gain50:+.2f} pp，mAP50-95 {full_gain95:+.2f} pp | 已支持 |",
        f"| Inner-WIoU 对当前累计模型有效 | 相比 LSCD 阶段，mAP50 {inner_delta:+.2f} pp | 已支持（单随机种子） |",
        f"| LSCD 独立提升精度 | 相比 MF-FPN 阶段，mAP50 {lscd_delta:+.2f} pp | 未支持，需新实验 |",
        f"| 完整模型改善小目标 | Small AP {small_ap_gain:+.2f} pp，Small AP50 {small_ap50_gain:+.2f} pp | 已支持（单随机种子） |",
        f"| 完整模型对各尺度均有利 | Large AP {large_ap_gain:+.2f} pp | 未支持 |",
        "| 模型具有部署轻量化优势 | 已有参数量/GFLOPs，但无统一预热FPS与端到端延迟 | 部分支持 |",
        "",
        "## 11. 审稿人式自检",
        "",
        "| 维度 | 判断 | 需要补充的证据 |",
        "|---|---|---|",
        "| 贡献 | 需要修订 | 明确本研究相对原论文是复现、验证还是进一步改进 |",
        "| 写作清晰度 | 通过 | 保持各阶段名称和累计关系一致 |",
        "| 实验强度 | 需要新实验 | 增加随机种子并报告均值±标准差 |",
        "| 评估完整性 | 需要修订 | 已补充尺寸分桶；仍需测试集、困难场景和定性失败案例 |",
        "| 方法设计可靠性 | 需要新实验 | 解释 LSCD 负增益并验证关键结构参数 |",
        "",
        "---",
        "",
        "本报告仅陈述当前日志能够支持的结论；未进行的实验均明确标记为待验证。",
    ]
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "unified_evaluation")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    figures = report_dir / "figures"
    (report_dir / "metrics").mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    records = load_records(report_dir)
    size_records = load_size_records(report_dir)
    frames, best_rows = load_training_rows()
    per_class = per_class_frame(records)
    size_metrics, per_class_size_metrics = size_frames(size_records)
    summary = write_csvs(records, frames, best_rows, per_class, report_dir)
    size_metrics.to_csv(report_dir / "metrics" / "size_metrics.csv", index=False, encoding="utf-8-sig")
    per_class_size_metrics.to_csv(
        report_dir / "metrics" / "per_class_size_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_overall(records, figures / "overall_metrics_comparison.png")
    plot_deltas(records, figures / "ablation_map_deltas.png")
    plot_training_curves(frames, figures / "training_map_curves.png")
    plot_per_class(per_class, figures / "per_class_ap50_comparison.png")
    plot_class_delta(per_class, figures / "per_class_full_vs_baseline_delta.png")
    plot_size_metrics(size_metrics, figures / "size_stratified_metrics.png")
    copy_evaluation_assets(records, figures)
    make_qualitative_comparison(figures)
    generate_markdown(records, summary, per_class, size_records, size_metrics, report_dir)
    print(f"Report written to {report_dir / 'README.md'}")


if __name__ == "__main__":
    main()
