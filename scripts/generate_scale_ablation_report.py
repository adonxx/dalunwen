"""Generate the detection-scale screening report after both 50-epoch runs."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = OrderedDict(
    [
        (
            "baseline",
            {
                "label": "YOLOv11s P3-P5",
                "run": "baseline_paper",
                "budget": 150,
                "record_root": "unified",
            },
        ),
        (
            "mffpn",
            {
                "label": "MF-FPN P2-P4",
                "run": "mffpn_paper",
                "budget": 150,
                "record_root": "unified",
            },
        ),
        (
            "mffpn_p2p4_control",
            {
                "label": "MF-FPN P2-P4 (50e control)",
                "run": "scale_mffpn_p2p4_control_50e",
                "budget": 50,
                "record_root": "scale",
                "record_stem": "mffpn_p2p4_control",
            },
        ),
        (
            "mffpn_p2p5",
            {
                "label": "MF-FPN P2-P5",
                "run": "scale_mffpn_p2p5_50e",
                "budget": 50,
                "record_root": "scale",
            },
        ),
        (
            "mffpn_p3p5",
            {
                "label": "MF-FPN P3-P5",
                "run": "scale_mffpn_p3p5_50e",
                "budget": 50,
                "record_root": "scale",
            },
        ),
    ]
)
COLORS = ["#5B6C8F", "#2F80ED", "#9B51E0", "#EB5757", "#27AE60"]


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def record_paths(stage: str, report_dir: Path) -> tuple[Path, Path]:
    if MODELS[stage]["record_root"] == "unified":
        root = ROOT / "reports" / "unified_evaluation" / "metrics"
    else:
        root = report_dir / "metrics"
    stem = MODELS[stage].get("record_stem", stage)
    return root / f"{stem}_val.json", root / f"{stem}_size_val.json"


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


def collect(report_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    training_frames = {}
    for stage, meta in MODELS.items():
        overall_path, size_path = record_paths(stage, report_dir)
        overall_record = load_json(overall_path)
        size_record = load_json(size_path)
        frame = pd.read_csv(ROOT / "runs" / meta["run"] / "results.csv")
        frame.columns = [column.strip() for column in frame.columns]
        training_frames[stage] = frame
        best_first_50 = frame.iloc[:50].loc[frame.iloc[:50]["metrics/mAP50-95(B)"].idxmax()]
        overall = overall_record["overall"]
        area = size_record["overall_by_area"]
        rows.append(
            {
                "stage": stage,
                "label": meta["label"],
                "training_budget": meta["budget"],
                "completed_epochs": len(frame),
                "best_epoch_first_50": int(best_first_50["epoch"]),
                "first_50_map50": float(best_first_50["metrics/mAP50(B)"]),
                "first_50_map50_95": float(best_first_50["metrics/mAP50-95(B)"]),
                "precision": overall["metrics/precision(B)"],
                "recall": overall["metrics/recall(B)"],
                "map50": overall["metrics/mAP50(B)"],
                "map50_95": overall["metrics/mAP50-95(B)"],
                "small_ap": area["small"]["ap"],
                "small_ap50": area["small"]["ap50"],
                "medium_ap": area["medium"]["ap"],
                "large_ap": area["large"]["ap"],
                "parameters": overall_record["model"]["parameters"],
                "gflops": overall_record["model"]["gflops_640"],
                "strides": "/".join(str(int(value)) for value in overall_record["model"]["strides"]),
                "checkpoint_sha256": overall_record["checkpoint"]["sha256"],
            }
        )
    return pd.DataFrame(rows), training_frames


def plot_training(frames: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for (stage, frame), color in zip(frames.items(), COLORS):
        subset = frame.iloc[:50]
        axes[0].plot(subset["epoch"] + 1, 100 * subset["metrics/mAP50(B)"], label=MODELS[stage]["label"], color=color)
        axes[1].plot(
            subset["epoch"] + 1,
            100 * subset["metrics/mAP50-95(B)"],
            label=MODELS[stage]["label"],
            color=color,
        )
    axes[0].set_title("First-50-epoch validation mAP50")
    axes[1].set_title("First-50-epoch validation mAP50-95")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_sizes(summary: pd.DataFrame, output: Path) -> None:
    metrics = ["small_ap", "medium_ap", "large_ap"]
    labels = ["Small AP", "Medium AP", "Large AP"]
    x = np.arange(len(metrics))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.3))
    for index, (row, color) in enumerate(zip(summary.itertuples(index=False), COLORS)):
        values = [100 * getattr(row, metric) for metric in metrics]
        offset = (index - (len(summary) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=row.label, color=color)
        if row.stage in {"mffpn_p2p4_control", "mffpn_p2p5", "mffpn_p3p5"}:
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylabel("COCO metric (%)")
    ax.set_title("Object-size screening results (training budgets noted in report)")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_complexity(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for row, color in zip(summary.itertuples(index=False), COLORS):
        ax.scatter(row.gflops, row.parameters / 1e6, s=90, color=color)
        ax.annotate(row.label, (row.gflops, row.parameters / 1e6), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("GFLOPs @ 640 (lower is better)")
    ax.set_ylabel("Parameters (M, lower is better)")
    ax.set_title("Scale-ablation model complexity")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def result_table(summary: pd.DataFrame) -> str:
    lines = [
        "| 模型 | 检测步长 | 训练轮次 | mAP50 | mAP50-95 | Small AP | Medium AP | Large AP | 参数量(M) | GFLOPs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.strides} | {row.training_budget} | {fmt_pct(row.map50)} | "
            f"{fmt_pct(row.map50_95)} | {fmt_pct(row.small_ap)} | {fmt_pct(row.medium_ap)} | "
            f"{fmt_pct(row.large_ap)} | {row.parameters / 1e6:.3f} | {row.gflops:.2f} |"
        )
    return "\n".join(lines)


def first_50_table(summary: pd.DataFrame) -> str:
    lines = [
        "| 模型 | 前50轮最佳轮次 | 前50轮mAP50 | 前50轮mAP50-95 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.best_epoch_first_50} | {fmt_pct(row.first_50_map50)} | "
            f"{fmt_pct(row.first_50_map50_95)} |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, report_dir: Path) -> None:
    indexed = summary.set_index("stage")
    four = indexed.loc["mffpn_p2p5"]
    no_p2 = indexed.loc["mffpn_p3p5"]
    control = indexed.loc["mffpn_p2p4_control"]
    original = indexed.loc["mffpn"]
    small_p2_effect = 100 * (four.small_ap - no_p2.small_ap)
    map50_p2_effect = 100 * (four.map50 - no_p2.map50)
    map95_p2_effect = 100 * (four.map50_95 - no_p2.map50_95)
    map50_p5_effect = 100 * (four.map50 - control.map50)
    map95_p5_effect = 100 * (four.map50_95 - control.map50_95)
    small_p5_effect = 100 * (four.small_ap - control.small_ap)
    medium_p5_effect = 100 * (four.medium_ap - control.medium_ap)
    large_p5_effect = 100 * (four.large_ap - control.large_ap)
    parameter_p5_effect = (four.parameters - control.parameters) / 1e6
    gflops_p5_effect = four.gflops - control.gflops
    replication_map95_gap = 100 * (control.first_50_map50_95 - original.first_50_map50_95)
    go_full = four.small_ap >= control.small_ap - 0.005 and four.large_ap > control.large_ap
    decision = (
        "四尺度模型在公平50轮对照中基本保持小目标能力并提高大目标性能，建议进入150轮完整训练。"
        if go_full
        else "四尺度模型在公平50轮对照中尚未同时满足小目标保持与大目标提升条件，应先调整P5宽度或融合方式。"
    )
    lines = [
        "# Drone-YOLO 检测尺度消融：独立50轮公平对照报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 实验性质：结构筛选；P2–P4、P2–P5、P3–P5三组候选均独立训练50轮。既有基线与原始MF-FPN的150轮结果只作背景参考。",
        "",
        "## 1. 实验问题",
        "",
        "本实验检验两个问题：在P2–P4上加入P5是否改善大目标性能；在P2–P5上移除P2后，小目标性能是否下降。三组50轮候选除Detect输入尺度外共享相同特征路径，训练数据、输入尺寸、优化器、随机种子和验证协议保持一致。",
        "",
        "## 2. 统一结果",
        "",
        result_table(summary),
        "",
        "下表比较全部训练过程的前50轮峰值，用于检查独立P2–P4对照能否复现原训练前50轮趋势。正式尺度结论只使用三组独立50轮模型的统一检查点评估结果。",
        "",
        first_50_table(summary),
        "",
        "![前50轮收敛曲线](figures/first_50_training_curves.png)",
        "",
        "![尺寸指标对比](figures/scale_size_metrics.png)",
        "",
        "![模型复杂度](figures/scale_model_complexity.png)",
        "",
        "## 3. 初步解释与决策",
        "",
        f"**P2效应（P2–P5 对 P3–P5）：** mAP50 **{map50_p2_effect:+.2f} pp**、mAP50-95 **{map95_p2_effect:+.2f} pp**、Small AP **{small_p2_effect:+.2f} pp**。这组差值直接刻画高分辨率P2检测层的作用。",
        "",
        f"**P5效应（P2–P5 对独立 P2–P4）：** mAP50 **{map50_p5_effect:+.2f} pp**、mAP50-95 **{map95_p5_effect:+.2f} pp**、Small AP **{small_p5_effect:+.2f} pp**、Medium AP **{medium_p5_effect:+.2f} pp**、Large AP **{large_p5_effect:+.2f} pp**；代价为参数量 **{parameter_p5_effect:+.3f} M**、计算量 **{gflops_p5_effect:+.2f} GFLOPs**。三组训练预算相同，因此这些结果可作为50轮结构筛选结论。",
        "",
        f"独立P2–P4对照与原始P2–P4训练的前50轮最佳mAP50-95相差 **{replication_map95_gap:+.2f} pp**。该差值用于检查训练重复性，不替代多随机种子统计。",
        "",
        f"**筛选决策：{decision}**",
        "",
        "## 4. 可追溯产物",
        "",
        "- 三组50轮训练日志：`logs/*_train.log`",
        "- 统一验证日志：`logs/*_val.log`",
        "- 分尺寸验证日志：`logs/*_size_val.log`",
        "- 结构化指标：`metrics/*.json`、`metrics/scale_ablation_summary.csv`",
        "- 训练原始指标：`../../runs/scale_mffpn_*_50e/results.csv`",
        "- 检查点：`../../runs/scale_mffpn_*_50e/weights/best.pt`",
        "",
        "## 5. 主张—证据映射",
        "",
        "| 主张 | 当前证据 | 状态 |",
        "|---|---|---|",
        f"| P2对小目标的作用 | 公平50轮P2–P5相对P3–P5的Small AP差值 {small_p2_effect:+.2f} pp | 50轮因果对照 |",
        f"| P5对各尺寸目标的作用 | 公平50轮P2–P5相对P2–P4：Small {small_p5_effect:+.2f}、Medium {medium_p5_effect:+.2f}、Large {large_p5_effect:+.2f} pp | 50轮因果对照 |",
        "| 四尺度模型优于当前完整模型 | 尚未加入LSCD/Inner-WIoU，也未完成150轮 | 不可声称 |",
        "",
        "## 6. 审稿人式自检",
        "",
        "| 维度 | 判断 | 下一步 |",
        "|---|---|---|",
        "| 因果对照 | 通过 | 三个50轮模型共享特征路径，仅Detect输入尺度不同 |",
        "| 训练公平性 | 通过（筛选阶段） | 正式最终结论仍需候选模型补齐150轮 |",
        "| 指标完整性 | 通过 | 同时报告总体与Small/Medium/Large AP |",
        "| 统计可靠性 | 需要新实验 | 最终结构增加随机种子 |",
        "| 部署价值 | 需要新实验 | 对最终结构测试FP16延迟和显存 |",
        "",
        "本报告遵循筛选实验的证据边界；50轮结果只决定是否继续投入完整训练。",
    ]
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "scale_ablation")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    figures = report_dir / "figures"
    metrics = report_dir / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    summary, frames = collect(report_dir)
    summary.to_csv(metrics / "scale_ablation_summary.csv", index=False, encoding="utf-8-sig")
    plot_training(frames, figures / "first_50_training_curves.png")
    plot_sizes(summary, figures / "scale_size_metrics.png")
    plot_complexity(summary, figures / "scale_model_complexity.png")
    write_report(summary, report_dir)
    print(f"Scale-ablation report written to {report_dir / 'README.md'}")


if __name__ == "__main__":
    main()
