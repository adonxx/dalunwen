"""Generate the focused 50-epoch adaptive weighted-fusion ablation report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import build_model  # noqa: E402
from drone_yolo.fusion import AdaptiveWeightedConcat  # noqa: E402


MODELS = OrderedDict(
    [
        (
            "control",
            {
                "label": "MF-FPN equal concat",
                "run": "scale_mffpn_p2p4_control_50e",
                "overall": ROOT / "reports" / "scale_ablation" / "metrics" / "mffpn_p2p4_control_val.json",
                "size": ROOT / "reports" / "scale_ablation" / "metrics" / "mffpn_p2p4_control_size_val.json",
            },
        ),
        (
            "weighted",
            {
                "label": "MF-FPN adaptive weighted concat",
                "run": "mffpn_weighted_50e",
                "overall": None,
                "size": None,
            },
        ),
    ]
)
COLORS = ["#5B6C8F", "#EB5757"]
FUSION_NODES = [
    ("TD P3", "upsampled P4", "lateral P3"),
    ("TD P2", "upsampled P3", "lateral P2"),
    ("BU P3", "downsampled P2", "top-down P3"),
    ("BU P4", "downsampled P3", "deep P4"),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


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
    frames = {}
    for key, meta in MODELS.items():
        overall_path = meta["overall"] or report_dir / "metrics" / "mffpn_weighted_val.json"
        size_path = meta["size"] or report_dir / "metrics" / "mffpn_weighted_size_val.json"
        overall_record = load_json(overall_path)
        size_record = load_json(size_path)
        frame = pd.read_csv(ROOT / "runs" / meta["run"] / "results.csv")
        frame.columns = [column.strip() for column in frame.columns]
        frames[key] = frame
        first_50 = frame.iloc[:50]
        best = first_50.loc[first_50["metrics/mAP50-95(B)"].idxmax()]
        overall = overall_record["overall"]
        area = size_record["overall_by_area"]
        rows.append(
            {
                "variant": key,
                "label": meta["label"],
                "completed_epochs": len(frame),
                "best_epoch": int(best["epoch"]),
                "training_peak_map50": float(best["metrics/mAP50(B)"]),
                "training_peak_map50_95": float(best["metrics/mAP50-95(B)"]),
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
                "checkpoint_sha256": overall_record["checkpoint"]["sha256"],
            }
        )
    return pd.DataFrame(rows), frames


def extract_fusion_weights(report_dir: Path) -> pd.DataFrame:
    checkpoint = ROOT / "runs" / "mffpn_weighted_50e" / "weights" / "best.pt"
    model = build_model("mffpn_weighted", weights=str(checkpoint)).model
    modules = [module for module in model.model if isinstance(module, AdaptiveWeightedConcat)]
    if len(modules) != len(FUSION_NODES):
        raise RuntimeError(f"expected {len(FUSION_NODES)} weighted fusion nodes, found {len(modules)}")
    rows = []
    for (node, branch_1, branch_2), module in zip(FUSION_NODES, modules):
        weights = module.normalized_weights().detach().cpu().tolist()
        rows.append(
            {
                "node": node,
                "branch_1": branch_1,
                "branch_1_weight": weights[0],
                "branch_2": branch_2,
                "branch_2_weight": weights[1],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(report_dir / "metrics" / "learned_fusion_weights.csv", index=False, encoding="utf-8-sig")
    return frame


def plot_training(frames: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for (key, frame), color in zip(frames.items(), COLORS):
        axes[0].plot(frame["epoch"], 100 * frame["metrics/mAP50(B)"], color=color, label=MODELS[key]["label"])
        axes[1].plot(frame["epoch"], 100 * frame["metrics/mAP50-95(B)"], color=color, label=MODELS[key]["label"])
    axes[0].set_title("Validation mAP50")
    axes[1].set_title("Validation mAP50-95")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_metrics(summary: pd.DataFrame, output: Path) -> None:
    metrics = ["map50_95", "small_ap", "medium_ap", "large_ap"]
    labels = ["mAP50-95", "Small AP", "Medium AP", "Large AP"]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for index, (row, color) in enumerate(zip(summary.itertuples(index=False), COLORS)):
        values = [100 * getattr(row, metric) for metric in metrics]
        bars = ax.bar(x + (index - 0.5) * width, values, width, label=row.label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("COCO metric (%)")
    ax.set_title("Equal vs adaptive weighted feature fusion (50 epochs)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_weights(weights: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(weights))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    first = ax.bar(x - width / 2, weights["branch_1_weight"], width, label="Branch 1", color="#2F80ED")
    second = ax.bar(x + width / 2, weights["branch_2_weight"], width, label="Branch 2", color="#F2994A")
    ax.bar_label(first, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(second, fmt="%.3f", padding=2, fontsize=8)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1, label="Equal-fusion initialization")
    ax.set_xticks(x, weights["node"])
    ax.set_ylabel("Mean-normalized branch weight")
    ax.set_title("Learned branch importance at the best checkpoint")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def result_table(summary: pd.DataFrame) -> str:
    lines = [
        "| 结构 | mAP50 | mAP50-95 | Small AP | Small AP50 | Medium AP | Large AP | 参数量(M) | GFLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.label} | {fmt_pct(row.map50)} | {fmt_pct(row.map50_95)} | {fmt_pct(row.small_ap)} | "
            f"{fmt_pct(row.small_ap50)} | {fmt_pct(row.medium_ap)} | {fmt_pct(row.large_ap)} | "
            f"{row.parameters / 1e6:.6f} | {row.gflops:.2f} |"
        )
    return "\n".join(lines)


def weights_table(weights: pd.DataFrame) -> str:
    lines = [
        "| 融合节点 | 分支1及权重 | 分支2及权重 |",
        "|---|---:|---:|",
    ]
    for row in weights.itertuples(index=False):
        lines.append(
            f"| {row.node} | {row.branch_1}: {row.branch_1_weight:.4f} | "
            f"{row.branch_2}: {row.branch_2_weight:.4f} |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, weights: pd.DataFrame, report_dir: Path) -> None:
    indexed = summary.set_index("variant")
    control = indexed.loc["control"]
    weighted = indexed.loc["weighted"]
    deltas = {
        "map50": 100 * (weighted.map50 - control.map50),
        "map50_95": 100 * (weighted.map50_95 - control.map50_95),
        "small_ap": 100 * (weighted.small_ap - control.small_ap),
        "small_ap50": 100 * (weighted.small_ap50 - control.small_ap50),
        "medium_ap": 100 * (weighted.medium_ap - control.medium_ap),
        "large_ap": 100 * (weighted.large_ap - control.large_ap),
        "parameters_pct": 100 * (weighted.parameters - control.parameters) / control.parameters,
        "gflops_pct": 100 * (weighted.gflops - control.gflops) / control.gflops,
    }
    criteria = {
        "small": deltas["small_ap"] >= 0.30,
        "overall": deltas["map50_95"] >= -0.10,
        "large": deltas["large_ap"] >= -1.00,
        "cost": abs(deltas["parameters_pct"]) <= 0.1 and abs(deltas["gflops_pct"]) <= 0.1,
    }
    promote = all(criteria.values())
    decision = (
        "达到预注册晋级条件，建议进入150轮完整训练，并增加至少3个随机种子。"
        if promote
        else "未同时达到预注册晋级条件，不建议直接投入150轮；应依据失败项决定调整或停止该方向。"
    )
    criteria_rows = [
        ("Small AP至少+0.30 pp", deltas["small_ap"], criteria["small"]),
        ("mAP50-95下降不超过0.10 pp", deltas["map50_95"], criteria["overall"]),
        ("Large AP下降不超过1.00 pp", deltas["large_ap"], criteria["large"]),
    ]
    lines = [
        "# 自适应加权 MF-FPN：50轮消融报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 对照性质：仅将4个普通Concat替换为均值归一化的AdaptiveWeightedConcat；其余训练与验证协议一致。",
        "",
        "## 1. 结果",
        "",
        result_table(summary),
        "",
        "![指标对比](figures/weighted_fusion_metrics.png)",
        "",
        "![训练曲线](figures/weighted_fusion_training.png)",
        "",
        "## 2. 相对公平对照的变化",
        "",
        f"加权融合相对等权融合的 mAP50 变化为 **{deltas['map50']:+.2f} pp**，mAP50-95为 **{deltas['map50_95']:+.2f} pp**；Small AP为 **{deltas['small_ap']:+.2f} pp**、Small AP50为 **{deltas['small_ap50']:+.2f} pp**、Medium AP为 **{deltas['medium_ap']:+.2f} pp**、Large AP为 **{deltas['large_ap']:+.2f} pp**。参数量变化 **{deltas['parameters_pct']:+.4f}%**，GFLOPs变化 **{deltas['gflops_pct']:+.4f}%**。",
        "",
        "## 3. 学得的融合权重",
        "",
        weights_table(weights),
        "",
        "每个节点的两项权重均值为1；高于1表示该分支相对等权初始化被增强，低于1表示被抑制。权重用于解释模型行为，不单独证明因果收益。",
        "",
        "![融合权重](figures/learned_fusion_weights.png)",
        "",
        "## 4. 预注册判定",
        "",
        "| 条件 | 实际变化 | 是否通过 |",
        "|---|---:|---:|",
        *[f"| {name} | {value:+.2f} pp | {'是' if passed else '否'} |" for name, value, passed in criteria_rows],
        f"| 参数量和GFLOPs变化均不超过0.1% | {deltas['parameters_pct']:+.4f}% / {deltas['gflops_pct']:+.4f}% | {'是' if criteria['cost'] else '否'} |",
        "",
        f"**结论：{decision}**",
        "",
        "## 5. 主张—证据映射",
        "",
        "| 主张 | 证据 | 状态 |",
        "|---|---|---|",
        f"| 加权融合改善小目标检测 | Small AP变化 {deltas['small_ap']:+.2f} pp | {'50轮支持' if criteria['small'] else '当前不支持'} |",
        f"| 加权融合不增加实质部署开销 | 参数 {deltas['parameters_pct']:+.4f}%，GFLOPs {deltas['gflops_pct']:+.4f}% | {'支持' if criteria['cost'] else '不支持'} |",
        "| 加权融合是最终有效贡献 | 尚无150轮、多随机种子与延迟结果 | 不能声称 |",
        "",
        "## 6. 可追溯产物",
        "",
        "- 预注册方案：`EXPERIMENT_PLAN.md`",
        "- 日志：`logs/`",
        "- 指标与权重：`metrics/`",
        "- 最佳检查点：`../../runs/mffpn_weighted_50e/weights/best.pt`",
        "",
        "本报告只把50轮结果用于结构筛选，并明确保留单随机种子的证据限制。",
    ]
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "weighted_fusion")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    figures = report_dir / "figures"
    metrics = report_dir / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    summary, frames = collect(report_dir)
    summary.to_csv(metrics / "weighted_fusion_summary.csv", index=False, encoding="utf-8-sig")
    weights = extract_fusion_weights(report_dir)
    plot_training(frames, figures / "weighted_fusion_training.png")
    plot_metrics(summary, figures / "weighted_fusion_metrics.png")
    plot_weights(weights, figures / "learned_fusion_weights.png")
    write_report(summary, weights, report_dir)
    print(f"Weighted-fusion report written to {report_dir / 'README.md'}")


if __name__ == "__main__":
    main()
