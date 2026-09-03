"""Generate the three-seed paired adaptive-fusion stability report."""

from __future__ import annotations

import argparse
import json
import sys
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


VARIANTS = {
    "control": "MF-FPN equal concat",
    "weighted": "MF-FPN adaptive weighted concat",
}
COLORS = {"control": "#5B6C8F", "weighted": "#EB5757"}
METRICS = ["map50_95", "small_ap", "medium_ap", "large_ap"]
METRIC_LABELS = ["mAP50-95", "Small AP", "Medium AP", "Large AP"]
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


def paths_for(seed: int, variant: str, report_dir: Path) -> tuple[Path, Path, Path]:
    if seed == 0 and variant == "control":
        metric_root = ROOT / "reports" / "scale_ablation" / "metrics"
        overall = metric_root / "mffpn_p2p4_control_val.json"
        size = metric_root / "mffpn_p2p4_control_size_val.json"
        run = ROOT / "runs" / "scale_mffpn_p2p4_control_50e"
    elif seed == 0 and variant == "weighted":
        metric_root = ROOT / "reports" / "weighted_fusion" / "metrics"
        overall = metric_root / "mffpn_weighted_val.json"
        size = metric_root / "mffpn_weighted_size_val.json"
        run = ROOT / "runs" / "mffpn_weighted_50e"
    else:
        stem = "mffpn" if variant == "control" else "mffpn_weighted"
        overall = report_dir / "metrics" / f"{stem}_seed{seed}_val.json"
        size = report_dir / "metrics" / f"{stem}_seed{seed}_size_val.json"
        run = ROOT / "runs" / f"{stem}_seed{seed}_50e"
    return overall, size, run


def checkpoint_for_weighted_seed(seed: int) -> Path:
    run_name = "mffpn_weighted_50e" if seed == 0 else f"mffpn_weighted_seed{seed}_50e"
    return ROOT / "runs" / run_name / "weights" / "best.pt"


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


def collect(report_dir: Path) -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame]]:
    rows = []
    training_frames = {}
    for seed in (0, 1, 2):
        for variant in ("control", "weighted"):
            overall_path, size_path, run_dir = paths_for(seed, variant, report_dir)
            overall_record = load_json(overall_path)
            size_record = load_json(size_path)
            frame = pd.read_csv(run_dir / "results.csv")
            frame.columns = [column.strip() for column in frame.columns]
            training_frames[(seed, variant)] = frame.iloc[:50].copy()
            best = frame.iloc[:50].loc[frame.iloc[:50]["metrics/mAP50-95(B)"].idxmax()]
            overall = overall_record["overall"]
            area = size_record["overall_by_area"]
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "label": VARIANTS[variant],
                    "completed_epochs": len(frame),
                    "best_epoch": int(best["epoch"]),
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
    return pd.DataFrame(rows), training_frames


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aggregate_metrics = ["map50", "map50_95", "small_ap", "small_ap50", "medium_ap", "large_ap"]
    for variant in ("control", "weighted"):
        subset = summary[summary["variant"] == variant]
        row = {"variant": variant, "label": VARIANTS[variant]}
        for metric in aggregate_metrics:
            row[f"{metric}_mean"] = subset[metric].mean()
            row[f"{metric}_std"] = subset[metric].std(ddof=1)
        row["parameters"] = int(subset["parameters"].iloc[0])
        row["gflops"] = float(subset["gflops"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    control = summary[summary["variant"] == "control"].set_index("seed")
    weighted = summary[summary["variant"] == "weighted"].set_index("seed")
    rows = []
    for seed in (0, 1, 2):
        row = {"seed": seed}
        for metric in ["map50", "map50_95", "small_ap", "small_ap50", "medium_ap", "large_ap"]:
            row[f"control_{metric}"] = control.loc[seed, metric]
            row[f"weighted_{metric}"] = weighted.loc[seed, metric]
            row[f"delta_{metric}"] = weighted.loc[seed, metric] - control.loc[seed, metric]
        rows.append(row)
    return pd.DataFrame(rows)


def extract_fusion_weights(report_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in (0, 1, 2):
        model = build_model("mffpn_weighted", weights=str(checkpoint_for_weighted_seed(seed))).model
        modules = [module for module in model.model if isinstance(module, AdaptiveWeightedConcat)]
        if len(modules) != len(FUSION_NODES):
            raise RuntimeError(f"seed {seed}: expected {len(FUSION_NODES)} weighted nodes, found {len(modules)}")
        for (node, branch_1, branch_2), module in zip(FUSION_NODES, modules):
            values = module.normalized_weights().detach().cpu().tolist()
            rows.append(
                {
                    "seed": seed,
                    "node": node,
                    "branch_1": branch_1,
                    "branch_1_weight": values[0],
                    "branch_2": branch_2,
                    "branch_2_weight": values[1],
                }
            )
    weights = pd.DataFrame(rows)
    weights.to_csv(report_dir / "metrics" / "fusion_weights_all_seeds.csv", index=False, encoding="utf-8-sig")
    return weights


def plot_metric_means(aggregated: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(METRICS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for index, row in enumerate(aggregated.itertuples(index=False)):
        means = [100 * getattr(row, f"{metric}_mean") for metric in METRICS]
        errors = [100 * getattr(row, f"{metric}_std") for metric in METRICS]
        bars = ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=row.label,
            color=COLORS[row.variant],
        )
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.set_xticks(x, METRIC_LABELS)
    ax.set_ylabel("Mean COCO metric across 3 seeds (%)")
    ax.set_title("Three-seed mean and standard deviation")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_paired_small(pairs: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for row in pairs.itertuples(index=False):
        values = [100 * row.control_small_ap, 100 * row.weighted_small_ap]
        color = "#27AE60" if row.delta_small_ap > 0 else "#EB5757"
        ax.plot([0, 1], values, marker="o", linewidth=2, color=color, label=f"seed={row.seed}")
    ax.set_xticks([0, 1], ["Equal concat", "Adaptive weighted concat"])
    ax.set_ylabel("Small AP (%)")
    ax.set_title("Paired Small AP change for each seed")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_training_mean(frames: dict[tuple[int, str], pd.DataFrame], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for variant in ("control", "weighted"):
        values = np.stack(
            [frames[(seed, variant)]["metrics/mAP50-95(B)"].to_numpy(dtype=float) for seed in (0, 1, 2)]
        )
        epochs = frames[(0, variant)]["epoch"].to_numpy(dtype=float)
        mean = 100 * values.mean(axis=0)
        std = 100 * values.std(axis=0, ddof=1)
        ax.plot(epochs, mean, color=COLORS[variant], label=VARIANTS[variant])
        ax.fill_between(epochs, mean - std, mean + std, color=COLORS[variant], alpha=0.16)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP50-95 (%)")
    ax.set_title("Mean training trajectory with ±1 SD across seeds")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_weight_stability(weights: pd.DataFrame, output: Path) -> None:
    stats = weights.groupby("node", sort=False)[["branch_1_weight", "branch_2_weight"]].agg(["mean", "std"])
    x = np.arange(len(stats))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    first = ax.bar(
        x - width / 2,
        stats[("branch_1_weight", "mean")],
        width,
        yerr=stats[("branch_1_weight", "std")],
        capsize=4,
        label="Branch 1",
        color="#2F80ED",
    )
    second = ax.bar(
        x + width / 2,
        stats[("branch_2_weight", "mean")],
        width,
        yerr=stats[("branch_2_weight", "std")],
        capsize=4,
        label="Branch 2",
        color="#F2994A",
    )
    ax.bar_label(first, fmt="%.3f", padding=4, fontsize=8)
    ax.bar_label(second, fmt="%.3f", padding=4, fontsize=8)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(x, stats.index)
    ax.set_ylabel("Mean-normalized weight across 3 seeds")
    ax.set_title("Fusion-weight stability (mean ± SD)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def format_mean_std(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def aggregate_table(aggregated: pd.DataFrame) -> str:
    lines = [
        "| 结构 | mAP50 | mAP50-95 | Small AP | Small AP50 | Medium AP | Large AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregated.itertuples(index=False):
        lines.append(
            f"| {row.label} | {format_mean_std(row.map50_mean, row.map50_std)} | "
            f"{format_mean_std(row.map50_95_mean, row.map50_95_std)} | "
            f"{format_mean_std(row.small_ap_mean, row.small_ap_std)} | "
            f"{format_mean_std(row.small_ap50_mean, row.small_ap50_std)} | "
            f"{format_mean_std(row.medium_ap_mean, row.medium_ap_std)} | "
            f"{format_mean_std(row.large_ap_mean, row.large_ap_std)} |"
        )
    return "\n".join(lines)


def paired_table(pairs: pd.DataFrame) -> str:
    lines = [
        "| Seed | ΔmAP50-95 | ΔSmall AP | ΔSmall AP50 | ΔMedium AP | ΔLarge AP |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {100 * row.delta_map50_95:+.2f} | {100 * row.delta_small_ap:+.2f} | "
            f"{100 * row.delta_small_ap50:+.2f} | {100 * row.delta_medium_ap:+.2f} | "
            f"{100 * row.delta_large_ap:+.2f} |"
        )
    means = pairs[[column for column in pairs.columns if column.startswith("delta_")]].mean()
    lines.append(
        f"| 平均 | {100 * means['delta_map50_95']:+.2f} | {100 * means['delta_small_ap']:+.2f} | "
        f"{100 * means['delta_small_ap50']:+.2f} | {100 * means['delta_medium_ap']:+.2f} | "
        f"{100 * means['delta_large_ap']:+.2f} |"
    )
    return "\n".join(lines)


def write_report(
    summary: pd.DataFrame,
    aggregated: pd.DataFrame,
    pairs: pd.DataFrame,
    weights: pd.DataFrame,
    report_dir: Path,
) -> None:
    del summary, weights
    mean_delta = pairs[[column for column in pairs.columns if column.startswith("delta_")]].mean()
    small_wins = int((pairs["delta_small_ap"] > 0).sum())
    control = aggregated.set_index("variant").loc["control"]
    weighted = aggregated.set_index("variant").loc["weighted"]
    parameter_change = 100 * (weighted.parameters - control.parameters) / control.parameters
    gflops_change = 100 * (weighted.gflops - control.gflops) / control.gflops
    criteria = {
        "small_mean": mean_delta["delta_small_ap"] >= 0.003,
        "small_wins": small_wins >= 2,
        "overall": mean_delta["delta_map50_95"] >= -0.001,
        "large": mean_delta["delta_large_ap"] >= -0.01,
        "cost": abs(parameter_change) <= 0.1 and abs(gflops_change) <= 0.1,
    }
    promote = all(criteria.values())
    decision = (
        "通过预注册条件。下一步构建加权MF-FPN+LSCD+Inner-WIoU完整模型，并进行150轮最终训练。"
        if promote
        else "未通过全部预注册条件。停止加权融合晋级，下一步转入NWD/IoU混合回归损失筛选。"
    )
    lines = [
        "# 自适应加权 MF-FPN：三随机种子配对复验报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 报告口径：三次独立50轮运行的均值±样本标准差；配对差值均按同一种子“加权−普通”计算。",
        "",
        "## 1. 三种子汇总",
        "",
        aggregate_table(aggregated),
        "",
        "![三种子均值](figures/multiseed_metric_means.png)",
        "",
        "## 2. 配对差值",
        "",
        paired_table(pairs),
        "",
        f"三个种子中有 **{small_wins}/3** 个种子的Small AP为正增益。均值和标准差用于稳定性描述；由于只有3个种子，本实验不作强统计显著性声明。",
        "",
        "![Small AP配对变化](figures/paired_small_ap.png)",
        "",
        "![平均训练轨迹](figures/multiseed_training_curve.png)",
        "",
        "## 3. 融合权重稳定性",
        "",
        "![融合权重稳定性](figures/fusion_weight_stability.png)",
        "",
        "每组权重被归一化为均值1。跨种子方向一致才可作为稳定行为解释；权重本身不替代检测指标的因果证据。明细见`metrics/fusion_weights_all_seeds.csv`。",
        "",
        "## 4. 预注册晋级判定",
        "",
        "| 条件 | 实际结果 | 是否通过 |",
        "|---|---:|---:|",
        f"| 平均Small AP增量≥+0.30 pp | {100 * mean_delta['delta_small_ap']:+.2f} pp | {'是' if criteria['small_mean'] else '否'} |",
        f"| Small AP正增益种子数≥2/3 | {small_wins}/3 | {'是' if criteria['small_wins'] else '否'} |",
        f"| 平均mAP50-95增量≥-0.10 pp | {100 * mean_delta['delta_map50_95']:+.2f} pp | {'是' if criteria['overall'] else '否'} |",
        f"| 平均Large AP增量≥-1.00 pp | {100 * mean_delta['delta_large_ap']:+.2f} pp | {'是' if criteria['large'] else '否'} |",
        f"| 参数量和GFLOPs变化≤0.1% | {parameter_change:+.4f}% / {gflops_change:+.4f}% | {'是' if criteria['cost'] else '否'} |",
        "",
        f"**结论：{decision}**",
        "",
        "## 5. 主张—证据映射",
        "",
        "| 主张 | 当前证据 | 状态 |",
        "|---|---|---|",
        f"| 加权融合稳定改善小目标 | 平均ΔSmall AP {100 * mean_delta['delta_small_ap']:+.2f} pp，正增益{small_wins}/3 | {'三种子筛选支持' if criteria['small_mean'] and criteria['small_wins'] else '不支持'} |",
        f"| 加权融合不损害总体和大目标 | 平均ΔmAP50-95 {100 * mean_delta['delta_map50_95']:+.2f} pp，ΔLarge AP {100 * mean_delta['delta_large_ap']:+.2f} pp | {'支持' if criteria['overall'] and criteria['large'] else '不支持'} |",
        "| 加权融合是最终完整模型贡献 | 尚未进行完整模型150轮实验 | 不能声称 |",
        "",
        "## 6. 可追溯产物",
        "",
        "- 预注册方案：`EXPERIMENT_PLAN.md`",
        "- 全部日志：`logs/`",
        "- 单次、聚合、配对及融合权重数据：`metrics/`",
        "- 四个新增检查点：`../../runs/mffpn*_seed[12]_50e/weights/best.pt`",
    ]
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "weighted_fusion_multiseed")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    figures = report_dir / "figures"
    metrics = report_dir / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    summary, frames = collect(report_dir)
    aggregated = aggregate(summary)
    pairs = paired_deltas(summary)
    weights = extract_fusion_weights(report_dir)
    summary.to_csv(metrics / "all_seed_results.csv", index=False, encoding="utf-8-sig")
    aggregated.to_csv(metrics / "aggregate_mean_std.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(metrics / "paired_deltas.csv", index=False, encoding="utf-8-sig")
    plot_metric_means(aggregated, figures / "multiseed_metric_means.png")
    plot_paired_small(pairs, figures / "paired_small_ap.png")
    plot_training_mean(frames, figures / "multiseed_training_curve.png")
    plot_weight_stability(weights, figures / "fusion_weight_stability.png")
    write_report(summary, aggregated, pairs, weights, report_dir)
    print(f"Multi-seed report written to {report_dir / 'README.md'}")


if __name__ == "__main__":
    main()
