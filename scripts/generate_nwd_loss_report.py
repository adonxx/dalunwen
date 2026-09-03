"""Generate the focused 50-epoch NWD regression-loss screening report."""

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
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo.loss import InnerWIoULoss, NormalizedWassersteinLoss  # noqa: E402


VARIANTS = OrderedDict(
    [
        ("inner_wiou", {"label": "Inner-WIoU", "run": "loss_inner_wiou_50e"}),
        ("nwd", {"label": "NWD", "run": "loss_nwd_50e"}),
        ("hybrid", {"label": "50% Inner-WIoU + 50% NWD", "run": "loss_hybrid_50e"}),
    ]
)
COLORS = ["#5B6C8F", "#27AE60", "#EB5757"]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def collect(report_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    frames = {}
    for variant, meta in VARIANTS.items():
        overall_record = load_json(report_dir / "metrics" / f"{variant}_val.json")
        size_record = load_json(report_dir / "metrics" / f"{variant}_size_val.json")
        frame = pd.read_csv(ROOT / "runs" / meta["run"] / "results.csv")
        frame.columns = [column.strip() for column in frame.columns]
        frames[variant] = frame.iloc[:50].copy()
        best = frame.iloc[:50].loc[frame.iloc[:50]["metrics/mAP50-95(B)"].idxmax()]
        overall = overall_record["overall"]
        area = size_record["overall_by_area"]
        rows.append(
            {
                "variant": variant,
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


def deltas_from_control(summary: pd.DataFrame) -> pd.DataFrame:
    control = summary.set_index("variant").loc["inner_wiou"]
    rows = []
    for variant in ("nwd", "hybrid"):
        row = summary.set_index("variant").loc[variant]
        rows.append(
            {
                "variant": variant,
                "label": row.label,
                "delta_map50": row.map50 - control.map50,
                "delta_map50_95": row.map50_95 - control.map50_95,
                "delta_small_ap": row.small_ap - control.small_ap,
                "delta_small_ap50": row.small_ap50 - control.small_ap50,
                "delta_medium_ap": row.medium_ap - control.medium_ap,
                "delta_large_ap": row.large_ap - control.large_ap,
                "delta_parameters": row.parameters - control.parameters,
                "delta_gflops": row.gflops - control.gflops,
            }
        )
    return pd.DataFrame(rows)


def plot_training(frames: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True)
    for (variant, frame), color in zip(frames.items(), COLORS):
        axes[0].plot(frame["epoch"], 100 * frame["metrics/mAP50(B)"], color=color, label=VARIANTS[variant]["label"])
        axes[1].plot(
            frame["epoch"],
            100 * frame["metrics/mAP50-95(B)"],
            color=color,
            label=VARIANTS[variant]["label"],
        )
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
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.3))
    for index, (row, color) in enumerate(zip(summary.itertuples(index=False), COLORS)):
        values = [100 * getattr(row, metric) for metric in metrics]
        bars = ax.bar(x + (index - 1) * width, values, width, label=row.label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylabel("COCO metric (%)")
    ax.set_title("Regression-loss screening on the same detector (50 epochs)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_tiny_box_response(output: Path) -> None:
    shifts = torch.linspace(0, 8, 81)
    target = torch.tensor([[32.0, 32.0, 36.0, 36.0]]).repeat(len(shifts), 1)
    pred = target.clone()
    pred[:, [0, 2]] += shifts[:, None]
    inner = InnerWIoULoss()
    inner.eval()
    nwd = NormalizedWassersteinLoss(constant=12.8)
    with torch.no_grad():
        inner_values = inner(pred, target).cpu().numpy()
        nwd_values = nwd(pred, target).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(shifts.numpy(), inner_values, color=COLORS[0], label="Inner-WIoU")
    ax.plot(shifts.numpy(), nwd_values, color=COLORS[1], label="NWD (C=12.8)")
    ax.set_xlabel("Horizontal displacement of a 4x4-pixel box (pixels)")
    ax.set_ylabel("Regression loss")
    ax.set_title("Loss response to tiny-box displacement")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def result_table(summary: pd.DataFrame) -> str:
    lines = [
        "| 回归损失 | mAP50 | mAP50-95 | Small AP | Small AP50 | Medium AP | Large AP | 参数量(M) | GFLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.label} | {fmt_pct(row.map50)} | {fmt_pct(row.map50_95)} | {fmt_pct(row.small_ap)} | "
            f"{fmt_pct(row.small_ap50)} | {fmt_pct(row.medium_ap)} | {fmt_pct(row.large_ap)} | "
            f"{row.parameters / 1e6:.3f} | {row.gflops:.2f} |"
        )
    return "\n".join(lines)


def delta_table(deltas: pd.DataFrame) -> str:
    lines = [
        "| 候选损失 | ΔmAP50-95 | ΔSmall AP | ΔSmall AP50 | ΔMedium AP | ΔLarge AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in deltas.itertuples(index=False):
        lines.append(
            f"| {row.label} | {100 * row.delta_map50_95:+.2f} | {100 * row.delta_small_ap:+.2f} | "
            f"{100 * row.delta_small_ap50:+.2f} | {100 * row.delta_medium_ap:+.2f} | "
            f"{100 * row.delta_large_ap:+.2f} |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, deltas: pd.DataFrame, report_dir: Path) -> None:
    indexed = deltas.set_index("variant")
    criteria = {}
    for variant in ("nwd", "hybrid"):
        row = indexed.loc[variant]
        criteria[variant] = {
            "small": row.delta_small_ap >= 0.003,
            "overall": row.delta_map50_95 >= -0.001,
            "large": row.delta_large_ap >= -0.01,
            "cost": row.delta_parameters <= 0 and row.delta_gflops <= 1e-6,
        }
    passed = [variant for variant in ("nwd", "hybrid") if all(criteria[variant].values())]
    if passed:
        passed.sort(
            key=lambda variant: (indexed.loc[variant].delta_small_ap, indexed.loc[variant].delta_map50_95),
            reverse=True,
        )
        selected = passed[0]
        decision = f"{VARIANTS[selected]['label']}通过预注册条件，建议进入多随机种子复验。"
    else:
        selected = None
        decision = "NWD和混合损失均未通过预注册条件，不建议进入150轮；应停止该损失方向或重新审视NWD用于标签分配的方案。"
    lines = [
        "# NWD回归损失：50轮筛选报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        "> 三组使用相同MF-FPN+LSCD结构；仅框回归项不同。TAL标签分配、DFL和训练超参数保持一致。",
        "",
        "## 1. 统一结果",
        "",
        result_table(summary),
        "",
        "![分尺寸指标](figures/loss_ablation_metrics.png)",
        "",
        "![训练曲线](figures/loss_ablation_training.png)",
        "",
        "## 2. 相对Inner-WIoU的变化",
        "",
        delta_table(deltas),
        "",
        "## 3. 微小框位移响应",
        "",
        "![损失响应](figures/tiny_box_loss_response.png)",
        "",
        "该曲线只说明不同损失对4×4像素框位移的数值响应，不构成检测性能证据。",
        "",
        "## 4. 预注册判定",
        "",
        "| 候选 | Small AP≥+0.30 pp | mAP50-95≥-0.10 pp | Large AP≥-1.00 pp | 无推理开销增加 |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {VARIANTS[variant]['label']} | {'是' if criteria[variant]['small'] else '否'} | "
            f"{'是' if criteria[variant]['overall'] else '否'} | {'是' if criteria[variant]['large'] else '否'} | "
            f"{'是' if criteria[variant]['cost'] else '否'} |"
            for variant in ("nwd", "hybrid")
        ],
        "",
        f"**结论：{decision}**",
        "",
        "## 5. 主张—证据映射",
        "",
        "| 主张 | 当前证据 | 状态 |",
        "|---|---|---|",
        f"| NWD回归改善小目标 | ΔSmall AP {100 * indexed.loc['nwd'].delta_small_ap:+.2f} pp | {'50轮支持' if all(criteria['nwd'].values()) else '不支持'} |",
        f"| 混合回归改善小目标 | ΔSmall AP {100 * indexed.loc['hybrid'].delta_small_ap:+.2f} pp | {'50轮支持' if all(criteria['hybrid'].values()) else '不支持'} |",
        "| 候选损失为最终贡献 | 尚无150轮与多随机种子证据 | 不能声称 |",
        "",
        "## 6. 可追溯产物",
        "",
        "- 预注册方案：`EXPERIMENT_PLAN.md`",
        "- 训练和评估日志：`logs/`",
        "- JSON及CSV指标：`metrics/`",
        "- 三组检查点：`../../runs/loss_*_50e/weights/best.pt`",
    ]
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "nwd_loss_ablation")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    figures = report_dir / "figures"
    metrics = report_dir / "metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    summary, frames = collect(report_dir)
    deltas = deltas_from_control(summary)
    summary.to_csv(metrics / "nwd_loss_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(metrics / "nwd_loss_deltas.csv", index=False, encoding="utf-8-sig")
    plot_training(frames, figures / "loss_ablation_training.png")
    plot_metrics(summary, figures / "loss_ablation_metrics.png")
    plot_tiny_box_response(figures / "tiny_box_loss_response.png")
    write_report(summary, deltas, report_dir)
    print(f"NWD loss report written to {report_dir / 'README.md'}")


if __name__ == "__main__":
    main()
