"""Generate the thesis-level experiment tables and summary figures.

The values in this file are copied from the committed experiment reports under
``reports/``.  Keeping the plotting input explicit makes the final figures easy
to audit without depending on local training runs or untracked checkpoints.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "thesis_experiment_chapter"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"


CORE_ROWS = [
    {
        "method": "YOLOv11s",
        "seeds": 3,
        "epochs": 150,
        "map50_mean": 39.64,
        "map50_sd": 0.15,
        "map5095_mean": 23.24,
        "map5095_sd": 0.08,
        "small_ap_mean": 13.78,
        "small_ap_sd": 0.24,
        "medium_ap_mean": 35.09,
        "medium_ap_sd": 0.17,
        "large_ap_mean": 47.06,
        "large_ap_sd": 1.75,
        "params_m": 9.432,
        "gflops": 21.56,
    },
    {
        "method": "Drone-YOLO (Full)",
        "seeds": 3,
        "epochs": 150,
        "map50_mean": 40.99,
        "map50_sd": 0.35,
        "map5095_mean": 24.03,
        "map5095_sd": 0.27,
        "small_ap_mean": 15.46,
        "small_ap_sd": 0.22,
        "medium_ap_mean": 34.85,
        "medium_ap_sd": 0.39,
        "large_ap_mean": 44.28,
        "large_ap_sd": 1.62,
        "params_m": 2.772,
        "gflops": 19.19,
    },
]


ABLATION_ROWS = [
    {
        "stage": "YOLOv11s",
        "best_epoch": 88,
        "precision": 53.22,
        "recall": 39.77,
        "map50": 39.64,
        "map5095": 23.23,
        "small_ap": 13.52,
        "medium_ap": 35.26,
        "large_ap": 48.17,
        "params_m": 9.432,
        "gflops": 21.56,
    },
    {
        "stage": "MF-FPN",
        "best_epoch": 139,
        "precision": 54.88,
        "recall": 40.98,
        "map50": 41.08,
        "map5095": 24.27,
        "small_ap": 15.63,
        "medium_ap": 35.28,
        "large_ap": 40.47,
        "params_m": 3.169,
        "gflops": 24.52,
    },
    {
        "stage": "+LSCD",
        "best_epoch": 142,
        "precision": 53.20,
        "recall": 41.44,
        "map50": 40.24,
        "map5095": 23.66,
        "small_ap": 15.27,
        "medium_ap": 34.21,
        "large_ap": 45.72,
        "params_m": 2.772,
        "gflops": 19.19,
    },
    {
        "stage": "+Inner-WIoU (Full)",
        "best_epoch": 140,
        "precision": 52.98,
        "recall": 42.30,
        "map50": 41.40,
        "map5095": 24.34,
        "small_ap": 15.62,
        "medium_ap": 35.24,
        "large_ap": 46.04,
        "params_m": 2.772,
        "gflops": 19.19,
    },
]


SCREENING_ROWS = [
    {
        "candidate": "Adaptive weighted fusion",
        "budget": "50e, 3 seeds",
        "delta_map50": -0.21,
        "delta_map5095": -0.15,
        "delta_small_ap": -0.15,
        "delta_medium_ap": -0.14,
        "delta_large_ap": -0.04,
        "decision": "reject",
    },
    {
        "candidate": "NWD loss",
        "budget": "50e, seed 0",
        "delta_map50": -0.80,
        "delta_map5095": -0.47,
        "delta_small_ap": -0.99,
        "delta_medium_ap": -0.06,
        "delta_large_ap": 6.49,
        "decision": "reject",
    },
    {
        "candidate": "Inner-WIoU + NWD",
        "budget": "50e, seed 0",
        "delta_map50": -0.30,
        "delta_map5095": -0.21,
        "delta_small_ap": -0.31,
        "delta_medium_ap": -0.07,
        "delta_large_ap": 4.08,
        "decision": "reject",
    },
    {
        "candidate": "Small-aware TAL",
        "budget": "50e, seed 0",
        "delta_map50": -0.71,
        "delta_map5095": -0.84,
        "delta_small_ap": -0.62,
        "delta_medium_ap": -0.96,
        "delta_large_ap": 4.65,
        "decision": "reject",
    },
    {
        "candidate": "Small-object crop training",
        "budget": "50e, seed 0",
        "delta_map50": -1.91,
        "delta_map5095": -1.06,
        "delta_small_ap": -1.21,
        "delta_medium_ap": -0.78,
        "delta_large_ap": 5.84,
        "decision": "reject",
    },
    {
        "candidate": "Overlap-tile training",
        "budget": "50e, seed 0",
        "delta_map50": -1.32,
        "delta_map5095": -0.69,
        "delta_small_ap": -0.99,
        "delta_medium_ap": -0.13,
        "delta_large_ap": 8.57,
        "decision": "reject",
    },
    {
        "candidate": "Selective KD",
        "budget": "50e, seed 0",
        "delta_map50": -0.84,
        "delta_map5095": -0.60,
        "delta_small_ap": -0.60,
        "delta_medium_ap": -0.40,
        "delta_large_ap": 2.79,
        "decision": "reject",
    },
    {
        "candidate": "P5-lite feedback",
        "budget": "150e, seed 0",
        "delta_map50": -0.87,
        "delta_map5095": -0.63,
        "delta_small_ap": -0.42,
        "delta_medium_ap": -1.07,
        "delta_large_ap": -0.18,
        "decision": "reject",
    },
    {
        "candidate": "P5-lite P4-only",
        "budget": "50e, seed 0",
        "delta_map50": -0.69,
        "delta_map5095": -0.44,
        "delta_small_ap": -0.57,
        "delta_medium_ap": -0.25,
        "delta_large_ap": 1.77,
        "decision": "reject",
    },
]


INFERENCE_ROWS = [
    {
        "method": "Whole-640",
        "map5095": 23.51,
        "small_ap": 14.75,
        "small_ar": 24.46,
        "medium_ap": 34.10,
        "large_ap": 45.29,
        "latency_ms": 26.35,
        "fps": 37.95,
    },
    {
        "method": "Full Scale-48",
        "map5095": 26.82,
        "small_ap": 19.55,
        "small_ar": 33.35,
        "medium_ap": 34.81,
        "large_ap": 44.09,
        "latency_ms": 35.13,
        "fps": 28.47,
    },
    {
        "method": "Top-1",
        "map5095": 26.55,
        "small_ap": 19.31,
        "small_ar": 32.57,
        "medium_ap": 34.55,
        "large_ap": 44.09,
        "latency_ms": 31.72,
        "fps": 31.53,
    },
    {
        "method": "Gate-Q20",
        "map5095": 26.53,
        "small_ap": 19.26,
        "small_ar": 32.45,
        "medium_ap": 34.55,
        "large_ap": 44.09,
        "latency_ms": 25.63,
        "fps": 39.02,
    },
    {
        "method": "Gate-Q30",
        "map5095": 26.49,
        "small_ap": 19.19,
        "small_ar": 32.32,
        "medium_ap": 34.52,
        "large_ap": 44.09,
        "latency_ms": 24.36,
        "fps": 41.05,
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_core_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    colors = ["#8B95A5", "#2274A5"]
    methods = [row["method"] for row in CORE_ROWS]

    metric_specs = [
        ("map50_mean", "map50_sd", "mAP50"),
        ("map5095_mean", "map5095_sd", "mAP50-95"),
        ("small_ap_mean", "small_ap_sd", "Small AP"),
    ]
    x = range(len(metric_specs))
    width = 0.34
    for method_index, row in enumerate(CORE_ROWS):
        values = [row[item[0]] for item in metric_specs]
        errors = [row[item[1]] for item in metric_specs]
        positions = [value + (method_index - 0.5) * width for value in x]
        axes[0].bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=methods[method_index],
            color=colors[method_index],
        )
    axes[0].set_xticks(list(x), [item[2] for item in metric_specs])
    axes[0].set_ylabel("AP (%)")
    axes[0].set_title("Three-seed accuracy (mean +/- SD)")
    axes[0].legend(frameon=False, fontsize=8)
    style_axes(axes[0])

    paired_delta = [1.36, 0.78, 1.68, -0.25, -2.78]
    paired_sd = [0.38, 0.29, 0.37, 0.34, 2.10]
    labels = ["mAP50", "mAP50-95", "Small", "Medium", "Large"]
    delta_colors = ["#2A9D8F" if value >= 0 else "#E76F51" for value in paired_delta]
    axes[1].bar(labels, paired_delta, yerr=paired_sd, capsize=3, color=delta_colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Paired delta (pp)")
    axes[1].set_title("Full minus baseline")
    axes[1].tick_params(axis="x", rotation=20)
    style_axes(axes[1])

    reductions = [70.6, 11.0]
    axes[2].bar(["Parameters", "GFLOPs"], reductions, color=["#6A4C93", "#F4A261"])
    axes[2].set_ylabel("Reduction (%)")
    axes[2].set_ylim(0, 80)
    axes[2].set_title("Model complexity reduction")
    for index, value in enumerate(reductions):
        axes[2].text(index, value + 1.5, f"-{value:.1f}%", ha="center")
    style_axes(axes[2])

    fig.savefig(FIGURES / "core_accuracy_efficiency.png", dpi=220)
    plt.close(fig)


def plot_screening_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(8.7, 5.8), constrained_layout=True)
    x = [row["delta_small_ap"] for row in SCREENING_ROWS]
    y = [row["delta_large_ap"] for row in SCREENING_ROWS]
    c = [row["delta_map5095"] for row in SCREENING_ROWS]
    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="RdYlGn",
        vmin=-1.1,
        vmax=0.2,
        s=85,
        edgecolor="white",
        linewidth=0.8,
    )
    short_names = ["Weighted", "NWD", "Hybrid", "TAL", "Crop", "Tile", "KD", "P5-150", "P4-50"]
    for name, x_value, y_value in zip(short_names, x, y):
        ax.annotate(name, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Delta Small AP (pp)")
    ax.set_ylabel("Delta Large AP (pp)")
    ax.set_title("Screening experiments: scale trade-off")
    style_axes(ax)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Delta mAP50-95 (pp)")
    fig.savefig(FIGURES / "screening_scale_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_inference_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.5), constrained_layout=True)
    latency = [row["latency_ms"] for row in INFERENCE_ROWS]
    accuracy = [row["map5095"] for row in INFERENCE_ROWS]
    small_ap = [row["small_ap"] for row in INFERENCE_ROWS]
    scatter = ax.scatter(
        latency,
        accuracy,
        c=small_ap,
        cmap="viridis",
        s=110,
        edgecolor="white",
        linewidth=0.8,
    )
    for row in INFERENCE_ROWS:
        ax.annotate(
            row["method"],
            (row["latency_ms"], row["map5095"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("End-to-end latency (ms/image)")
    ax.set_ylabel("mAP50-95 (%)")
    ax.set_title("Accuracy-latency trade-off of inference strategies")
    style_axes(ax)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Small AP (%)")
    fig.savefig(FIGURES / "inference_accuracy_latency.png", dpi=220)
    plt.close(fig)


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS / "core_three_seed.csv", CORE_ROWS)
    write_csv(METRICS / "seed0_ablation.csv", ABLATION_ROWS)
    write_csv(METRICS / "screening_results.csv", SCREENING_ROWS)
    write_csv(METRICS / "inference_tradeoff.csv", INFERENCE_ROWS)
    plot_core_summary()
    plot_screening_tradeoff()
    plot_inference_tradeoff()
    print(f"Generated thesis summary artifacts in: {OUT}")


if __name__ == "__main__":
    main()
