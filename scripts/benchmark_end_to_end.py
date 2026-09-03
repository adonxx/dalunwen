"""Benchmark the six audited checkpoints with a fixed end-to-end protocol."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import build_model  # noqa: E402


VARIANTS = {
    "baseline": ("baseline", ["baseline_paper", "baseline_paper_seed1", "baseline_paper_seed2"]),
    "full": ("full", ["full_paper", "full_paper_seed1", "full_paper_seed2"]),
}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def image_source() -> tuple[np.ndarray, Path]:
    with (ROOT / "configs" / "data" / "visdrone2019.yaml").open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    image_dir = Path(data["path"]) / data["val"]
    paths = sorted(image_dir.glob("*"))
    if not paths:
        raise FileNotFoundError(f"No validation images found in {image_dir}")
    path = paths[0]
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read fixed benchmark image: {path}")
    return image, path


def sync() -> None:
    torch.cuda.synchronize(0)


def run_one(
    *,
    variant: str,
    stage: str,
    run_name: str,
    half: bool,
    image: np.ndarray,
    warmup: int,
    repeats: int,
) -> dict:
    weights = ROOT / "runs" / run_name / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")
    model = build_model(stage, weights=str(weights))
    kwargs = {
        "source": image,
        "imgsz": 640,
        "device": "0",
        "half": half,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "verbose": False,
    }
    for _ in range(warmup):
        model.predict(**kwargs)
    sync()
    torch.cuda.reset_peak_memory_stats(0)
    e2e, pre, infer, post = [], [], [], []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        results = model.predict(**kwargs)
        sync()
        e2e.append((time.perf_counter() - start) * 1000)
        speed = results[0].speed
        pre.append(float(speed["preprocess"]))
        infer.append(float(speed["inference"]))
        post.append(float(speed["postprocess"]))
    peak_mib = torch.cuda.max_memory_allocated(0) / 1024**2
    predictor = model.predictor
    # AutoBackend keeps the underlying PyTorch module in ``.model`` but does
    # not itself expose parameters as registered children in every version.
    dtype = None
    for candidate in (getattr(predictor.model, "model", None), predictor.model, model.model):
        if candidate is None:
            continue
        parameter = next(candidate.parameters(), None)
        if parameter is not None:
            dtype = str(parameter.dtype).replace("torch.", "")
            break
    if dtype is None:
        raise RuntimeError("Could not determine the effective inference model dtype")
    model_dtype = dtype
    record = {
        "variant": variant,
        "stage": stage,
        "run_name": run_name,
        "checkpoint": str(weights),
        "precision": "FP16" if half else "FP32",
        "effective_model_dtype": model_dtype,
        "warmup": warmup,
        "repeats": repeats,
        "latency_ms": {
            "e2e_mean": statistics.fmean(e2e),
            "e2e_p50": percentile(e2e, 50),
            "e2e_p95": percentile(e2e, 95),
            "preprocess_mean": statistics.fmean(pre),
            "inference_mean": statistics.fmean(infer),
            "postprocess_mean": statistics.fmean(post),
        },
        "fps_e2e": 1000 / statistics.fmean(e2e),
        "peak_allocated_mib": peak_mib,
        "model": {
            "parameters_m": sum(p.numel() for p in model.model.parameters()) / 1e6,
        },
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def aggregate(records: list[dict]) -> list[dict]:
    output = []
    for precision in ("FP32", "FP16"):
        for variant in VARIANTS:
            rows = [row for row in records if row["precision"] == precision and row["variant"] == variant]
            fields = {
                "e2e_mean_ms": [r["latency_ms"]["e2e_mean"] for r in rows],
                "e2e_p50_ms": [r["latency_ms"]["e2e_p50"] for r in rows],
                "e2e_p95_ms": [r["latency_ms"]["e2e_p95"] for r in rows],
                "preprocess_ms": [r["latency_ms"]["preprocess_mean"] for r in rows],
                "inference_ms": [r["latency_ms"]["inference_mean"] for r in rows],
                "postprocess_ms": [r["latency_ms"]["postprocess_mean"] for r in rows],
                "fps": [r["fps_e2e"] for r in rows],
                "peak_mib": [r["peak_allocated_mib"] for r in rows],
                "parameters_m": [r["model"]["parameters_m"] for r in rows],
            }
            row = {"precision": precision, "variant": variant, "seeds": len(rows)}
            for name, values in fields.items():
                row[f"{name}_mean"] = statistics.fmean(values)
                row[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output.append(row)
    return output


def generate_report(records: list[dict], image_path: Path, warmup: int, repeats: int) -> None:
    report = ROOT / "reports" / "deployment_benchmark"
    metrics_dir, figures_dir = report / "metrics", report / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(records)
    (metrics_dir / "raw_runs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams["axes.unicode_minus"] = False
    for metric, ylabel, filename in [
        ("e2e_mean_ms", "End-to-end latency (ms)", "e2e_latency.png"),
        ("fps", "End-to-end FPS", "e2e_fps.png"),
    ]:
        figure, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
        x = np.arange(2)
        width = 0.34
        for offset, precision in [(-width / 2, "FP32"), (width / 2, "FP16")]:
            rows = [r for r in summary if r["precision"] == precision]
            values = [next(r for r in rows if r["variant"] == variant)[f"{metric}_mean"] for variant in VARIANTS]
            errors = [next(r for r in rows if r["variant"] == variant)[f"{metric}_std"] for variant in VARIANTS]
            axis.bar(x + offset, values, width, yerr=errors, capsize=4, label=precision)
        axis.set_xticks(x, ["YOLOv11s baseline", "Drone-YOLO"])
        axis.set_ylabel(ylabel)
        axis.set_title("Three-seed deployment benchmark")
        axis.legend()
        figure.savefig(figures_dir / filename, dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for axis, precision in zip(axes, ("FP32", "FP16")):
        rows = [r for r in summary if r["precision"] == precision]
        for variant, colour in [("baseline", "#4c78a8"), ("full", "#f58518")]:
            row = next(r for r in rows if r["variant"] == variant)
            values = [row[f"{key}_mean"] for key in ("preprocess_ms", "inference_ms", "postprocess_ms")]
            axis.bar([variant], [values[0]], color="#8da0cb", label="Preprocess" if variant == "baseline" else None)
            axis.bar([variant], [values[1]], bottom=[values[0]], color=colour, label="Inference" if variant == "baseline" else None)
            axis.bar([variant], [values[2]], bottom=[values[0] + values[1]], color="#999999", label="Postprocess" if variant == "baseline" else None)
        axis.set_title(precision)
        axis.set_ylabel("Framework timing (ms/image)")
        axis.legend()
    figure.savefig(figures_dir / "latency_components.png", dpi=180)
    plt.close(figure)

    def row(precision: str, variant: str) -> dict:
        return next(item for item in summary if item["precision"] == precision and item["variant"] == variant)

    report_rows = []
    for precision in ("FP32", "FP16"):
        for variant in ("baseline", "full"):
            item = row(precision, variant)
            report_rows.append(
                f"| {precision} | {'YOLOv11s 基线' if variant == 'baseline' else 'Drone-YOLO'} | "
                f"{item['e2e_mean_ms_mean']:.2f} ± {item['e2e_mean_ms_std']:.2f} | "
                f"{item['e2e_p95_ms_mean']:.2f} ± {item['e2e_p95_ms_std']:.2f} | "
                f"{item['fps_mean']:.2f} ± {item['fps_std']:.2f} | "
                f"{item['peak_mib_mean']:.0f} ± {item['peak_mib_std']:.0f} | "
                f"{item['parameters_m_mean']:.3f} |"
            )
    fp32_base, fp32_full = row("FP32", "baseline"), row("FP32", "full")
    fp16_base, fp16_full = row("FP16", "baseline"), row("FP16", "full")
    fp32_change = (fp32_full["e2e_mean_ms_mean"] / fp32_base["e2e_mean_ms_mean"] - 1) * 100
    fp16_change = (fp16_full["e2e_mean_ms_mean"] / fp16_base["e2e_mean_ms_mean"] - 1) * 100
    markdown = f"""# 三种子端到端部署基准

> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  
> 设备：{torch.cuda.get_device_name(0)}；固定输入：`{image_path}`；batch=1，640×640，预热{warmup}次、测量{repeats}次。

## 1. 端到端结果

| 精度 | 模型 | 平均延迟（ms） | p95 延迟（ms） | FPS | 峰值分配显存（MiB） | 参数量（M） |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(report_rows)}

![端到端延迟](figures/e2e_latency.png)

![端到端吞吐](figures/e2e_fps.png)

## 2. 时间组成

![延迟组成](figures/latency_components.png)

Drone-YOLO 相对基线的平均端到端延迟变化：FP32 **{fp32_change:+.1f}%**，FP16 **{fp16_change:+.1f}%**。该数字只描述本机固定协议下的实测结果；应结合上图中的预处理、前向与后处理时间解释，不应用参数量或 GFLOPs 直接替代。

## 3. 可追溯性与边界

- 每个模型均使用三个已完成训练种子；原始逐检查点结果在 `metrics/raw_runs.json`，聚合统计在 `metrics/summary.json`。
- 测试包括图像预处理、模型前向和 NMS/结果后处理，不包括磁盘 I/O、模型加载及首次初始化。
- 此处是单卡、batch=1 的框架级基准；TensorRT、ONNX 或不同 GPU 的绝对延迟需另行测量。
"""
    (report / "README.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup and repeats must both be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    image, path = image_source()
    records = []
    for half in (False, True):
        for variant, (stage, run_names) in VARIANTS.items():
            for run_name in run_names:
                print(f"Benchmarking {variant} {run_name} {'FP16' if half else 'FP32'}")
                records.append(
                    run_one(
                        variant=variant,
                        stage=stage,
                        run_name=run_name,
                        half=half,
                        image=image,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                )
    generate_report(records, path, args.warmup, args.repeats)
    print("Deployment benchmark completed")


if __name__ == "__main__":
    main()
