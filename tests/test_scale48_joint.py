from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_scale48_joint import filter_scale_tensors, timing_summary  # noqa: E402


def test_scale_filter_uses_original_box_area() -> None:
    boxes = torch.tensor([[0.0, 0.0, 47.0, 48.0], [0.0, 0.0, 48.0, 48.0]])
    scores = torch.tensor([0.9, 0.8])
    classes = torch.tensor([1, 2])
    kept_boxes, kept_scores, kept_classes = filter_scale_tensors(boxes, scores, classes, 48.0)
    assert kept_boxes.shape == (1, 4)
    assert kept_scores.tolist() == [scores[0].item()]
    assert kept_classes.tolist() == [1]


def test_timing_summary_reports_mean_and_fps() -> None:
    first = {
        "decode_ms": 1.0,
        "whole_ms": 2.0,
        "slice_window_ms": 0.1,
        "sahi_ms": 4.0,
        "filter_and_fusion_ms": 0.5,
        "serialization_ms": 0.1,
        "end_to_end_ms": 10.0,
    }
    second = {**first, "end_to_end_ms": 20.0}
    summary = timing_summary([first, second])
    assert summary["mean_end_to_end_ms"] == 15.0
    assert summary["fps_from_mean_end_to_end"] == 1000.0 / 15.0
