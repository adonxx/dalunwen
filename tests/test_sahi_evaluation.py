from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_sahi import axis_origins, merge_detections, slice_windows  # noqa: E402


def test_axis_origins_cover_final_boundary() -> None:
    assert axis_origins(1920, 640, 0.20) == [0, 512, 1024, 1280]
    assert axis_origins(540, 640, 0.20) == [0]


def test_slice_windows_cover_image_corners() -> None:
    windows = slice_windows(1360, 765, 640, 0.20)
    assert windows[0] == (0, 0, 640, 640)
    assert windows[-1] == (720, 125, 1360, 765)
    assert len(windows) == 6


def test_class_aware_merge_keeps_different_classes() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [1.0, 1.0, 11.0, 11.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    classes = torch.tensor([0, 0, 1])
    kept_boxes, kept_scores, kept_classes = merge_detections(boxes, scores, classes, 0.5, 300)
    assert kept_boxes.shape == (2, 4)
    assert torch.allclose(kept_scores, torch.tensor([0.9, 0.7]))
    assert kept_classes.tolist() == [0, 1]
