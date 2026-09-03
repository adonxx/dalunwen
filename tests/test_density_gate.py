from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_density_gate import density_scores, evenly_spaced, select_top_window  # noqa: E402


def test_density_score_uses_small_confident_box_centers() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [90.0, 0.0, 140.0, 50.0],
            [110.0, 0.0, 120.0, 10.0],
        ]
    )
    scores = torch.tensor([0.6, 0.9, 0.04])
    windows = [(0, 0, 100, 100), (100, 0, 200, 100)]
    values = density_scores(boxes, scores, windows, area_threshold=48, confidence_threshold=0.05)
    assert values == [0.6000000238418579, 0.0]


def test_gate_returns_one_deterministic_tile() -> None:
    windows = [(0, 0, 10, 10), (10, 0, 20, 10)]
    selected, score, index = select_top_window(windows, [0.5, 0.8], threshold=0.7)
    assert selected == [windows[1]]
    assert score == 0.8
    assert index == 1
    assert select_top_window(windows, [0.5, 0.8], threshold=0.9)[0] == []


def test_evenly_spaced_subset_preserves_endpoints() -> None:
    paths = [Path(str(index)) for index in range(10)]
    selected = evenly_spaced(paths, 4)
    assert selected[0] == paths[0]
    assert selected[-1] == paths[-1]
    assert len(selected) == 4
