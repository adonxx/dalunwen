from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fuse_sahi_predictions import merge_image_predictions, select_sahi  # noqa: E402


def prediction(x: float, width: float, score: float, category: int = 1) -> dict:
    return {
        "image_id": 1,
        "category_id": category,
        "bbox": [x, 0.0, width, width],
        "score": score,
    }


def test_scale_selection_uses_strict_area_boundary() -> None:
    records = [prediction(0, 31.0, 0.8), prediction(40, 32.0, 0.7)]
    selected = select_sahi(records, 32.0)
    assert selected == [records[0]]
    assert select_sahi(records, None) == records


def test_fusion_suppresses_same_class_duplicate() -> None:
    whole = [prediction(0, 10, 0.8)]
    sahi = [prediction(1, 10, 0.9)]
    merged, stats = merge_image_predictions(whole, sahi, 0.5, 300)
    assert merged == [sahi[0]]
    assert stats["kept_whole"] == 0
    assert stats["kept_sahi"] == 1


def test_fusion_preserves_different_classes() -> None:
    whole = [prediction(0, 10, 0.8, category=1)]
    sahi = [prediction(1, 10, 0.9, category=2)]
    merged, stats = merge_image_predictions(whole, sahi, 0.5, 300)
    assert len(merged) == 2
    assert stats["kept_whole"] == 1
    assert stats["kept_sahi"] == 1
