from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from ultralytics.data.dataset import YOLODataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo.overlap_tiles import (  # noqa: E402
    OverlapTileDataset,
    axis_origins,
    crop_labels_to_window,
    overlap_windows,
)


def _raw_label(boxes: np.ndarray, classes: np.ndarray) -> dict:
    return {
        "shape": (540, 960),
        "cls": classes.reshape(-1, 1).astype(np.float32),
        "bboxes": boxes.astype(np.float32),
        "segments": [],
        "keypoints": None,
        "normalized": True,
        "bbox_format": "xywh",
    }


def test_axis_origins_cover_dimension_edges() -> None:
    assert axis_origins(960, 640, 0.20) == [0, 320]
    assert axis_origins(540, 640, 0.20) == [0]
    assert overlap_windows(960, 540, 640, 0.20) == [(0, 0, 640, 540), (320, 0, 960, 540)]


def test_crop_rewrites_boxes_in_tile_coordinates_and_filters_outside_box() -> None:
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    label = _raw_label(
        np.array(
            [
                [0.10, 0.50, 0.05, 0.10],
                [0.90, 0.50, 0.10, 0.20],
            ]
        ),
        np.array([1, 8]),
    )
    output = crop_labels_to_window(label, image, (0, 0, 640, 540))
    assert output["img"].shape == (540, 640, 3)
    assert output["cls"].reshape(-1).tolist() == [1.0]
    assert np.allclose(output["bboxes"][0], [0.15, 0.50, 0.075, 0.10], atol=1e-6)


def test_selected_window_contains_a_small_box() -> None:
    dataset = OverlapTileDataset.__new__(OverlapTileDataset)
    dataset.imgsz = 640
    random.seed(0)
    label = _raw_label(np.array([[0.10, 0.50, 0.04, 0.08]]), np.array([1]))
    window = dataset._select_window(label)
    assert window is not None
    x1, y1, x2, y2 = window
    assert 76.8 >= x1 and 115.2 <= x2 and 248.4 >= y1 and 291.6 <= y2


def test_overlap_tile_dataset_is_a_yolo_dataset() -> None:
    assert issubclass(OverlapTileDataset, YOLODataset)
