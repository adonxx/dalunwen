from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo.small_crop import SmallObjectCenteredCrop, SmallObjectCropDataset  # noqa: E402


def _labels(boxes: np.ndarray, classes: np.ndarray) -> dict:
    return {
        "img": np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3),
        "cls": classes.reshape(-1, 1).astype(np.float32),
        "instances": Instances(
            boxes.astype(np.float32),
            segments=np.zeros((len(boxes), 0, 2), dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        ),
    }


def test_crop_preserves_shape_and_doubles_selected_small_box() -> None:
    random.seed(7)
    labels = _labels(np.array([[0.25, 0.50, 0.05, 0.10]]), np.array([3]))
    output = SmallObjectCenteredCrop(probability=1.0, crop_scale=0.5)(labels)
    assert output["img"].shape == (100, 200, 3)
    assert output["cls"].reshape(-1).tolist() == [3.0]
    box = output["instances"].bboxes[0]
    assert np.allclose(box[2:], [0.10, 0.20], atol=1e-6)


def test_crop_filters_mostly_invisible_boxes_but_keeps_selected_target() -> None:
    random.seed(11)
    boxes = np.array(
        [
            [0.20, 0.50, 0.05, 0.10],
            [0.90, 0.50, 0.18, 0.50],
        ]
    )
    output = SmallObjectCenteredCrop(probability=1.0, crop_scale=0.5)(_labels(boxes, np.array([1, 8])))
    assert output["cls"].reshape(-1).tolist() == [1.0]
    assert len(output["instances"]) == 1


def test_crop_is_noop_when_there_is_no_small_target() -> None:
    labels = _labels(np.array([[0.50, 0.50, 0.50, 0.80]]), np.array([4]))
    image_before = labels["img"].copy()
    boxes_before = labels["instances"].bboxes.copy()
    output = SmallObjectCenteredCrop(probability=1.0, crop_scale=0.5)(labels)
    assert np.array_equal(output["img"], image_before)
    assert np.allclose(output["instances"].bboxes, boxes_before)


def test_crop_dataset_is_a_yolo_dataset() -> None:
    assert issubclass(SmallObjectCropDataset, YOLODataset)
