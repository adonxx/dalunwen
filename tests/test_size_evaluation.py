from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_object_sizes import area_bucket, build_coco_ground_truth  # noqa: E402


def test_area_bucket_uses_coco_boundaries() -> None:
    assert area_bucket(32**2 - 1) == "small"
    assert area_bucket(32**2) == "medium"
    assert area_bucket(96**2 - 1) == "medium"
    assert area_bucket(96**2) == "large"


def test_yolo_labels_convert_to_original_image_coco_boxes(tmp_path: Path) -> None:
    image_dir = tmp_path / "val" / "images"
    label_dir = tmp_path / "val" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(image_dir / "frame_a.jpg")
    (label_dir / "frame_a.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    output = tmp_path / "ground_truth.json"

    dataset_count, area_counts = build_coco_ground_truth(
        {
            "path": str(tmp_path),
            "val": "val/images",
            "names": {0: "target"},
        },
        "val",
        output,
    )

    coco = json.loads(output.read_text(encoding="utf-8"))
    assert dataset_count == {"images": 1, "annotations": 1}
    assert area_counts == {"all": 1, "small": 1, "medium": 0, "large": 0}
    assert coco["images"][0]["id"] == "frame_a"
    assert coco["annotations"][0]["category_id"] == 1
    assert coco["annotations"][0]["bbox"] == [90.0, 40.0, 20.0, 20.0]
    assert coco["annotations"][0]["area"] == 400.0
