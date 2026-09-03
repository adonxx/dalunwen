"""Fixed-grid overlapping tile sampling for small-object training."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.patches import imread
from ultralytics.utils.torch_utils import unwrap_model


def axis_origins(length: int, window: int, overlap: float) -> list[int]:
    """Return deterministic window origins covering one image dimension."""
    if length <= 0 or window <= 0:
        raise ValueError("length and window must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must lie in [0, 1)")
    if length <= window:
        return [0]
    step = max(1, int(round(window * (1.0 - overlap))))
    final_origin = length - window
    origins = list(range(0, final_origin + 1, step))
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def overlap_windows(width: int, height: int, window: int = 640, overlap: float = 0.20) -> list[tuple[int, int, int, int]]:
    """Return fixed-grid xyxy windows covering an image without padding."""
    return [
        (x, y, min(x + window, width), min(y + window, height))
        for y in axis_origins(height, window, overlap)
        for x in axis_origins(width, window, overlap)
    ]


def xywhn_to_xyxy(bboxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized xywh boxes to absolute xyxy boxes."""
    xyxy = np.empty_like(bboxes, dtype=np.float32)
    xyxy[:, 0] = (bboxes[:, 0] - bboxes[:, 2] / 2) * width
    xyxy[:, 1] = (bboxes[:, 1] - bboxes[:, 3] / 2) * height
    xyxy[:, 2] = (bboxes[:, 0] + bboxes[:, 2] / 2) * width
    xyxy[:, 3] = (bboxes[:, 1] + bboxes[:, 3] / 2) * height
    return xyxy


def xyxy_to_xywhn(bboxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert absolute xyxy boxes to normalized xywh boxes."""
    xywh = np.empty_like(bboxes, dtype=np.float32)
    xywh[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / (2 * width)
    xywh[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / (2 * height)
    xywh[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / width
    xywh[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / height
    return xywh


def crop_labels_to_window(
    label: dict,
    image: np.ndarray,
    window: tuple[int, int, int, int],
    min_visibility: float = 0.5,
    min_box_size: float = 2.0,
) -> dict:
    """Crop an original-image label dictionary to one xyxy tile window."""
    if label["bbox_format"] != "xywh" or not label["normalized"]:
        raise ValueError("overlap tile training expects normalized xywh detection labels")
    x1, y1, x2, y2 = window
    tile_width, tile_height = x2 - x1, y2 - y1
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("window must have positive width and height")
    height, width = image.shape[:2]
    boxes_before = xywhn_to_xyxy(label["bboxes"], width, height)
    original_wh = np.maximum(boxes_before[:, 2:] - boxes_before[:, :2], 0.0)
    original_area = original_wh.prod(axis=1)

    boxes_after = boxes_before.copy()
    boxes_after[:, [0, 2]] = np.clip(boxes_after[:, [0, 2]], x1, x2) - x1
    boxes_after[:, [1, 3]] = np.clip(boxes_after[:, [1, 3]], y1, y2) - y1
    clipped_wh = np.maximum(boxes_after[:, 2:] - boxes_after[:, :2], 0.0)
    visibility = clipped_wh.prod(axis=1) / np.maximum(original_area, 1e-9)
    keep = (
        (visibility >= min_visibility)
        & (clipped_wh[:, 0] >= min_box_size)
        & (clipped_wh[:, 1] >= min_box_size)
    )

    output = deepcopy(label)
    output.pop("shape", None)
    output["img"] = image[y1:y2, x1:x2].copy()
    output["ori_shape"] = (tile_height, tile_width)
    output["resized_shape"] = (tile_height, tile_width)
    output["ratio_pad"] = (1.0, 1.0)
    output["bboxes"] = xyxy_to_xywhn(boxes_after[keep], tile_width, tile_height)
    output["cls"] = output["cls"][keep]
    if output.get("segments"):
        output["segments"] = [segment for segment, retained in zip(output["segments"], keep) if retained]
    return output


class OverlapTileDataset(YOLODataset):
    """Train on a fixed-size dataset, replacing selected reads with small-object tiles."""

    probability = 0.5
    window_size = 640
    overlap = 0.20
    small_area_threshold = 32.0**2
    min_visibility = 0.5
    min_box_size = 2.0

    def _select_window(self, label: dict) -> tuple[int, int, int, int] | None:
        height, width = label["shape"]
        windows = overlap_windows(width, height, self.window_size, self.overlap)
        if len(windows) == 1 and windows[0] == (0, 0, width, height):
            return None
        boxes = xywhn_to_xyxy(label["bboxes"], width, height)
        box_wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0)
        input_scale = self.imgsz / max(height, width)
        is_small = box_wh.prod(axis=1) * input_scale**2 < self.small_area_threshold
        eligible: list[tuple[int, list[tuple[int, int, int, int]]]] = []
        for index in np.flatnonzero(is_small).tolist():
            box = boxes[index]
            containing = [
                window
                for window in windows
                if box[0] >= window[0] and box[1] >= window[1] and box[2] <= window[2] and box[3] <= window[3]
            ]
            if containing:
                eligible.append((index, containing))
        if not eligible:
            return None
        _, containing = random.choice(eligible)
        return random.choice(containing)

    def get_image_and_label(self, index: int) -> dict:
        if not self.augment or random.random() >= self.probability:
            return super().get_image_and_label(index)
        label = self.labels[index]
        window = self._select_window(label)
        if window is None:
            return super().get_image_and_label(index)
        # Keep Ultralytics' augmentation buffer populated. Mosaic samples its
        # secondary images from this buffer, while the tile below is read at
        # original resolution to preserve the intended object scale.
        self.load_image(index)
        image = imread(str(Path(self.im_files[index])), flags=self.cv2_flag)
        if image is None:
            raise FileNotFoundError(f"Image Not Found {self.im_files[index]}")
        return self.update_labels_info(
            crop_labels_to_window(label, image, window, self.min_visibility, self.min_box_size)
        )


class OverlapTileTrainer(DetectionTrainer):
    """Detection trainer replacing only the training dataset with overlap-tile sampling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode=mode, batch=batch)
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return OverlapTileDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0,
            prefix=colorstr("train: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )
