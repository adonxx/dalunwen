"""Small-object-centred training crops without inference-time changes."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model


class SmallObjectCenteredCrop:
    """Zoom a random small object by cropping an aspect-preserving image region."""

    def __init__(
        self,
        probability: float = 0.5,
        crop_scale: float = 0.5,
        small_area_threshold: float = 32.0**2,
        min_visibility: float = 0.5,
        min_box_size: float = 2.0,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if not 0.0 < crop_scale < 1.0:
            raise ValueError("crop_scale must lie in (0, 1)")
        if small_area_threshold <= 0:
            raise ValueError("small_area_threshold must be positive")
        if not 0.0 <= min_visibility <= 1.0:
            raise ValueError("min_visibility must lie in [0, 1]")
        if min_box_size < 0:
            raise ValueError("min_box_size must be non-negative")
        self.probability = probability
        self.crop_scale = crop_scale
        self.small_area_threshold = small_area_threshold
        self.min_visibility = min_visibility
        self.min_box_size = min_box_size

    @staticmethod
    def _random_valid_origin(box_min: float, box_max: float, crop_size: int, image_size: int) -> int:
        """Choose an integer crop origin that keeps the selected box fully visible."""
        low = max(0, math.ceil(box_max - crop_size))
        high = min(math.floor(box_min), image_size - crop_size)
        if low <= high:
            return random.randint(low, high)
        center = (box_min + box_max) / 2
        return int(round(min(max(center - crop_size / 2, 0), image_size - crop_size)))

    def __call__(self, labels: dict) -> dict:
        if random.random() >= self.probability:
            return labels
        image = labels["img"]
        instances = labels["instances"]
        if len(instances) == 0:
            return labels

        height, width = image.shape[:2]
        crop_width = max(2, int(round(width * self.crop_scale)))
        crop_height = max(2, int(round(height * self.crop_scale)))
        if crop_width >= width or crop_height >= height:
            return labels

        original_format = instances._bboxes.format
        instances.convert_bbox("xyxy")
        instances.denormalize(width, height)
        boxes_before = instances.bboxes.copy()
        wh_before = np.maximum(boxes_before[:, 2:] - boxes_before[:, :2], 0.0)
        areas_before = wh_before.prod(axis=1)
        candidates = np.flatnonzero(areas_before < self.small_area_threshold)
        if len(candidates) == 0:
            instances.normalize(width, height)
            instances.convert_bbox(original_format)
            return labels

        selected = int(random.choice(candidates.tolist()))
        selected_box = boxes_before[selected]
        x0 = self._random_valid_origin(selected_box[0], selected_box[2], crop_width, width)
        y0 = self._random_valid_origin(selected_box[1], selected_box[3], crop_height, height)

        cropped = image[y0 : y0 + crop_height, x0 : x0 + crop_width]
        labels["img"] = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

        instances.add_padding(-x0, -y0)
        instances.clip(crop_width, crop_height)
        clipped_boxes = instances.bboxes
        clipped_wh = np.maximum(clipped_boxes[:, 2:] - clipped_boxes[:, :2], 0.0)
        clipped_areas = clipped_wh.prod(axis=1)
        visibility = clipped_areas / np.maximum(areas_before, 1e-9)
        keep = (
            (visibility >= self.min_visibility)
            & (clipped_wh[:, 0] >= self.min_box_size)
            & (clipped_wh[:, 1] >= self.min_box_size)
        )
        instances = instances[keep]
        labels["cls"] = labels["cls"][keep]

        instances.scale(width / crop_width, height / crop_height)
        instances.clip(width, height)
        instances.normalize(width, height)
        instances.convert_bbox(original_format)
        labels["instances"] = instances
        return labels


class SmallObjectCropDataset(YOLODataset):
    """YOLO detection dataset applying the crop only during training reads."""

    def __init__(self, *args, **kwargs) -> None:
        self.small_object_crop = SmallObjectCenteredCrop(
            probability=0.5,
            crop_scale=0.5,
            small_area_threshold=32.0**2,
            min_visibility=0.5,
            min_box_size=2.0,
        )
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index: int) -> dict:
        labels = super().get_image_and_label(index)
        return self.small_object_crop(labels) if self.augment else labels


class SmallObjectCropTrainer(DetectionTrainer):
    """Detection trainer replacing only the training dataset implementation."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode != "train":
            return super().build_dataset(img_path, mode=mode, batch=batch)
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return SmallObjectCropDataset(
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
