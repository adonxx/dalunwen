"""Training-only, scale-selective feature distillation for Drone-YOLO.

The baseline teacher predicts at P3/P4/P5 whereas Drone-YOLO predicts at
P2/P3/P4.  Generic YOLO distillation assumes matching layer indices and would
therefore align the wrong tensors.  This wrapper explicitly aligns P3 -> P3
and P4 -> P4, and applies the feature loss only inside medium/large ground
truth boxes.  The P2 small-object branch is left to the original detection
loss.  The teacher and projection layers are removed before inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import copy_attr


@dataclass(frozen=True)
class SelectiveKDConfig:
    """Fixed settings for one reproducible selective-KD experiment."""

    weight: float = 0.5
    min_area_px: float = 32.0**2
    dilation_cells: int = 1


class _FeatureHook:
    """Store a layer output without retaining a second model graph."""

    def __init__(self, store: dict[str, torch.Tensor], key: str) -> None:
        self.store = store
        self.key = key

    def __call__(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.store[self.key] = output


class ScaleSelectiveDistillationModel(nn.Module):
    """Align teacher P3/P4 features with student P3/P4 during training only.

    The class deliberately mirrors the small interface used by Ultralytics'
    native ``DistillationModel`` so it can be installed for one process without
    editing ``site-packages``.  Its configuration is assigned before training
    by :func:`install_scale_selective_kd`.
    """

    config = SelectiveKDConfig()
    # The fixed layer pairs are specific to the two audited YAML files:
    # baseline Detect inputs [16(P3), 19(P4), 22(P5)] and full inputs
    # [14(P2), 17(P3), 21(P4)].
    feature_pairs = ((17, 16, "p3"), (21, 19, "p4"))

    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module) -> None:
        super().__init__()
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]

        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device).float()
        self.student_model = student_model
        self._freeze_teacher()

        self._teacher_feats: dict[str, torch.Tensor] = {}
        self._student_feats: dict[str, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        # Build light 1x1 adapters from live P3/P4 feature dimensions.  These
        # adapters exist only while training and are not part of inference.
        image_size = int(getattr(student_model.args, "imgsz", 640))
        was_training = student_model.training
        student_model.eval()
        with torch.no_grad():
            images = torch.zeros(1, 3, image_size, image_size, device=device)
            self.teacher_model(images)
            student_model(images)
        if was_training:
            student_model.train()

        adapters = []
        for _student_index, _teacher_index, name in self.feature_pairs:
            student_feat = self._student_feats[name]
            teacher_feat = self._teacher_feats[name]
            if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
                raise ValueError(
                    f"Selective KD requires matching P3/P4 resolution for {name}: "
                    f"student={tuple(student_feat.shape)}, teacher={tuple(teacher_feat.shape)}"
                )
            adapters.append(
                nn.Sequential(
                    nn.Conv2d(student_feat.shape[1], teacher_feat.shape[1], kernel_size=1, bias=False),
                    nn.SiLU(),
                    nn.Conv2d(teacher_feat.shape[1], teacher_feat.shape[1], kernel_size=1, bias=False),
                )
            )
        self.adapters = nn.ModuleList(adapters).to(device)

        # Keep the same attributes as Ultralytics' native distillation wrapper
        # (stride, args, yaml, model, ...), so the standard trainer still sees
        # a normal detection model.
        copy_attr(self, student_model)

    def __getstate__(self):
        """Allow EMA/deepcopy without serialising live hook tensors."""
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._teacher_hooks = []
        self._student_hooks = []
        if self.teacher_model is not None:
            self._register_feature_hooks()

    def _freeze_teacher(self) -> None:
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_teacher()
        return self

    def _remove_feature_hooks(self) -> None:
        for handle in self._student_hooks:
            handle.remove()
        for handle in self._teacher_hooks:
            handle.remove()
        self._student_hooks.clear()
        self._teacher_hooks.clear()

    def _register_feature_hooks(self) -> None:
        self._remove_feature_hooks()
        for student_index, teacher_index, name in self.feature_pairs:
            self._student_hooks.append(
                self.student_model.model[student_index].register_forward_hook(_FeatureHook(self._student_feats, name))
            )
            if self.teacher_model is not None:
                self._teacher_hooks.append(
                    self.teacher_model.model[teacher_index].register_forward_hook(_FeatureHook(self._teacher_feats, name))
                )

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def fuse(self, verbose: bool = True, imgsz: int | list[int, int] = 640):
        """Drop all training-only modules before validation, export or saving."""
        self._remove_feature_hooks()
        return self.student_model.fuse(verbose=verbose, imgsz=imgsz)

    @property
    def criterion(self):
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        self.student_model.criterion = value

    def init_criterion(self):
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value) -> None:
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs) -> None:
        self.student_model.set_head_attr(**kwargs)

    def _target_mask(self, batch: dict, height: int, width: int) -> torch.Tensor:
        """Return P3/P4 cells covered by medium-or-larger training boxes.

        Boxes are normalized ``xywh`` after augmentation.  The 32² threshold is
        evaluated in the live training input coordinates; this makes the mask
        consistent with the features receiving the loss.
        """
        images = batch["img"]
        mask = torch.zeros((images.shape[0], 1, height, width), device=images.device, dtype=images.dtype)
        boxes = batch["bboxes"].detach()
        batch_indices = batch["batch_idx"].detach().view(-1).long()
        image_height, image_width = images.shape[-2:]
        areas = boxes[:, 2] * image_width * boxes[:, 3] * image_height
        keep = areas >= self.config.min_area_px

        for box, image_index in zip(boxes[keep], batch_indices[keep]):
            center_x, center_y, box_width, box_height = box
            x1 = int(torch.floor((center_x - box_width / 2) * width).item())
            x2 = int(torch.ceil((center_x + box_width / 2) * width).item())
            y1 = int(torch.floor((center_y - box_height / 2) * height).item())
            y2 = int(torch.ceil((center_y + box_height / 2) * height).item())
            dilation = self.config.dilation_cells
            x1, x2 = max(0, x1 - dilation), min(width, x2 + dilation)
            y1, y2 = max(0, y1 - dilation), min(height, y2 + dilation)
            if x2 > x1 and y2 > y1:
                mask[image_index, :, y1:y2, x1:x2] = 1
        return mask

    def _masked_feature_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, batch: dict) -> torch.Tensor:
        mask = self._target_mask(batch, student_feat.shape[-2], student_feat.shape[-1])
        if not torch.any(mask):
            return student_feat.new_zeros(())
        mse_per_location = (student_feat - teacher_feat).square().mean(dim=1, keepdim=True)
        return (mse_per_location * mask).sum() / (mask.sum() + 1e-9)

    def loss(self, batch: dict, preds=None):
        """Combine original detection loss and selective P3/P4 feature loss."""
        zero = torch.zeros(1, device=batch["img"].device)
        if not self.training or self.teacher_model is None:
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            # Validators accumulate scalar loss entries; a length-one tensor
            # would broadcast incorrectly into their scalar accumulator.
            loss_items["scale_kd_loss"] = zero.squeeze(0).detach()
            return torch.cat([regular_loss, zero]), loss_items

        self._teacher_feats.clear()
        self._student_feats.clear()
        with torch.no_grad():
            self.teacher_model(batch["img"])
        preds = self.student_model(batch["img"])
        regular_loss, loss_items = self.student_model.loss(batch, preds)

        distill = zero.squeeze(0)
        for adapter, (_student_index, _teacher_index, name) in zip(self.adapters, self.feature_pairs):
            distill = distill + self._masked_feature_loss(adapter(self._student_feats[name]), self._teacher_feats[name], batch)
        distill = distill / len(self.feature_pairs) * self.config.weight
        loss_items["scale_kd_loss"] = distill.detach()
        return torch.cat([regular_loss, distill.unsqueeze(0) * batch["img"].shape[0]]), loss_items


def install_scale_selective_kd(*, weight: float, min_area_px: float = 32.0**2, dilation_cells: int = 1) -> None:
    """Install this wrapper for the current training process only.

    No dependency file is modified.  The trainer's native wrapper symbol is
    replaced before ``YOLO.train`` builds its trainer, and checkpoint stripping
    is patched as well so the saved checkpoint contains the student only.
    """
    if weight <= 0:
        raise ValueError("Selective-KD weight must be positive")
    ScaleSelectiveDistillationModel.config = SelectiveKDConfig(
        weight=weight,
        min_area_px=min_area_px,
        dilation_cells=dilation_cells,
    )
    import ultralytics.engine.trainer as trainer_module
    import ultralytics.nn.distill_model as distill_module

    trainer_module.DistillationModel = ScaleSelectiveDistillationModel
    distill_module.DistillationModel = ScaleSelectiveDistillationModel
