"""Box-regression losses used by the controlled Drone-YOLO ablations."""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss

from .assigner import SmallAwareTaskAlignedAssigner, normalized_wasserstein_similarity


def _box_geometry(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, ...]:
    """Return IoU, center-distance squared and enclosing diagonal squared."""
    b1_xy1, b1_xy2 = box1[..., :2], box1[..., 2:]
    b2_xy1, b2_xy2 = box2[..., :2], box2[..., 2:]
    wh1 = (b1_xy2 - b1_xy1).clamp_min(eps)
    wh2 = (b2_xy2 - b2_xy1).clamp_min(eps)
    intersection_wh = (torch.minimum(b1_xy2, b2_xy2) - torch.maximum(b1_xy1, b2_xy1)).clamp_min(0)
    intersection = intersection_wh.prod(-1)
    union = wh1.prod(-1) + wh2.prod(-1) - intersection + eps
    iou = intersection / union

    center1 = (b1_xy1 + b1_xy2) / 2
    center2 = (b2_xy1 + b2_xy2) / 2
    center_distance2 = (center1 - center2).square().sum(-1)
    enclosing_wh = torch.maximum(b1_xy2, b2_xy2) - torch.minimum(b1_xy1, b2_xy1)
    enclosing_diagonal2 = enclosing_wh.square().sum(-1).clamp_min(eps)
    return iou, center_distance2, enclosing_diagonal2, center1, center2, wh1, wh2


def _inner_iou(box1: torch.Tensor, box2: torch.Tensor, ratio: float, eps: float = 1e-7) -> torch.Tensor:
    """Calculate IoU between auxiliary boxes scaled about their centres."""
    center1 = (box1[..., :2] + box1[..., 2:]) / 2
    center2 = (box2[..., :2] + box2[..., 2:]) / 2
    half1 = (box1[..., 2:] - box1[..., :2]).clamp_min(eps) * ratio / 2
    half2 = (box2[..., 2:] - box2[..., :2]).clamp_min(eps) * ratio / 2
    inner1 = torch.cat((center1 - half1, center1 + half1), dim=-1)
    inner2 = torch.cat((center2 - half2, center2 + half2), dim=-1)
    return _box_geometry(inner1, inner2, eps)[0]


class InnerWIoULoss(nn.Module):
    """Inner-WIoU v3 with the paper's ratio, alpha and delta values."""

    def __init__(
        self,
        iou_mean: torch.Tensor | None = None,
        ratio: float = 0.8,
        alpha: float = 1.7,
        delta: float = 2.7,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        if iou_mean is None:
            self.register_buffer("iou_mean", torch.tensor(1.0))
        else:
            self.iou_mean = iou_mean

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        outer_iou, center_distance2, enclosing_diagonal2, *_ = _box_geometry(pred, target)
        base_iou_loss = 1.0 - outer_iou

        if self.training:
            with torch.no_grad():
                self.iou_mean.mul_(1.0 - self.momentum).add_(self.momentum * base_iou_loss.detach().mean())

        beta = base_iou_loss.detach() / self.iou_mean.clamp_min(1e-7)
        focusing = beta / (self.delta * torch.pow(self.alpha, beta - self.delta))
        distance_attention = torch.exp(center_distance2 / enclosing_diagonal2.detach())
        wiou_v1 = distance_attention * base_iou_loss
        wiou_v3 = focusing * wiou_v1

        inner_iou = _inner_iou(pred, target, self.ratio)
        return wiou_v3 + (outer_iou - inner_iou)


class NormalizedWassersteinLoss(nn.Module):
    """NWD loss for axis-aligned boxes represented as 2-D Gaussians.

    Inputs are expected in absolute pixel coordinates. Following the tiny
    object detection formulation, the Gaussian Wasserstein distance combines
    centre displacement with half-width/half-height displacement and is
    normalized by a fixed constant C=12.8.
    """

    def __init__(self, constant: float = 12.8, eps: float = 1e-7) -> None:
        super().__init__()
        if constant <= 0:
            raise ValueError("NWD normalization constant must be positive")
        self.constant = constant
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        similarity = normalized_wasserstein_similarity(pred, target, constant=self.constant, eps=self.eps)
        return 1.0 - similarity


def _foreground_boxes_in_pixels(
    boxes: torch.Tensor,
    fg_mask: torch.Tensor,
    stride: torch.Tensor,
) -> torch.Tensor:
    """Convert positive boxes from feature-grid units to absolute pixels."""
    stride_per_anchor = stride.reshape(1, -1, 1).expand(boxes.shape[0], -1, -1)
    return boxes[fg_mask] * stride_per_anchor[fg_mask]


class InnerWIoUBboxLoss(BboxLoss):
    """Ultralytics BboxLoss adapter preserving the original DFL calculation."""

    def __init__(self, reg_max: int, iou_mean: torch.Tensor) -> None:
        super().__init__(reg_max)
        self.inner_wiou = InnerWIoULoss(iou_mean=iou_mean, ratio=0.8, alpha=1.7, delta=2.7)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )
        weight = target_scores[fg_mask].sum(-1)
        loss = self.inner_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = (loss * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


class NWDBboxLoss(BboxLoss):
    """Ultralytics BboxLoss adapter replacing IoU regression with NWD."""

    def __init__(self, reg_max: int, constant: float = 12.8) -> None:
        super().__init__(reg_max)
        self.nwd = NormalizedWassersteinLoss(constant=constant)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )
        weight = target_scores[fg_mask].sum(-1)
        pred_pixels = _foreground_boxes_in_pixels(pred_bboxes, fg_mask, stride)
        target_pixels = _foreground_boxes_in_pixels(target_bboxes, fg_mask, stride)
        loss_nwd = self.nwd(pred_pixels, target_pixels)
        return (loss_nwd * weight).sum() / target_scores_sum, loss_dfl


class InnerWIoUNWDBboxLoss(BboxLoss):
    """Equal-weight hybrid of Inner-WIoU and pixel-space NWD."""

    def __init__(self, reg_max: int, iou_mean: torch.Tensor, nwd_weight: float = 0.5) -> None:
        super().__init__(reg_max)
        if not 0.0 <= nwd_weight <= 1.0:
            raise ValueError("nwd_weight must lie in [0, 1]")
        self.nwd_weight = nwd_weight
        self.inner_wiou = InnerWIoULoss(iou_mean=iou_mean, ratio=0.8, alpha=1.7, delta=2.7)
        self.nwd = NormalizedWassersteinLoss(constant=12.8)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )
        weight = target_scores[fg_mask].sum(-1)
        loss_inner = self.inner_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        pred_pixels = _foreground_boxes_in_pixels(pred_bboxes, fg_mask, stride)
        target_pixels = _foreground_boxes_in_pixels(target_bboxes, fg_mask, stride)
        loss_nwd = self.nwd(pred_pixels, target_pixels)
        loss = (1.0 - self.nwd_weight) * loss_inner + self.nwd_weight * loss_nwd
        return (loss * weight).sum() / target_scores_sum, loss_dfl


class InnerWIoUDetectionLoss(v8DetectionLoss):
    """Detection criterion that replaces CIoU while retaining TAL and DFL."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if "inner_wiou_mean" not in dict(model.named_buffers()):
            model.register_buffer("inner_wiou_mean", torch.tensor(1.0, device=self.device))
        self.bbox_loss = InnerWIoUBboxLoss(self.reg_max, model.inner_wiou_mean).to(self.device)


class SmallAwareInnerWIoUDetectionLoss(InnerWIoUDetectionLoss):
    """Inner-WIoU regression with small-object-aware NWD/TAL assignment."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        original = self.assigner
        self.assigner = SmallAwareTaskAlignedAssigner(
            topk=original.topk,
            num_classes=original.num_classes,
            alpha=original.alpha,
            beta=original.beta,
            stride=original.stride,
            eps=original.eps,
            topk2=original.topk2,
            small_area_threshold=32.0**2,
            nwd_constant=12.8,
            nwd_weight=0.5,
        )


class NWDDetectionLoss(v8DetectionLoss):
    """Detection criterion using NWD regression while retaining TAL and DFL."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bbox_loss = NWDBboxLoss(self.reg_max, constant=12.8).to(self.device)


class InnerWIoUNWDDetectionLoss(v8DetectionLoss):
    """Detection criterion with a fixed 50/50 Inner-WIoU and NWD mixture."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if "inner_wiou_mean" not in dict(model.named_buffers()):
            model.register_buffer("inner_wiou_mean", torch.tensor(1.0, device=self.device))
        self.bbox_loss = InnerWIoUNWDBboxLoss(
            self.reg_max,
            model.inner_wiou_mean,
            nwd_weight=0.5,
        ).to(self.device)
