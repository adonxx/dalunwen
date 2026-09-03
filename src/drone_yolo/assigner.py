"""Small-object-aware label assignment for controlled Drone-YOLO experiments."""

from __future__ import annotations

import torch
from ultralytics.utils.tal import TaskAlignedAssigner


def normalized_wasserstein_similarity(
    box1: torch.Tensor,
    box2: torch.Tensor,
    constant: float = 12.8,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return NWD similarity for xyxy boxes in absolute pixel coordinates."""
    if constant <= 0:
        raise ValueError("NWD normalization constant must be positive")
    center1 = (box1[..., :2] + box1[..., 2:]) / 2
    center2 = (box2[..., :2] + box2[..., 2:]) / 2
    wh1 = (box1[..., 2:] - box1[..., :2]).clamp_min(eps)
    wh2 = (box2[..., 2:] - box2[..., :2]).clamp_min(eps)
    distance_squared = (center1 - center2).square().sum(-1)
    distance_squared = distance_squared + ((wh1 - wh2) / 2).square().sum(-1)
    distance = torch.sqrt(distance_squared + eps) - eps**0.5
    return torch.exp(-distance / constant)


class SmallAwareTaskAlignedAssigner(TaskAlignedAssigner):
    """Use a CIoU/NWD localization quality only for input-space small GT boxes."""

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2=None,
        small_area_threshold: float = 32.0**2,
        nwd_constant: float = 12.8,
        nwd_weight: float = 0.5,
    ) -> None:
        super().__init__(
            topk=topk,
            num_classes=num_classes,
            alpha=alpha,
            beta=beta,
            stride=stride,
            eps=eps,
            topk2=topk2,
        )
        if small_area_threshold <= 0:
            raise ValueError("small_area_threshold must be positive")
        if nwd_constant <= 0:
            raise ValueError("nwd_constant must be positive")
        if not 0.0 <= nwd_weight <= 1.0:
            raise ValueError("nwd_weight must lie in [0, 1]")
        self.small_area_threshold = small_area_threshold
        self.nwd_constant = nwd_constant
        self.nwd_weight = nwd_weight

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute TAL metrics with a small-GT-only hybrid localization quality."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        quality = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device
        )

        batch_ind = torch.arange(self.bs, device=pd_scores.device)[:, None]
        bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_labels.squeeze(-1).long()][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        ciou = self.iou_calculation(gt_boxes, pd_boxes)
        nwd = normalized_wasserstein_similarity(gt_boxes, pd_boxes, constant=self.nwd_constant)
        gt_wh = (gt_boxes[..., 2:] - gt_boxes[..., :2]).clamp_min(0)
        is_small = gt_wh.prod(-1) < self.small_area_threshold
        hybrid_quality = (1.0 - self.nwd_weight) * ciou + self.nwd_weight * nwd
        quality[mask_gt] = torch.where(is_small, hybrid_quality, ciou).clamp(0.0, 1.0)

        align_metric = bbox_scores.pow(self.alpha) * quality.pow(self.beta)
        return align_metric, quality
