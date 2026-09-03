from __future__ import annotations

import sys
from pathlib import Path

import torch
from ultralytics.utils.tal import TaskAlignedAssigner


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo.assigner import SmallAwareTaskAlignedAssigner, normalized_wasserstein_similarity  # noqa: E402


def _metrics(assigner, gt_box: torch.Tensor, pred_box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assigner.bs = 1
    assigner.n_max_boxes = 1
    pd_scores = torch.tensor([[[0.8]]])
    pd_bboxes = pred_box.reshape(1, 1, 4)
    gt_labels = torch.zeros((1, 1, 1))
    gt_bboxes = gt_box.reshape(1, 1, 4)
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)
    return assigner.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)


def test_normalized_wasserstein_similarity_is_one_for_identical_boxes() -> None:
    boxes = torch.tensor([[10.0, 20.0, 14.0, 26.0]])
    similarity = normalized_wasserstein_similarity(boxes, boxes)
    assert torch.allclose(similarity, torch.ones_like(similarity), atol=1e-7)


def test_small_boxes_use_hybrid_localization_quality() -> None:
    gt = torch.tensor([0.0, 0.0, 16.0, 16.0])
    pred = torch.tensor([2.0, 0.0, 18.0, 16.0])
    original = TaskAlignedAssigner(topk=1, num_classes=1, alpha=0.5, beta=6.0, stride=[4, 8, 16])
    small_aware = SmallAwareTaskAlignedAssigner(
        topk=1, num_classes=1, alpha=0.5, beta=6.0, stride=[4, 8, 16]
    )
    _, original_quality = _metrics(original, gt, pred)
    _, hybrid_quality = _metrics(small_aware, gt, pred)
    nwd = normalized_wasserstein_similarity(gt, pred)
    expected = 0.5 * original_quality.squeeze() + 0.5 * nwd.squeeze()
    assert torch.allclose(hybrid_quality.squeeze(), expected, atol=1e-7)
    assert not torch.allclose(hybrid_quality, original_quality)


def test_non_small_boxes_match_original_tal() -> None:
    gt = torch.tensor([0.0, 0.0, 64.0, 64.0])
    pred = torch.tensor([2.0, 0.0, 66.0, 64.0])
    original = TaskAlignedAssigner(topk=1, num_classes=1, alpha=0.5, beta=6.0, stride=[4, 8, 16])
    small_aware = SmallAwareTaskAlignedAssigner(
        topk=1, num_classes=1, alpha=0.5, beta=6.0, stride=[4, 8, 16]
    )
    original_metric, original_quality = _metrics(original, gt, pred)
    hybrid_metric, hybrid_quality = _metrics(small_aware, gt, pred)
    assert torch.allclose(hybrid_quality, original_quality, atol=1e-7)
    assert torch.allclose(hybrid_metric, original_metric, atol=1e-7)
