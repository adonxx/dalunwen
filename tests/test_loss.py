from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo.loss import InnerWIoULoss, NormalizedWassersteinLoss  # noqa: E402


def test_identical_boxes_have_zero_loss() -> None:
    boxes = torch.tensor([[1.0, 2.0, 5.0, 8.0]])
    loss = InnerWIoULoss()(boxes, boxes)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_loss_is_finite_and_differentiable() -> None:
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0]], requires_grad=True)
    target = torch.tensor([[0.5, 0.5, 2.5, 2.5], [0.0, 0.0, 1.0, 1.0]])
    loss = InnerWIoULoss()(pred, target).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_nwd_identical_boxes_have_zero_loss() -> None:
    boxes = torch.tensor([[10.0, 20.0, 14.0, 26.0]])
    loss = NormalizedWassersteinLoss()(boxes, boxes)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_nwd_is_symmetric_bounded_and_differentiable() -> None:
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [4.0, 3.0, 9.0, 8.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 3.0, 2.0], [3.0, 2.0, 7.0, 6.0]])
    criterion = NormalizedWassersteinLoss(constant=12.8)
    forward = criterion(pred, target)
    reverse = criterion(target, pred)
    assert torch.allclose(forward, reverse, atol=1e-7)
    assert torch.all((forward >= 0.0) & (forward < 1.0))
    forward.sum().backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
