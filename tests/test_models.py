from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from ultralytics.cfg import get_cfg


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_yolo import build_model  # noqa: E402
from drone_yolo.fusion import AdaptiveWeightedConcat  # noqa: E402
from drone_yolo.head import LSCDDetect  # noqa: E402
from drone_yolo.loss import (  # noqa: E402
    InnerWIoUDetectionLoss,
    InnerWIoUNWDDetectionLoss,
    NWDDetectionLoss,
    SmallAwareInnerWIoUDetectionLoss,
)


@pytest.mark.parametrize(
    ("stage", "expected_stride"),
    [
        ("baseline", [8.0, 16.0, 32.0]),
        ("mffpn", [4.0, 8.0, 16.0]),
        ("mffpn_weighted", [4.0, 8.0, 16.0]),
        ("mffpn_p2p5", [4.0, 8.0, 16.0, 32.0]),
        ("mffpn_p3p5", [8.0, 16.0, 32.0]),
        ("lscd", [4.0, 8.0, 16.0]),
        ("lscd_nwd", [4.0, 8.0, 16.0]),
        ("full_nwd_hybrid", [4.0, 8.0, 16.0]),
        ("full_small_tal", [4.0, 8.0, 16.0]),
    ],
)
def test_stage_builds_with_expected_strides(stage: str, expected_stride: list[float]) -> None:
    model = build_model(stage).model
    assert model.stride.tolist() == expected_stride
    if stage == "lscd":
        assert isinstance(model.model[-1], LSCDDetect)


def test_scale_ablation_models_share_the_same_feature_path() -> None:
    four_scale = build_model("mffpn_p2p5").model
    three_scale = build_model("mffpn_p3p5").model
    assert len(four_scale.model) == len(three_scale.model)
    for four_module, three_module in zip(four_scale.model[:-1], three_scale.model[:-1]):
        four_shapes = [tuple(parameter.shape) for parameter in four_module.parameters()]
        three_shapes = [tuple(parameter.shape) for parameter in three_module.parameters()]
        assert four_shapes == three_shapes
    assert four_scale.model[-1].nl == 4
    assert three_scale.model[-1].nl == 3


def test_weighted_fusion_is_initially_equivalent_and_stage_local() -> None:
    weighted = build_model("mffpn_weighted").model
    fusion_layers = [module for module in weighted.model if isinstance(module, AdaptiveWeightedConcat)]
    assert len(fusion_layers) == 4
    assert sum(module.raw_weights.numel() for module in fusion_layers) == 8
    for module in fusion_layers:
        assert torch.allclose(module.normalized_weights(), torch.ones(2), atol=1e-6)
        left = torch.randn(2, 3, 8, 8)
        right = torch.randn(2, 5, 8, 8)
        assert torch.allclose(module([left, right]), torch.cat([left, right], dim=1), atol=1e-6)

    standard = build_model("mffpn").model
    assert not any(isinstance(module, AdaptiveWeightedConcat) for module in standard.model)


def test_weighted_fusion_receives_branch_specific_gradients() -> None:
    model = build_model("mffpn_weighted").model
    model.args = get_cfg()
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.15, 0.15]]),
    }
    losses, _ = model(batch)
    losses.sum().backward()
    gradients = [
        module.raw_weights.grad
        for module in model.model
        if isinstance(module, AdaptiveWeightedConcat)
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(not torch.allclose(gradient, torch.zeros_like(gradient)) for gradient in gradients)


def test_lscd_matches_shared_head_topology() -> None:
    head = build_model("lscd").model.model[-1]
    assert isinstance(head, LSCDDetect)
    assert len(head.input_conv) == 3
    assert all(layer.conv.kernel_size == (1, 1) for layer in head.input_conv)
    assert len(head.shared_conv) == 2
    assert all(layer.conv.kernel_size == (3, 3) for layer in head.shared_conv)
    assert all(layer.conv.groups == 1 for layer in head.shared_conv)
    assert all(layer.norm.num_groups == 32 for layer in head.shared_conv)
    assert isinstance(head.cv2, nn.Conv2d)
    assert isinstance(head.cv3, nn.Conv2d)
    before = (id(head.cv2), id(head.cv3))
    head.fuse()
    assert (id(head.cv2), id(head.cv3)) == before


@pytest.mark.parametrize("stage", ["full", "lscd_nwd", "full_nwd_hybrid", "full_small_tal"])
def test_custom_loss_stages_compute_finite_loss_and_gradients(stage: str) -> None:
    model = build_model(stage).model
    model.args = get_cfg()
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.15, 0.15]]),
    }
    losses, _ = model(batch)
    total = losses.sum()
    assert torch.isfinite(total)
    total.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize(
    ("stage", "criterion_type"),
    [
        ("full", InnerWIoUDetectionLoss),
        ("lscd_nwd", NWDDetectionLoss),
        ("full_nwd_hybrid", InnerWIoUNWDDetectionLoss),
        ("full_small_tal", SmallAwareInnerWIoUDetectionLoss),
    ],
)
def test_custom_loss_stage_uses_expected_criterion(stage: str, criterion_type: type) -> None:
    model = build_model(stage).model
    assert isinstance(model.init_criterion(), criterion_type)
