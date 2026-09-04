"""Lightweight feature-fusion modules for controlled MF-FPN ablations."""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules.conv import Concat


class AdaptiveWeightedConcat(Concat):
    """Concatenate adaptively reweighted inputs without changing their shape.

    Each branch receives a positive scalar weight. The weights are normalized
    to have mean one, so equal initialization is functionally equivalent to a
    standard concatenation while still allowing the optimizer to change the
    relative contribution of every incoming scale.
    """

    def __init__(self, dimension: int = 1, n_inputs: int = 2) -> None:
        super().__init__(dimension)
        if n_inputs < 2:
            raise ValueError("AdaptiveWeightedConcat requires at least two inputs")
        self.raw_weights = nn.Parameter(torch.zeros(n_inputs))

    def normalized_weights(self) -> torch.Tensor:
        """Return strictly positive weights whose mean equals one."""
        positive = nn.functional.softplus(self.raw_weights)
        return positive * (positive.numel() / positive.sum())

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        if len(x) != self.raw_weights.numel():
            raise ValueError(f"expected {self.raw_weights.numel()} inputs, received {len(x)}")
        weights = self.normalized_weights().to(dtype=x[0].dtype)
        return torch.cat([feature * weights[index] for index, feature in enumerate(x)], self.d)


class P5LiteSpatialGate(nn.Module):
    """Feed a compact P5 context back into P4 through a pixel-wise gate.

    The module deliberately keeps the input/output tensor shape unchanged.  It
    first creates a low-cost 1/32-scale semantic context from the 1/16-scale
    P4 feature, projects that context back to P4, and applies a one-channel
    spatial gate before residual fusion.  The gate bias starts at -2 so the
    untrained branch is conservative (sigmoid(-2) ~= 0.12), preserving the
    pre-existing MF-FPN path at the beginning of training.

    ``channels`` is explicit because the project YAML is fixed to the YOLO11s
    width multiplier; the Ultralytics generic custom-layer parser does not
    inject input channels for project-local modules.
    """

    def __init__(self, channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("channels and hidden_channels must be positive")
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.context_project = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spatial_gate = nn.Conv2d(channels * 2, 1, 1, bias=True)
        nn.init.constant_(self.spatial_gate.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"P5LiteSpatialGate expects [B, {self.channels}, H, W], received {tuple(x.shape)}"
            )
        context = self.downsample(x)
        context = nn.functional.interpolate(context, size=x.shape[-2:], mode="nearest")
        context = self.context_project(context)
        gate = torch.sigmoid(self.spatial_gate(torch.cat((x, context), dim=1)))
        return x + gate * context
