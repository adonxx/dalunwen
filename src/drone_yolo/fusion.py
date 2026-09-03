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
