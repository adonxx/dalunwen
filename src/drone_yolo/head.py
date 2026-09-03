"""Lightweight shared-convolution detection head used by Drone-YOLO."""

from __future__ import annotations

import math

import torch
from torch import nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import autopad


def _valid_groups(channels: int, preferred: int) -> int:
    """Return the largest valid GroupNorm group count not exceeding preferred."""
    for groups in range(min(channels, preferred), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class Scale(nn.Module):
    """Learnable scale used to retain level-specific regression behaviour."""

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class ConvGN(nn.Module):
    """Convolution followed by GroupNorm and SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        gn_groups: int = 32,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride,
            autopad(kernel_size),
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_valid_groups(c2, gn_groups), c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LSCDDetect(Detect):
    """YOLO11-compatible lightweight shared-convolution detection head.

    The topology follows Figure 3 and the public Detect_LSCD implementation:
    each pyramid level is aligned by a dedicated 1x1 Conv-GN block, two full
    3x3 Conv-GN blocks are shared across levels, and the 1x1 regression and
    classification predictors are shared as well. Per-level learnable scales
    retain scale-specific regression behaviour.

    The paper fixes GroupNorm to 32 groups but does not disclose the hidden
    width. A width of 32 is used because it preserves exactly 32 GN groups and
    most closely matches the reported 19.6 GFLOPs among valid widths.
    """

    hidden_channels = 32

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()) -> None:
        nn.Module.__init__(self)
        if end2end:
            raise ValueError("LSCDDetect reproduces the paper's one-to-many YOLO11 head only")

        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.end2end = False
        self.dynamic = False
        self.export = False
        self.format = None
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

        hidden = self.hidden_channels
        self.input_conv = nn.ModuleList(ConvGN(c, hidden, 1, gn_groups=32) for c in ch)
        self.shared_conv = nn.Sequential(
            ConvGN(hidden, hidden, 3, gn_groups=32),
            ConvGN(hidden, hidden, 3, gn_groups=32),
        )
        self.cv2 = nn.Conv2d(hidden, 4 * reg_max, 1)
        self.cv3 = nn.Conv2d(hidden, nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply shared feature extraction and independent prediction layers."""
        box_head = self.cv2 if box_head is None else box_head
        cls_head = self.cv3 if cls_head is None else cls_head
        features = [self.shared_conv(self.input_conv[i](feature)) for i, feature in enumerate(x)]
        batch = features[0].shape[0]
        boxes = torch.cat(
            [self.scale[i](box_head(features[i])).view(batch, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head(features[i]).view(batch, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": features}

    def bias_init(self) -> None:
        """Initialize the shared predictors after model strides are known."""
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def fuse(self) -> None:
        """Keep the custom shared head intact during Ultralytics model fusion."""
