"""Model-stage factory for the paper's ablation sequence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

from .runtime import configure_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Stage:
    model_yaml: str
    use_lscd: bool = False
    use_inner_wiou: bool = False
    use_weighted_concat: bool = False
    use_nwd: bool = False
    use_inner_wiou_nwd: bool = False
    use_small_aware_tal: bool = False


STAGES = {
    "baseline": Stage("yolo11s_baseline.yaml"),
    "mffpn": Stage("drone-yolo11s-mffpn.yaml"),
    "mffpn_weighted": Stage("drone-yolo11s-mffpn.yaml", use_weighted_concat=True),
    "mffpn_p2p5": Stage("drone-yolo11s-mffpn-p2p5.yaml"),
    "mffpn_p3p5": Stage("drone-yolo11s-mffpn-p3p5.yaml"),
    "lscd": Stage("drone-yolo11s-mffpn.yaml", use_lscd=True),
    "full": Stage("drone-yolo11s-mffpn.yaml", use_lscd=True, use_inner_wiou=True),
    "lscd_nwd": Stage("drone-yolo11s-mffpn.yaml", use_lscd=True, use_nwd=True),
    "full_nwd_hybrid": Stage("drone-yolo11s-mffpn.yaml", use_lscd=True, use_inner_wiou_nwd=True),
    "full_small_tal": Stage(
        "drone-yolo11s-mffpn.yaml",
        use_lscd=True,
        use_inner_wiou=True,
        use_small_aware_tal=True,
    ),
}


def build_model(stage_name: str, weights: str | None = None) -> YOLO:
    """Build one ablation stage and optionally transfer compatible weights."""
    if stage_name not in STAGES:
        raise KeyError(f"unknown stage {stage_name!r}; choose from {', '.join(STAGES)}")
    stage = STAGES[stage_name]
    configure_runtime(
        use_lscd=stage.use_lscd,
        use_inner_wiou=stage.use_inner_wiou,
        use_weighted_concat=stage.use_weighted_concat,
        use_nwd=stage.use_nwd,
        use_inner_wiou_nwd=stage.use_inner_wiou_nwd,
        use_small_aware_tal=stage.use_small_aware_tal,
    )
    model_path = PROJECT_ROOT / "configs" / "models" / stage.model_yaml
    model = YOLO(str(model_path), task="detect")
    if (stage.use_inner_wiou or stage.use_inner_wiou_nwd) and "inner_wiou_mean" not in dict(model.model.named_buffers()):
        model.model.register_buffer("inner_wiou_mean", torch.tensor(1.0, device=next(model.model.parameters()).device))
    if weights:
        model.load(weights)
    return model
