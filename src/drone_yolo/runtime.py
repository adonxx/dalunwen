"""Runtime registration of local modules without editing site-packages."""

from __future__ import annotations

from ultralytics.nn import tasks

from .fusion import AdaptiveWeightedConcat, P5LiteSpatialGate
from .head import LSCDDetect
from .loss import (
    InnerWIoUDetectionLoss,
    InnerWIoUNWDDetectionLoss,
    NWDDetectionLoss,
    SmallAwareInnerWIoUDetectionLoss,
)


_ORIGINAL_DETECT = tasks.Detect
_ORIGINAL_CONCAT = tasks.Concat
_ORIGINAL_DETECTION_LOSS = tasks.v8DetectionLoss


def configure_runtime(
    *,
    use_lscd: bool,
    use_inner_wiou: bool,
    use_weighted_concat: bool = False,
    use_nwd: bool = False,
    use_inner_wiou_nwd: bool = False,
    use_small_aware_tal: bool = False,
    use_p5lite_gate: bool = False,
) -> None:
    """Select local head/loss implementations for the current process only."""
    selected_losses = sum((use_inner_wiou, use_nwd, use_inner_wiou_nwd))
    if selected_losses > 1:
        raise ValueError("Only one custom regression loss may be active at a time")
    tasks.Detect = LSCDDetect if use_lscd else _ORIGINAL_DETECT
    tasks.Concat = AdaptiveWeightedConcat if use_weighted_concat else _ORIGINAL_CONCAT
    # parse_model resolves project-local YAML module names from this namespace.
    # Registering the class is harmless for all other stages because only the
    # P5-lite YAML names it.
    if use_p5lite_gate:
        tasks.P5LiteSpatialGate = P5LiteSpatialGate
    if use_small_aware_tal and not use_inner_wiou:
        raise ValueError("Small-aware TAL is currently defined only with Inner-WIoU regression")
    if use_small_aware_tal:
        tasks.v8DetectionLoss = SmallAwareInnerWIoUDetectionLoss
    elif use_inner_wiou:
        tasks.v8DetectionLoss = InnerWIoUDetectionLoss
    elif use_nwd:
        tasks.v8DetectionLoss = NWDDetectionLoss
    elif use_inner_wiou_nwd:
        tasks.v8DetectionLoss = InnerWIoUNWDDetectionLoss
    else:
        tasks.v8DetectionLoss = _ORIGINAL_DETECTION_LOSS
