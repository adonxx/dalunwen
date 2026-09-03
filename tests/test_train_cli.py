from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402


def test_seed_can_be_overridden_from_the_command_line(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage", "mffpn", "--seed", "2"])
    args = train.parse_args()
    assert args.seed == 2


def test_small_object_crop_can_be_enabled_from_the_command_line(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage", "full", "--small-object-crop"])
    args = train.parse_args()
    assert args.small_object_crop is True


def test_overlap_tile_training_can_be_enabled_from_the_command_line(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage", "full", "--overlap-tile-training"])
    args = train.parse_args()
    assert args.overlap_tile_training is True
