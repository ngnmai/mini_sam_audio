"""Model loading utilities for mini SAM-Audio student model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def _extract_state_dict(raw_checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "state_dict" in raw_checkpoint and isinstance(raw_checkpoint["state_dict"], dict):
        return raw_checkpoint["state_dict"]
    if "model" in raw_checkpoint and isinstance(raw_checkpoint["model"], dict):
        return raw_checkpoint["model"]
    return raw_checkpoint


def _load_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    map_location: str = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict-like object")
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return model


def load_student_model(
    *,
    config: Optional[object] = None,
    checkpoint_path: Optional[str] = None,
    strict: bool = True,
    map_location: str = "cpu",
    **config_kwargs: Any,
):
    """Create the mini SAM-Audio student model and optionally load weights."""
    from mini_sam_audio.model.config import SAMAudioConfig as StudentConfig
    from mini_sam_audio.model.model import SAMAudio as StudentSAMAudio

    student_config = config or StudentConfig(**config_kwargs)
    model = StudentSAMAudio(student_config)
    if checkpoint_path is not None:
        _load_weights(
            model,
            checkpoint_path,
            map_location=map_location,
            strict=strict,
        )
    return model