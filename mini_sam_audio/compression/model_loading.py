"""Model loading utilities for mini SAM-Audio student model."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional

import torch

from mini_sam_audio.model.config import ImageBindRankerConfig, MiniSAMAudioConfig
from mini_sam_audio.model.model import MiniSAMAudio
from mini_sam_audio.processor import MiniSAMAudioProcessor


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


def _apply_visual_ranker_options(
    config: MiniSAMAudioConfig,
    use_visual_ranker: Optional[bool],
    visual_ranker_checkpoint: Optional[Path],
) -> None:
    if use_visual_ranker is not None:
        config.use_visual_ranker = use_visual_ranker
    if visual_ranker_checkpoint is None:
        return
    if not config.use_visual_ranker:
        print(
            f"[ranker] warning: visual_ranker_checkpoint={visual_ranker_checkpoint} ignored "
            "because the visual ranker is disabled",
            flush=True,
        )
        return
    if not isinstance(config.visual_ranker, ImageBindRankerConfig):
        kind = getattr(config.visual_ranker, "kind", None)
        raise ValueError(
            f"visual_ranker_checkpoint is only supported for the imagebind ranker; "
            f"config visual_ranker kind is {kind!r}."
        )
    resolved = Path(visual_ranker_checkpoint).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"visual_ranker_checkpoint not found: {resolved}")
    # Copy so shared or default ranker config objects are never mutated.
    ranker_config = copy.copy(config.visual_ranker)
    ranker_config.checkpoint = str(resolved)
    config.visual_ranker = ranker_config


def load_student_model(
    *,
    init_mode: str = "checkpoint",
    config: Optional[object] = None,
    checkpoint_path: Optional[str] = None,
    strict: bool = True,
    map_location: str = "cpu",
    freeze_inference_backbones: Optional[bool] = None,
    pretrained_backbones: Optional[bool] = None,
    use_visual_ranker: Optional[bool] = None,
    visual_ranker_checkpoint: Optional[Path] = None,
    **config_kwargs: Any,
):
    """Create the mini SAM-Audio student model per init_mode and optionally load weights.

    Mirrors the train_mini_sam_audio.py bootstrap contract: ``init_mode="scratch"`` builds a
    fresh model from local config with no checkpoint required; ``init_mode="checkpoint"`` loads
    weights via the existing checkpoint path. Both modes default to frozen audio_codec/
    vision_encoder backbones, and this training method trains everything else from scratch.
    Pass ``freeze_inference_backbones=False`` to override and keep backbones trainable.

    ``pretrained_backbones=None`` means scratch downloads the pretrained DAC VAE and PE weights
    from Hugging Face, while checkpoint mode only builds the architecture and takes the backbone
    tensors from the checkpoint; an explicit bool overrides both modes.

    ``use_visual_ranker=None`` keeps the config value; ``visual_ranker_checkpoint`` points the
    ImageBind ranker at a local weights file instead of downloading it.
    """
    if init_mode not in ("scratch", "checkpoint"):
        raise ValueError(f"init_mode must be 'scratch' or 'checkpoint', got {init_mode!r}")
    if init_mode == "scratch" and checkpoint_path is not None:
        raise ValueError(
            "checkpoint_path is not used in init_mode='scratch'; omit it or use "
            "init_mode='checkpoint'."
        )
    if init_mode == "checkpoint" and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when init_mode='checkpoint'.")

    if config is None:
        student_config = MiniSAMAudioConfig(**config_kwargs)
    elif isinstance(config, dict):
        student_config = MiniSAMAudioConfig(**config)
    else:
        student_config = copy.copy(config)
    student_config.pretrained_backbones = (
        init_mode == "scratch" if pretrained_backbones is None else pretrained_backbones
    )
    _apply_visual_ranker_options(student_config, use_visual_ranker, visual_ranker_checkpoint)
    model = MiniSAMAudio.from_config(student_config)

    if init_mode == "checkpoint":
        _load_weights(
            model,
            checkpoint_path,
            map_location=map_location,
            strict=strict,
        )
        default_freeze = True
    else:
        # Scratch mode loads the pretrained DAC VAE and PE weights and freezes them by default;
        # the rest of the model trains from scratch.
        default_freeze = True

    should_freeze = default_freeze if freeze_inference_backbones is None else freeze_inference_backbones
    if should_freeze:
        model.freeze_inference_backbones()
    return model


def _load_json_config(path: Path, arg_name: str) -> dict:
    """Load a local JSON config file, rejecting anything that looks like a remote HF repo id."""
    if not path.exists() or path.is_dir():
        raise ValueError(
            f"{arg_name}={path} must be an existing local JSON file. Remote Hugging Face repo ids "
            "are not supported for this argument; download the config locally first."
        )
    with path.open() as fin:
        return json.load(fin)


def _build_model_config(model_config_path: Optional[Path]) -> tuple[MiniSAMAudioConfig, str]:
    if model_config_path is None:
        return MiniSAMAudioConfig(), "<built-in defaults>"
    config_dict = _load_json_config(model_config_path, "--model-config")
    return MiniSAMAudioConfig(**config_dict), str(model_config_path)


def _build_processor(
    model_config: MiniSAMAudioConfig, processor_config_path: Optional[Path]
) -> tuple[MiniSAMAudioProcessor, str]:
    if processor_config_path is None:
        return (
            MiniSAMAudioProcessor(
                audio_hop_length=model_config.audio_codec.hop_length,
                audio_sampling_rate=model_config.audio_codec.sample_rate,
            ),
            "<derived from model config>",
        )
    config_dict = _load_json_config(processor_config_path, "--processor-config")
    try:
        audio_hop_length = config_dict["audio_hop_length"]
        audio_sampling_rate = config_dict["audio_sampling_rate"]
    except KeyError as exc:
        raise ValueError(
            f"--processor-config={processor_config_path} is missing required field {exc}; expected "
            "both 'audio_hop_length' and 'audio_sampling_rate'."
        ) from exc
    return (
        MiniSAMAudioProcessor(
            audio_hop_length=audio_hop_length,
            audio_sampling_rate=audio_sampling_rate,
        ),
        str(processor_config_path),
    )


def _validate_config_parity(
    model_config: MiniSAMAudioConfig,
    processor: MiniSAMAudioProcessor,
    model_config_source: str,
    processor_config_source: str,
) -> None:
    mismatches = []
    if processor.audio_sampling_rate != model_config.audio_codec.sample_rate:
        mismatches.append(
            f"audio_sampling_rate: processor={processor.audio_sampling_rate} ({processor_config_source}) "
            f"!= model audio_codec.sample_rate={model_config.audio_codec.sample_rate} ({model_config_source})"
        )
    if processor.audio_hop_length != model_config.audio_codec.hop_length:
        mismatches.append(
            f"audio_hop_length: processor={processor.audio_hop_length} ({processor_config_source}) "
            f"!= model audio_codec.hop_length={model_config.audio_codec.hop_length} ({model_config_source})"
        )
    if mismatches:
        joined = "\n".join(f"  - {mismatch}" for mismatch in mismatches)
        raise RuntimeError(f"Model/processor configuration mismatch:\n{joined}")


def _resolve_checkpoint_model_config(
    checkpoint_path: Path, model_config_path: Optional[Path]
) -> tuple[MiniSAMAudioConfig, str]:
    if model_config_path is not None:
        return _build_model_config(model_config_path)
    sibling_config = checkpoint_path.parent / "config.json"
    if sibling_config.exists():
        with sibling_config.open() as fin:
            config_dict = json.load(fin)
        return MiniSAMAudioConfig(**config_dict), str(sibling_config)
    raise ValueError(
        "--init-mode=checkpoint requires a model config: pass --model-config explicitly, or place a "
        f"config.json next to the checkpoint at {sibling_config}."
    )


def bootstrap_model_and_processor(
    *,
    init_mode: str,
    checkpoint_path: Optional[str] = None,
    model_config_path: Optional[Path] = None,
    processor_config_path: Optional[Path] = None,
    unfreeze_inference_backbones: bool = False,
    strict: bool = True,
    pretrained_backbones: Optional[bool] = None,
    use_visual_ranker: Optional[bool] = None,
    visual_ranker_checkpoint: Optional[Path] = None,
) -> tuple[MiniSAMAudio, MiniSAMAudioProcessor, bool]:
    """Construct the model/processor per init_mode.

    Scratch mode builds the architecture from local JSON configs and downloads the pretrained
    DAC VAE and PE weights from Hugging Face, then freezes them. Checkpoint mode builds only the
    architecture and takes the backbone weights from the checkpoint. ``pretrained_backbones=None``
    selects these defaults; an explicit bool overrides both modes (e.g. False on resume, where
    the Lightning checkpoint restores the backbones). ``use_visual_ranker=None`` keeps the config
    value; ``visual_ranker_checkpoint`` points the ImageBind ranker at a local weights file.
    """
    if init_mode == "scratch":
        if checkpoint_path is not None and not Path(checkpoint_path).exists():
            raise ValueError(
                f"--checkpoint-path={checkpoint_path!r} is unused in scratch mode and is not a "
                "local path; omit --checkpoint-path or switch to --init-mode=checkpoint."
            )
        model_config, model_config_source = _build_model_config(model_config_path)
        processor, processor_config_source = _build_processor(model_config, processor_config_path)
        _validate_config_parity(model_config, processor, model_config_source, processor_config_source)
        model_config.pretrained_backbones = True if pretrained_backbones is None else pretrained_backbones
        _apply_visual_ranker_options(model_config, use_visual_ranker, visual_ranker_checkpoint)
        model = MiniSAMAudio.from_config(model_config)
        # Pretrained DAC VAE and PE weights are frozen by default; the rest trains from scratch.
        freeze_backbones = not unfreeze_inference_backbones
        return model, processor, freeze_backbones

    # Checkpoint mode.
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise ValueError(
            f"--checkpoint-path={resolved_checkpoint_path} does not exist locally. Checkpoint mode requires a "
            "local checkpoint; remote Hugging Face repo ids are not fetched automatically."
        )

    model_config, model_config_source = _resolve_checkpoint_model_config(
        resolved_checkpoint_path, model_config_path
    )
    # The checkpoint carries the backbone tensors, so skip the Hugging Face download by default.
    model_config.pretrained_backbones = False if pretrained_backbones is None else pretrained_backbones
    _apply_visual_ranker_options(model_config, use_visual_ranker, visual_ranker_checkpoint)
    model = MiniSAMAudio.from_config(model_config)
    _load_weights(model, str(resolved_checkpoint_path), strict=strict)

    if processor_config_path is not None:
        processor, processor_config_source = _build_processor(model_config, processor_config_path)
    elif resolved_checkpoint_path.is_dir() and (resolved_checkpoint_path / "config.json").exists():
        processor = MiniSAMAudioProcessor.from_pretrained(str(resolved_checkpoint_path))
        processor_config_source = str(resolved_checkpoint_path / "config.json")
    else:
        processor, processor_config_source = _build_processor(model_config, None)

    _validate_config_parity(model_config, processor, model_config_source, processor_config_source)
    # Checkpoint mode always freezes backbones to preserve prior behavior; the flag cannot unfreeze it.
    freeze_backbones = True
    return model, processor, freeze_backbones