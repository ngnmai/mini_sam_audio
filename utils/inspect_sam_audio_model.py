"""Inspect SAM-Audio (submodule) model size, parameter counts, and structure.

Examples:
    python utils/inspect_sam_audio_model.py
    python utils/inspect_sam_audio_model.py --checkpoint facebook/sam-audio-large
    python utils/inspect_sam_audio_model.py --checkpoint /path/to/local/model/dir
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM_AUDIO_ROOT = PROJECT_ROOT / "submodule" / "sam-audio"

if str(SAM_AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_AUDIO_ROOT))

from sam_audio import SAMAudio


DEFAULT_CHECKPOINT = "facebook/sam-audio-small"
BACKBONE_PREFIXES = ("audio_codec.", "vision_encoder.", "text_encoder.")


@dataclass(frozen=True)
class ParamStats:
    total_numel: int
    total_bytes: int
    trainable_numel: int
    trainable_bytes: int
    frozen_numel: int
    frozen_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the SAM-Audio model (submodule/sam-audio) and report parameter counts, "
            "estimated model size, and a module structure summary."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=(
            "SAM-Audio checkpoint path or Hugging Face repo id, e.g. 'facebook/sam-audio-large' "
            "or a local directory containing config.json and checkpoint.pt. "
            f"Defaults to '{DEFAULT_CHECKPOINT}'."
        ),
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        default=False,
        help="Use strict=False for checkpoint loading (default is strict=True).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of largest parameter tensors to print.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Depth for module structure summary from the model root.",
    )
    parser.add_argument(
        "--fail-if-backbone-trainable",
        action="store_true",
        help=(
            "Exit with non-zero status if any trainable parameters are found under "
            "audio_codec., vision_encoder., or text_encoder."
        ),
    )
    return parser.parse_args()


def _human_bytes(size_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes:.2f} B"


def _numel_and_bytes(params: Iterable[torch.nn.Parameter]) -> tuple[int, int]:
    total_numel = 0
    total_bytes = 0
    for p in params:
        total_numel += p.numel()
        total_bytes += p.numel() * p.element_size()
    return total_numel, total_bytes


def _compute_stats(named_params: list[tuple[str, torch.nn.Parameter]]) -> ParamStats:
    total_numel = 0
    total_bytes = 0
    trainable_numel = 0
    trainable_bytes = 0

    for _, p in named_params:
        numel = p.numel()
        bytes_size = numel * p.element_size()
        total_numel += numel
        total_bytes += bytes_size
        if p.requires_grad:
            trainable_numel += numel
            trainable_bytes += bytes_size

    frozen_numel = total_numel - trainable_numel
    frozen_bytes = total_bytes - trainable_bytes
    return ParamStats(
        total_numel=total_numel,
        total_bytes=total_bytes,
        trainable_numel=trainable_numel,
        trainable_bytes=trainable_bytes,
        frozen_numel=frozen_numel,
        frozen_bytes=frozen_bytes,
    )


def _print_stats_block(label: str, stats: ParamStats) -> None:
    print(f"{label}:")
    print(f"  Total params     : {stats.total_numel:,}")
    print(f"  Trainable params : {stats.trainable_numel:,}")
    print(f"  Frozen params    : {stats.frozen_numel:,}")
    print(f"  Total memory     : {_human_bytes(stats.total_bytes)}")
    print(f"  Trainable memory : {_human_bytes(stats.trainable_bytes)}")
    print(f"  Frozen memory    : {_human_bytes(stats.frozen_bytes)}")


def _print_parameter_overview(model: torch.nn.Module) -> None:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]

    total_numel, total_bytes = _numel_and_bytes(model.parameters())
    trainable_numel, trainable_bytes = _numel_and_bytes(trainable_params)
    frozen_numel, frozen_bytes = _numel_and_bytes(frozen_params)

    print("=== Parameter Counts ===")
    print(f"Total params      : {total_numel:,}")
    print(f"Trainable params  : {trainable_numel:,}")
    print(f"Frozen params     : {frozen_numel:,}")
    print()

    print("=== Estimated Parameter Memory (current dtype) ===")
    print(f"Total memory      : {_human_bytes(total_bytes)}")
    print(f"Trainable memory  : {_human_bytes(trainable_bytes)}")
    print(f"Frozen memory     : {_human_bytes(frozen_bytes)}")
    print()


def _group_parameter_stats(
    model: torch.nn.Module,
) -> tuple[dict[str, ParamStats], list[tuple[str, torch.nn.Parameter]], ParamStats]:
    named_params = list(model.named_parameters())

    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "audio_codec": [],
        "vision_encoder": [],
        "text_encoder": [],
        "transformer": [],
        "other": [],
    }

    for name, param in named_params:
        if name.startswith("audio_codec."):
            grouped["audio_codec"].append((name, param))
        elif name.startswith("vision_encoder."):
            grouped["vision_encoder"].append((name, param))
        elif name.startswith("text_encoder."):
            grouped["text_encoder"].append((name, param))
        elif name.startswith("transformer."):
            grouped["transformer"].append((name, param))
        else:
            grouped["other"].append((name, param))

    grouped_stats = {group: _compute_stats(items) for group, items in grouped.items()}

    excluded_params = [
        (name, param)
        for name, param in named_params
        if not name.startswith(BACKBONE_PREFIXES)
    ]
    excluded_stats = _compute_stats(excluded_params)
    return grouped_stats, named_params, excluded_stats


def _print_grouped_parameter_overview(grouped_stats: dict[str, ParamStats], excluded_stats: ParamStats) -> None:
    print("=== Parameter Counts by Group ===")
    _print_stats_block("audio_codec", grouped_stats["audio_codec"])
    print()
    _print_stats_block("vision_encoder", grouped_stats["vision_encoder"])
    print()
    _print_stats_block("text_encoder", grouped_stats["text_encoder"])
    print()
    _print_stats_block("transformer", grouped_stats["transformer"])
    print()
    _print_stats_block("other", grouped_stats["other"])
    print()

    print("=== Whole-Model Totals Excluding audio_codec + vision_encoder + text_encoder ===")
    _print_stats_block("excluding_backbones", excluded_stats)
    print()


def _enforce_frozen_backbone_guard(
    named_params: list[tuple[str, torch.nn.Parameter]],
    enabled: bool,
) -> None:
    if not enabled:
        return

    forbidden_trainable = [
        name
        for name, param in named_params
        if name.startswith(BACKBONE_PREFIXES) and param.requires_grad
    ]

    if not forbidden_trainable:
        print("Backbone trainability guard: PASS (no trainable audio_codec/vision_encoder/text_encoder params).")
        return

    print("Backbone trainability guard: FAIL")
    print("Trainable parameters found under frozen backbone prefixes:")
    for name in forbidden_trainable:
        print(f"  - {name}")
    raise SystemExit(1)


def _print_top_parameter_tensors(model: torch.nn.Module, topk: int) -> None:
    named = list(model.named_parameters())
    named.sort(key=lambda x: x[1].numel(), reverse=True)

    print(f"=== Top {min(topk, len(named))} Parameter Tensors by Size ===")
    for name, tensor in named[:topk]:
        bytes_size = tensor.numel() * tensor.element_size()
        print(
            f"{name:<70} shape={tuple(tensor.shape)!s:<25} "
            f"numel={tensor.numel():>12,} bytes={_human_bytes(bytes_size):>10} "
            f"dtype={str(tensor.dtype):>12}"
        )
    print()


def _module_depth(name: str) -> int:
    if not name:
        return 0
    return name.count(".") + 1


def _print_structure_summary(model: torch.nn.Module, max_depth: int) -> None:
    print(f"=== Structure Summary (depth <= {max_depth}) ===")
    print(f"[root] {model.__class__.__name__}")

    for name, module in model.named_modules():
        if not name:
            continue
        depth = _module_depth(name)
        if depth > max_depth:
            continue

        direct_params = sum(p.numel() for p in module.parameters(recurse=False))
        indent = "  " * depth
        print(f"{indent}- {name}: {module.__class__.__name__} (direct_params={direct_params:,})")

    print()


def main() -> None:
    args = parse_args()

    model = SAMAudio.from_pretrained(args.checkpoint, strict=not args.non_strict)
    model.eval()

    print("Loaded model:", model.__class__.__name__)
    print("Checkpoint:", args.checkpoint)
    print()

    _print_parameter_overview(model)
    grouped_stats, named_params, excluded_stats = _group_parameter_stats(model)
    _print_grouped_parameter_overview(grouped_stats, excluded_stats)
    _enforce_frozen_backbone_guard(named_params, enabled=args.fail_if_backbone_trainable)
    _print_top_parameter_tensors(model, topk=max(1, args.topk))
    _print_structure_summary(model, max_depth=max(1, args.max_depth))


if __name__ == "__main__":
    main()
