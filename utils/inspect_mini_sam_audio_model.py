"""Inspect mini SAM-Audio model size, parameter counts, and structure.

Examples:
    python utils/inspect_mini_sam_audio_model.py
    python utils/inspect_mini_sam_audio_model.py --checkpoint /path/to/mini_sam_audio_final.pt
    python utils/inspect_mini_sam_audio_model.py --checkpoint /path/to/ckpt.pt --strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_sam_audio.compression.model_loading import load_student_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the mini SAM-Audio student model and report parameter counts, "
            "estimated model size, and a module structure summary."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, reports stats for a fresh model init.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True for checkpoint loading (default is non-strict).",
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

    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    model = load_student_model(
        checkpoint_path=args.checkpoint,
        strict=args.strict,
        map_location="cpu",
    )
    model.eval()

    print("Loaded model:", model.__class__.__name__)
    if args.checkpoint:
        print("Checkpoint:", args.checkpoint)
    else:
        print("Checkpoint: <none> (fresh init)")
    print()

    _print_parameter_overview(model)
    _print_top_parameter_tensors(model, topk=max(1, args.topk))
    _print_structure_summary(model, max_depth=max(1, args.max_depth))


if __name__ == "__main__":
    main()
