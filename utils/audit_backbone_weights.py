"""Audit that the frozen audio_codec/vision_encoder backbones equal the pretrained reference tensors.

The references are read independently of the model constructors (``dacvae.DACVAE.load`` plus the
raw DAC VAE file, and the raw PE file), so a bug in the loading path cannot hide itself.
Run on Roihu: it needs the RAM of the full PE CLIP plus its reference state dict.

Examples:
    python utils/audit_backbone_weights.py
    python utils/audit_backbone_weights.py --init-mode checkpoint --checkpoint-path /path/to/ckpt.pt
    python utils/audit_backbone_weights.py --checkpoint /path/to/last.ckpt --json-out audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_sam_audio.compression.model_loading import bootstrap_model_and_processor
from mini_sam_audio.model.config import MiniSAMAudioConfig


AUDIO_PREFIX = "audio_codec."
VISION_PREFIX = "vision_encoder."
BACKBONE_PREFIXES = (AUDIO_PREFIX, VISION_PREFIX)
# Wrapper in mini_sam_audio/model/codec.py keeps only these groups of dacvae.DACVAE.
DAC_WRAPPER_GROUPS = ("encoder.", "quantizer.", "decoder.")
# MiniSAMAudio stores the PE CLIP as vision_encoder.model.
PE_MODEL_PREFIX = VISION_PREFIX + "model."
LIST_LIMIT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that audio_codec and vision_encoder tensors equal the pretrained DAC VAE and PE weights."
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        choices=["scratch", "checkpoint"],
        default="scratch",
        help="Bootstrap mode used to build the model to audit (same as the training scripts).",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Checkpoint for --init-mode checkpoint.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Local MiniSAMAudioConfig JSON. Defaults to config.json next to the checkpoint, else built-in defaults.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Audit this .pt state dict or Lightning .ckpt directly instead of building a model.",
    )
    parser.add_argument("--atol", type=float, default=0.0, help="Max allowed |a - b| per tensor (0 = exact).")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a JSON report.")
    args = parser.parse_args()
    if args.checkpoint is None and args.init_mode == "checkpoint" and args.checkpoint_path is None:
        parser.error("--checkpoint-path is required when --init-mode=checkpoint.")
    return args


def _torch_load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Lightning checkpoints can hold non-tensor objects; these are our own files.
        return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_config(args: argparse.Namespace) -> tuple[MiniSAMAudioConfig, str]:
    if args.model_config is not None:
        source = args.model_config
    else:
        ckpt = args.checkpoint if args.checkpoint is not None else (
            Path(args.checkpoint_path) if args.init_mode == "checkpoint" else None
        )
        source = ckpt.parent / "config.json" if ckpt is not None else None
        if source is not None and not source.exists():
            source = None
    if source is None:
        return MiniSAMAudioConfig(), "<built-in defaults>"
    with Path(source).open() as fin:
        return MiniSAMAudioConfig(**json.load(fin)), str(source)


def _checkpoint_backbone_state(path: Path) -> dict[str, torch.Tensor]:
    raw = _torch_load(str(path))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a dict-like checkpoint.")
    if isinstance(raw.get("state_dict"), dict):
        state = raw["state_dict"]
    elif isinstance(raw.get("model"), dict):
        state = raw["model"]
    else:
        state = raw
    # Lightning saves LightningModule.model.<key>; the final .pt has no prefix.
    stripped = {
        (key[len("model."):] if key.startswith("model.") else key): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    return {k: v for k, v in stripped.items() if k.startswith(BACKBONE_PREFIXES)}


def _load_dac_reference(pretrained_path: str) -> tuple[dict[str, torch.Tensor], list[str], dict]:
    import dacvae

    loaded = dacvae.DACVAE.load(pretrained_path)
    loaded_state = {k: v.detach() for k, v in loaded.state_dict().items()}
    del loaded
    reference = {
        AUDIO_PREFIX + k: v for k, v in loaded_state.items() if k.startswith(DAC_WRAPPER_GROUPS)
    }
    outside = sorted(k for k in loaded_state if not k.startswith(DAC_WRAPPER_GROUPS))

    # DACVAE.load uses strict=False, so re-read the raw file to prove nothing was silently skipped.
    raw_path = pretrained_path
    if not os.path.exists(raw_path) and raw_path.startswith("facebook/"):
        from huggingface_hub import hf_hub_download

        raw_path = hf_hub_download(repo_id=raw_path, filename="weights.pth")
    raw_state = _torch_load(raw_path)["state_dict"]
    check = _compare(loaded_state, raw_state, 0.0)
    del raw_state
    return reference, outside, {
        "raw_file": raw_path,
        "missing": check["missing"],
        "unexpected": check["unexpected"],
        "mismatched": [m["key"] for m in check["mismatches"]],
        "passed": check["passed"],
    }


def _load_pe_reference(pe_name: str) -> tuple[dict[str, torch.Tensor], str]:
    from core.vision_encoder.config import fetch_pe_checkpoint

    path = fetch_pe_checkpoint(pe_name, None)
    state = torch.load(path, weights_only=True)
    # Same normalisation as TextTransformer.load_ckpt (inherited by CLIP) in core/vision_encoder/pe.py.
    if "state_dict" in state:
        state = state["state_dict"]
    elif "weights" in state:
        state = state["weights"]
    return {PE_MODEL_PREFIX + k.replace("module.", ""): v for k, v in state.items()}, path


def _compare(
    subject: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], atol: float
) -> dict:
    missing = sorted(k for k in reference if k not in subject)
    unexpected = sorted(k for k in subject if k not in reference)
    mismatches = []
    verified_numel = compared = 0
    union_numel = sum(subject[k].numel() for k in subject) + sum(reference[k].numel() for k in missing)
    for key in sorted(set(subject) & set(reference)):
        a, b = subject[key], reference[key]
        compared += 1
        if a.shape != b.shape:
            mismatches.append({"key": key, "reason": f"shape {tuple(a.shape)} != {tuple(b.shape)}"})
        elif a.dtype != b.dtype:
            mismatches.append({"key": key, "reason": f"dtype {a.dtype} != {b.dtype}"})
        else:
            diff = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
            if diff <= atol:
                verified_numel += a.numel()
            else:
                mismatches.append({"key": key, "reason": f"max_abs_diff {diff:.6g} > atol {atol}", "max_abs_diff": diff})
    percent = 100.0 * verified_numel / union_numel if union_numel else 0.0
    return {
        "missing": missing,
        "unexpected": unexpected,
        "mismatches": mismatches,
        "tensors_compared": compared,
        "verified_numel": verified_numel,
        "total_numel": union_numel,
        "percent_verified": percent,
        "passed": not (missing or unexpected or mismatches) and verified_numel == union_numel,
    }


def _non_finite(subject: dict[str, torch.Tensor]) -> list[str]:
    return sorted(
        k for k, v in subject.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())
    )


def _non_persistent_buffers(model: torch.nn.Module) -> list[str]:
    names = []
    for module_name, module in model.named_modules():
        for buffer_name in getattr(module, "_non_persistent_buffers_set", ()):
            full = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if full.startswith(BACKBONE_PREFIXES):
                names.append(full)
    return sorted(names)


def _freeze_report(model: torch.nn.Module) -> dict:
    trainable = [n for n, p in model.named_parameters() if n.startswith(BACKBONE_PREFIXES) and p.requires_grad]
    return {
        "trainable_backbone_params": trainable,
        "audio_codec_training_flag": model.audio_codec.training,
        "vision_encoder_training_flag": model.vision_encoder.training,
        "passed": not trainable,
    }


def _first(items: list, limit: int = LIST_LIMIT) -> list:
    return items[:limit]


def _print_backbone(name: str, result: dict, non_finite: list[str]) -> None:
    print(f"=== {name} ===")
    print(f"missing keys    : {len(result['missing'])} {_first(result['missing'])}")
    print(f"unexpected keys : {len(result['unexpected'])} {_first(result['unexpected'])}")
    print(f"tensors compared: {result['tensors_compared']}  mismatched: {len(result['mismatches'])}")
    for item in _first(result["mismatches"]):
        print(f"  mismatch {item['key']}: {item['reason']}")
    print(
        f"verified params : {result['verified_numel']:,} / {result['total_numel']:,} "
        f"({result['percent_verified']:.4f}%)"
    )
    print(f"non-finite tensors: {len(non_finite)} {_first(non_finite)}")


def main() -> None:
    args = parse_args()
    config, config_source = _resolve_config(args)
    print(f"config: {config_source}")

    model: Optional[torch.nn.Module] = None
    if args.checkpoint is not None:
        subject_all = _checkpoint_backbone_state(args.checkpoint)
        print(f"mode: checkpoint file {args.checkpoint} ({len(subject_all)} backbone tensors)")
    else:
        model, _, freeze_backbones = bootstrap_model_and_processor(
            init_mode=args.init_mode,
            checkpoint_path=args.checkpoint_path,
            model_config_path=args.model_config,
        )
        if freeze_backbones:
            model.freeze_inference_backbones()
        subject_all = {
            k: v.detach() for k, v in model.state_dict().items() if k.startswith(BACKBONE_PREFIXES)
        }
        print(f"mode: built model (init-mode={args.init_mode})")

    dac_reference, dac_outside, dac_selfcheck = _load_dac_reference(config.audio_codec.pretrained_path)
    pe_reference, pe_path = _load_pe_reference(config.vision_encoder.name)
    print(f"DAC VAE reference: {config.audio_codec.pretrained_path} (raw file {dac_selfcheck['raw_file']})")
    print(f"PE reference     : {pe_path}")

    report: dict = {"config": config_source, "atol": args.atol, "backbones": {}}
    all_passed = True
    for name, prefix, reference in (
        ("audio_codec", AUDIO_PREFIX, dac_reference),
        ("vision_encoder", VISION_PREFIX, pe_reference),
    ):
        subject = {k: v for k, v in subject_all.items() if k.startswith(prefix)}
        result = _compare(subject, reference, args.atol)
        non_finite = _non_finite(subject)
        _print_backbone(name, result, non_finite)
        passed = result["passed"] and not non_finite
        all_passed = all_passed and passed
        report["backbones"][name] = {**result, "non_finite": non_finite, "passed": passed}

    print("=== DAC VAE reference self-check (DACVAE.load vs raw weights.pth) ===")
    print(
        f"missing: {len(dac_selfcheck['missing'])} {_first(dac_selfcheck['missing'])}  "
        f"unexpected: {len(dac_selfcheck['unexpected'])}  mismatched: {len(dac_selfcheck['mismatched'])} "
        f"-> {'PASS' if dac_selfcheck['passed'] else 'FAIL'}"
    )
    all_passed = all_passed and dac_selfcheck["passed"]
    report["dac_reference_selfcheck"] = dac_selfcheck

    print("=== Not verified ===")
    print(f"DAC VAE reference keys not part of the wrapper: {len(dac_outside)} {_first(dac_outside)}")
    report["not_part_of_wrapper"] = dac_outside
    if model is not None:
        buffers = _non_persistent_buffers(model)
        print(f"non-persistent backbone buffers (not in state_dict): {len(buffers)} {_first(buffers)}")
        report["non_persistent_buffers"] = buffers
        freeze = _freeze_report(model)
        print(
            f"freeze: trainable backbone params={len(freeze['trainable_backbone_params'])} "
            f"{_first(freeze['trainable_backbone_params'])} "
            f"(training flags: audio_codec={freeze['audio_codec_training_flag']}, "
            f"vision_encoder={freeze['vision_encoder_training_flag']}) -> {'PASS' if freeze['passed'] else 'FAIL'}"
        )
        all_passed = all_passed and freeze["passed"]
        report["freeze"] = freeze
    else:
        print("non-persistent buffers and freeze state: skipped in --checkpoint mode (no model built)")

    report["passed"] = all_passed
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"report written to {args.json_out}")

    print("PASS: backbones equal the pretrained reference." if all_passed else "FAIL: see mismatches above.")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
