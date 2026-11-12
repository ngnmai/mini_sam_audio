"""Functional sanity check for the pretrained DAC VAE and PE backbones of the mini SAM-Audio model.

Builds the model in scratch mode (pretrained backbones) and checks behaviour, not tensors
(use utils/audit_backbone_weights.py for tensor equality). Exits non-zero on any failure.

Examples:
    python utils/verify_pretrained_backbones.py --wav /path/to/mono.wav
    python utils/verify_pretrained_backbones.py --wav /path/to/mono.wav --video /path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_sam_audio.compression.model_loading import bootstrap_model_and_processor
from mini_sam_audio.model.loss import compute_si_sdr_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that the pretrained DAC VAE and PE backbones behave like trained models."
    )
    parser.add_argument("--wav", type=Path, required=True, help="Audio file to round-trip through the codec.")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional video file; two frames are encoded. Without it the PE frame-difference check is skipped.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional local MiniSAMAudioConfig JSON; defaults to the built-in config.",
    )
    parser.add_argument("--max-seconds", type=float, default=5.0, help="Audio length used for the round trip.")
    parser.add_argument(
        "--min-si-sdr-db",
        type=float,
        default=10.0,
        help="Minimum codec round-trip SI-SDR; a random codec scores near or below 0 dB.",
    )
    parser.add_argument(
        "--max-frame-cosine",
        type=float,
        default=0.999,
        help="Two different frames must have cosine similarity below this value.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_mono_wav(path: Path, sample_rate: int, max_seconds: float) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav[:, : int(max_seconds * sample_rate)]


def _read_two_frames(path: Path) -> torch.Tensor:
    """Return two RGB uint8 frames (10% and 90% into the clip) as [2, C, H, W]."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames < 2:
            raise RuntimeError(f"{path} reports fewer than 2 frames.")
        frames = []
        for index in (int(0.1 * (n_frames - 1)), int(0.9 * (n_frames - 1))):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index} from {path}.")
            frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
    finally:
        capture.release()
    if frames[0].shape != frames[1].shape:
        raise RuntimeError(f"Frames from {path} differ in shape.")
    return torch.stack(frames)


def check_codec(model, wav_path: Path, device: torch.device, max_seconds: float, min_si_sdr_db: float) -> bool:
    codec = model.audio_codec
    wav = _load_mono_wav(wav_path, codec.sample_rate, max_seconds).unsqueeze(0).to(device)  # [1, 1, T]
    length = wav.size(-1)
    with torch.no_grad():
        latent = codec(wav)
        recon = codec.decode(latent)[..., :length]
    finite = bool(torch.isfinite(latent).all() and torch.isfinite(recon).all())
    si_sdr = compute_si_sdr_db(recon.reshape(1, -1), wav.reshape(1, -1)).item()
    passed = finite and si_sdr >= min_si_sdr_db
    print(
        f"[codec] samples={length} latent_shape={tuple(latent.shape)} finite={finite} "
        f"round-trip SI-SDR={si_sdr:.2f} dB (min {min_si_sdr_db:.2f}) -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def check_vision(model, video_path: Path | None, device: torch.device, max_frame_cosine: float) -> bool:
    if video_path is None:
        print("[vision] SKIPPED: --video not given, so finite/unit-norm/frame-difference checks did not run.")
        return True
    encoder = model.vision_encoder
    frames = _read_two_frames(video_path).to(device)  # [2, C, H, W]
    with torch.no_grad():
        feats = encoder([frames])[0].float()  # [2, D]
    finite = bool(torch.isfinite(feats).all())
    passed = finite
    norms = feats.norm(dim=-1)
    if encoder.normalize_feature:
        unit = bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-3))
        passed = passed and unit
        print(f"[vision] feature norms={[round(n, 5) for n in norms.tolist()]} unit_norm={unit}")
    cosine = torch.nn.functional.cosine_similarity(feats[0], feats[1], dim=0).item()
    different = cosine < max_frame_cosine
    passed = passed and different
    print(
        f"[vision] feature_shape={tuple(feats.shape)} finite={finite} frame cosine={cosine:.4f} "
        f"(max {max_frame_cosine}) -> {'PASS' if passed else 'FAIL'}"
    )
    return passed


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _, _ = bootstrap_model_and_processor(
        init_mode="scratch", model_config_path=args.model_config
    )
    model = model.to(device).eval()

    results = {
        "codec": check_codec(model, args.wav, device, args.max_seconds, args.min_si_sdr_db),
        "vision": check_vision(model, args.video, device, args.max_frame_cosine),
    }
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"FAIL: {', '.join(failed)}")
        raise SystemExit(1)
    print("PASS: pretrained backbones behave as expected.")


if __name__ == "__main__":
    main()
