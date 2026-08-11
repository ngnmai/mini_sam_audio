"""Run SAM-Audio visual-prompt inference over matched audio/video/mask files.

The script expects three directory roots containing matching file stems:

* ``audio_root`` with ``.wav`` files
* ``video_root`` with ``.mp4`` files
* ``mask_root`` with ``.mp4`` files

For each matched stem, the corresponding video and mask are combined through
``SAMAudioProcessor.mask_videos`` and passed to ``SAMAudio.separate`` for visual
prompting inference. Work can be distributed across SLURM ranks or generic
``torch.distributed`` ranks by setting the usual environment variables.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torchaudio
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM_AUDIO_ROOT = PROJECT_ROOT / "submodule" / "sam-audio"

if SAM_AUDIO_ROOT.exists():
    sys.path.insert(0, str(SAM_AUDIO_ROOT))

from sam_audio import SAMAudio, SAMAudioProcessor


DEFAULT_CHECKPOINT_PATH = "facebook/sam-audio-large-tv"
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 0
DEFAULT_RERANKING_CANDIDATES = 1


@dataclass(frozen=True)
class SamplePair:
    stem: str
    audio_path: Path
    video_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM-Audio inference with visual prompting over matched audio/video/mask files."
    )
    parser.add_argument("--audio-root", type=Path, required=True, help="Directory containing .wav audio files.")
    parser.add_argument("--mask-root", type=Path, required=True, help="Directory containing .mp4 mask files.")
    parser.add_argument("--video-root", type=Path, required=True, help="Directory containing .mp4 video files.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where outputs are written. Defaults to <video_root.parent>/output.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="SAM-Audio checkpoint path or Hugging Face repo id.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of matched files processed per step.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Reserved for future batching extensions.")
    parser.add_argument(
        "--reranking-candidates",
        type=int,
        default=DEFAULT_RERANKING_CANDIDATES,
        help="Number of candidates to generate for reranking inside SAM-Audio.",
    )
    parser.add_argument(
        "--predict-spans",
        action="store_true",
        default=False,
        help="Enable span prediction inside SAM-Audio before separation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even when target/residual files already exist.",
    )
    return parser.parse_args()


def read_environment() -> dict[str, str]:
    environment = {}
    for key, value in os.environ.items():
        environment[key] = value
    return environment


def get_slurm_master_addr(environment: dict[str, str]) -> str:
    nodelist = environment.get("SLURM_NODELIST")
    if not nodelist:
        return environment.get("MASTER_ADDR", "127.0.0.1")

    try:
        import subprocess

        host = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).splitlines()[0]
        return host.strip()
    except Exception:
        return environment.get("MASTER_ADDR", "127.0.0.1")


def get_slurm_master_port(environment: dict[str, str]) -> str:
    existing_port = environment.get("MASTER_PORT")
    if existing_port is not None:
        return existing_port

    job_id = environment.get("SLURM_JOB_ID")
    if job_id is None:
        return "29500"

    return str(10000 + (int(job_id) % 50000))


def setup_distributed() -> tuple[int, int, int]:
    if not dist.is_available():
        return 0, 1, 0

    environment = read_environment()
    rank = int(environment.get("SLURM_PROCID", environment.get("RANK", "0")))
    world_size = int(environment.get("SLURM_NTASKS", environment.get("WORLD_SIZE", "1")))
    local_rank = int(environment.get("SLURM_LOCALID", environment.get("LOCAL_RANK", str(rank))))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=f"tcp://{get_slurm_master_addr(environment)}:{get_slurm_master_port(environment)}",
            rank=rank,
            world_size=world_size,
        )

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def collect_files(root: Path, suffix: str) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    files = sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())
    indexed: dict[str, Path] = {}
    duplicates: list[Path] = []

    for path in files:
        stem = path.stem
        if stem in indexed:
            duplicates.append(path)
            continue
        indexed[stem] = path

    if duplicates:
        duplicate_list = "\n".join(str(path) for path in duplicates)
        raise ValueError(f"Duplicate stems detected under {root}:\n{duplicate_list}")

    return indexed


def resolve_samples(audio_root: Path, mask_root: Path, video_root: Path) -> list[SamplePair]:
    audio_files = collect_files(audio_root, ".wav")
    mask_files = collect_files(mask_root, ".mp4")
    video_files = collect_files(video_root, ".mp4")

    shared_stems = sorted(set(audio_files) & set(mask_files) & set(video_files))
    if not shared_stems:
        raise ValueError(
            "No shared file stems were found across audio_root, mask_root, and video_root. "
            "Expected matching audio/video/mask filenames."
        )

    missing_audio = sorted((set(video_files) | set(mask_files)) - set(audio_files))
    missing_video = sorted((set(audio_files) | set(mask_files)) - set(video_files))
    missing_mask = sorted((set(audio_files) | set(video_files)) - set(mask_files))

    if missing_audio or missing_video or missing_mask:
        print("Warning: some files were skipped because a matching trio was not found.")
        if missing_audio:
            print(f"  Missing audio for {len(missing_audio)} stem(s): {', '.join(missing_audio[:10])}")
        if missing_video:
            print(f"  Missing video for {len(missing_video)} stem(s): {', '.join(missing_video[:10])}")
        if missing_mask:
            print(f"  Missing mask for {len(missing_mask)} stem(s): {', '.join(missing_mask[:10])}")

    return [
        SamplePair(
            stem=stem,
            audio_path=audio_files[stem],
            video_path=video_files[stem],
            mask_path=mask_files[stem],
        )
        for stem in shared_stems
    ]


def build_output_root(video_root: Path, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root
    return video_root.parent / "output"


def ensure_output_layout(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[SAMAudio, SAMAudioProcessor]:
    model = SAMAudio.from_pretrained(checkpoint_path).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(checkpoint_path)
    return model, processor


def save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = waveform.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(str(path), tensor, sample_rate)


def process_batch(
    batch_samples: list[SamplePair],
    model: SAMAudio,
    processor: SAMAudioProcessor,
    device: torch.device,
    output_root: Path,
    reranking_candidates: int,
    predict_spans: bool,
    overwrite: bool,
) -> None:
    audio_paths = [str(sample.audio_path) for sample in batch_samples]
    video_paths = [str(sample.video_path) for sample in batch_samples]
    mask_paths = [str(sample.mask_path) for sample in batch_samples]
    descriptions = [""] * len(batch_samples)

    masked_videos = processor.mask_videos(video_paths, mask_paths)
    batch = processor(audios=audio_paths, descriptions=descriptions, masked_videos=masked_videos).to(device)

    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
        )

    for sample, target, residual in zip(batch_samples, result.target, result.residual, strict=False):
        target_path = output_root / f"{sample.stem}_target.wav"
        residual_path = output_root / f"{sample.stem}_residual.wav"

        if not overwrite and target_path.exists() and residual_path.exists():
            continue

        save_waveform(target_path, target, processor.audio_sampling_rate)
        save_waveform(residual_path, residual, processor.audio_sampling_rate)


def iterate_batches(samples: list[SamplePair], batch_size: int) -> list[list[SamplePair]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    return [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    try:
        output_root = build_output_root(args.video_root, args.output_root)
        ensure_output_layout(output_root)

        if dist.is_initialized():
            dist.barrier()

        samples = resolve_samples(args.audio_root, args.mask_root, args.video_root)
        assigned_samples = samples[rank::world_size]

        if not assigned_samples:
            if rank == 0:
                print("No samples assigned to this run.")
            if dist.is_initialized():
                dist.barrier()
            return

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model, processor = load_model(args.checkpoint_path, device)

        if rank == 0:
            print(f"Found {len(samples)} matched sample(s). Writing outputs to: {output_root}")
            print(f"Using checkpoint: {args.checkpoint_path}")

        batches = iterate_batches(assigned_samples, args.batch_size)
        progress = tqdm(batches, desc=f"rank {rank}", disable=rank != 0)

        for batch_samples in progress:
            process_batch(
                batch_samples=batch_samples,
                model=model,
                processor=processor,
                device=device,
                output_root=output_root,
                reranking_candidates=args.reranking_candidates,
                predict_spans=args.predict_spans,
                overwrite=args.overwrite,
            )

        if dist.is_initialized():
            dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()