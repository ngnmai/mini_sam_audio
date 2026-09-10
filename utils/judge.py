import argparse
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from sam_audio import SAMAudioJudgeModel, SAMAudioJudgeProcessor
from tqdm import tqdm


INPUT_SUFFIX = ".wav"
TARGET_SUFFIX = "_target.wav"


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


def _list_audio_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob(f"*{INPUT_SUFFIX}") if path.is_file())


def _list_target_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob(f"*{TARGET_SUFFIX}") if path.is_file())


def build_file_pairs(input_dir: Path, separated_dir: Path) -> tuple[list[Path], list[Path]]:
    input_files = _list_audio_files(input_dir)
    target_files = _list_target_files(separated_dir)

    if not input_files:
        raise ValueError(f"No {INPUT_SUFFIX} files found in input directory: {input_dir}")
    if not target_files:
        raise ValueError(f"No {TARGET_SUFFIX} files found in separated directory: {separated_dir}")

    target_map = {path.name[: -len(TARGET_SUFFIX)]: path for path in target_files}
    missing = [path.stem for path in input_files if path.stem not in target_map]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing *_target.wav pair for {len(missing)} input file(s): {preview}")

    paired_inputs = input_files
    paired_targets = [target_map[path.stem] for path in input_files]
    return paired_inputs, paired_targets


def select_chunk(
    input_files: list[Path],
    target_files: list[Path],
    chunk_index: int,
    num_chunks: int,
) -> tuple[list[Path], list[Path]]:
    if num_chunks <= 0:
        raise ValueError("--num-chunks must be a positive integer.")
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError("--chunk-index must be in the range [0, --num-chunks).")

    return input_files[chunk_index::num_chunks], target_files[chunk_index::num_chunks]


def iterate_batches(
    input_files: list[Path],
    target_files: list[Path],
    batch_size: int,
) -> list[tuple[list[Path], list[Path]]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    return [
        (input_files[index : index + batch_size], target_files[index : index + batch_size])
        for index in range(0, len(input_files), batch_size)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation with SAM Audio Judge.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with original input .wav files.")
    parser.add_argument(
        "--separated-dir",
        type=Path,
        required=True,
        help="Directory with separated files. Uses only *_target.wav files.",
    )
    parser.add_argument(
        "--checkpoint",
        default="facebook/sam-audio-judge",
        help="Hugging Face checkpoint/model id.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for single-process inference (e.g. cuda, cpu). Ignored when distributed with CUDA.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of matched files processed per inference step.",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Zero-based chunk index for job-array style partitioning.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Total number of chunks used to split the sample list across jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    try:
        input_dir = args.input_dir.expanduser().resolve()
        separated_dir = args.separated_dir.expanduser().resolve()

        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Invalid input directory: {input_dir}")
        if not separated_dir.exists() or not separated_dir.is_dir():
            raise ValueError(f"Invalid separated directory: {separated_dir}")

        input_files, target_files = build_file_pairs(input_dir, separated_dir)
        chunk_inputs, chunk_targets = select_chunk(
            input_files,
            target_files,
            args.chunk_index,
            args.num_chunks,
        )

        assigned_inputs = chunk_inputs[rank::world_size]
        assigned_targets = chunk_targets[rank::world_size]

        if not assigned_inputs:
            if rank == 0:
                print("No samples assigned to this run.")
            if dist.is_initialized():
                dist.barrier()
            return

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else args.device)
        model = SAMAudioJudgeModel.from_pretrained(args.checkpoint).to(device).eval()
        processor = SAMAudioJudgeProcessor.from_pretrained(args.checkpoint)

        if rank == 0:
            print(
                f"Found {len(input_files)} matched sample(s). "
                f"Chunk {args.chunk_index + 1}/{args.num_chunks} has {len(chunk_inputs)} sample(s)."
            )
            print(f"Using checkpoint: {args.checkpoint}")

        batches = iterate_batches(assigned_inputs, assigned_targets, args.batch_size)
        progress = tqdm(batches, desc=f"rank {rank}", disable=rank != 0)

        for batch_inputs, batch_targets in progress:
            descriptions = [""] * len(batch_inputs)
            inputs = processor(
                text=descriptions,
                input_audio=[str(path) for path in batch_inputs],
                separated_audio=[str(path) for path in batch_targets],
            ).to(device)

            with torch.inference_mode():
                result = model(**inputs)

            for index, (input_path, target_path) in enumerate(zip(batch_inputs, batch_targets)):
                print(f"\n[rank {rank}] Example")
                print(f"  Input: {input_path.name}")
                print(f"  Separated(target): {target_path.name}")
                print(f"  Overall: {result.overall[index].item():.3f}")
                print(f"  Recall: {result.recall[index].item():.3f}")
                print(f"  Precision: {result.precision[index].item():.3f}")
                print(f"  Faithfulness: {result.faithfulness[index].item():.3f}")

        if dist.is_initialized():
            dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
