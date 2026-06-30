import subprocess
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torchcodec.decoders import VideoDecoder
from tqdm import trange

from ultralytics import YOLO


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def read_environment() -> dict[str, str]:
    environment = {}
    output = subprocess.check_output(["env"], text=True)
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        environment[key] = value
    return environment


def parse_args():
    parser = ArgumentParser(description="Generate YOLO11m-seg masks for a VGGSound split.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing train/ and test/ folders.")
    parser.add_argument("--split", required=True, choices=("train", "test"), help="Split to process.")
    parser.add_argument(
        "--num-videos",
        default="all",
        help='Number of videos to process or "all" for the full split.',
    )
    parser.add_argument("--model", default="yolo11m-seg.pt", help="Ultralytics segmentation model path.")
    return parser.parse_args()


def setup_distributed():
    if not dist.is_available():
        return 0, 1, 0

    environment = read_environment()
    rank = int(environment.get("SLURM_PROCID", environment.get("RANK", "0")))
    world_size = int(environment.get("SLURM_NTASKS", environment.get("WORLD_SIZE", "1")))
    local_rank = int(environment.get("SLURM_LOCALID", environment.get("LOCAL_RANK", str(rank))))
    master_addr = _get_slurm_master_addr(environment)
    master_port = _get_slurm_master_port(environment)

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _get_env(environment: dict[str, str], name: str, default: str | None = None) -> str | None:
    return environment.get(name, default)


def _get_slurm_master_addr(environment: dict[str, str]):
    nodelist = _get_env(environment, "SLURM_NODELIST")
    if not nodelist:
        return _get_env(environment, "MASTER_ADDR", "127.0.0.1")

    try:
        host = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).splitlines()[0]
        return host.strip()
    except Exception:
        return _get_env(environment, "MASTER_ADDR", "127.0.0.1")


def _get_slurm_master_port(environment: dict[str, str]):
    existing_port = _get_env(environment, "MASTER_PORT")
    if existing_port is not None:
        return existing_port

    job_id = _get_env(environment, "SLURM_JOB_ID")
    if job_id is None:
        return "29500"

    return str(10000 + (int(job_id) % 50000))


def resolve_video_files(video_dir, num_videos):
    files = sorted(
        path
        for path in Path(video_dir).iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No video files found in: {video_dir}")

    if num_videos != "all":
        limit = int(num_videos)
        if limit <= 0:
            raise ValueError("--num-videos must be a positive integer or 'all'.")
        files = files[:limit]

    return files


def load_model(model_path, local_rank):
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to(f"cuda:{local_rank}")
    return model


def save_mask_video(mask_frames, output_file, fps, width, height):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_file}")

    try:
        for mask in mask_frames:
            frame = (mask.astype(np.uint8) * 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()


def process_video(video_file, model, device):
    decoder = VideoDecoder(video_file)
    height, width = decoder.metadata.height, decoder.metadata.width
    fps = float(getattr(decoder.metadata, "average_fps", None) or getattr(decoder.metadata, "frame_rate", None) or 30.0)

    outputs = []
    for frame_index in trange(len(decoder), desc=video_file.name, leave=False):
        frame = decoder[frame_index]
        prediction = model.predict(frame, verbose=False, device=device)[0]
        if prediction.masks is None or prediction.masks.data is None:
            mask = np.zeros((height, width), dtype=bool)
        else:
            mask_tensor = prediction.masks.data.detach().cpu() > 0.5
            mask = mask_tensor.any(dim=0).numpy()
        outputs.append(mask.astype(bool))

    return np.stack(outputs, axis=0), fps


def generate_masks(data_root, split, num_videos, model, rank, world_size, device):
    split_dir = Path(data_root) / split
    video_dir = split_dir / "video"
    mask_dir = split_dir / "mask"
    if rank == 0:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    video_files = resolve_video_files(video_dir, num_videos)
    assigned_videos = video_files[rank::world_size]

    print(f"Rank {rank}: processing {len(assigned_videos)} of {len(video_files)} videos from {video_dir}")
    for video_file in assigned_videos:
        with torch.inference_mode():
            mask_frames, fps = process_video(video_file, model, device)
        decoder = VideoDecoder(video_file)
        output_file = mask_dir / f"{video_file.stem}.mp4"
        save_mask_video(mask_frames, output_file, fps, decoder.metadata.width, decoder.metadata.height)
        print(f"Rank {rank}: saved {output_file}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, local_rank)
    generate_masks(args.data_root, args.split, args.num_videos, model, rank, world_size, device)

    if dist.is_initialized():
        dist.destroy_process_group()





