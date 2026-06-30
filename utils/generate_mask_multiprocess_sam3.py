# This script was modified from https://github.com/facebookresearch/sam-audio/blob/main/examples/visual_prompting.ipynb
import tempfile
from io import BytesIO
from pathlib import Path
import os
import subprocess
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.distributed as dist
from sam3.model_builder import build_sam3_video_predictor
from torchcodec.decoders import VideoDecoder
from tqdm import trange

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def parse_args():
    parser = ArgumentParser(description="Generate SAM3 masks for a VGGSound split.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing train/ and test/ folders.")
    parser.add_argument("--split", required=True, choices=("train", "test"), help="Split to process.")
    parser.add_argument(
        "--num-videos",
        default="all",
        help='Number of videos to process or "all" for the full split.',
    )
    parser.add_argument("--prompt", default="The person on the left", help="Text prompt passed to SAM3.")
    return parser.parse_args()


def setup_distributed():
    if not dist.is_available():
        return 0, 1, 0

    if "SLURM_PROCID" in os.environ:
        os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
        os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
        os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", os.environ["SLURM_PROCID"]))
        os.environ.setdefault("MASTER_ADDR", _get_slurm_master_addr())
        os.environ.setdefault("MASTER_PORT", _get_slurm_master_port())

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _get_slurm_master_addr():
    nodelist = os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    try:
        host = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).splitlines()[0]
        return host.strip()
    except Exception:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")


def _get_slurm_master_port():
    if "MASTER_PORT" in os.environ:
        return os.environ["MASTER_PORT"]

    job_id = os.environ.get("SLURM_JOB_ID")
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


def load_model(local_rank):
    predictor = build_sam3_video_predictor()
    if torch.cuda.is_available() and hasattr(predictor, "to"):
        predictor = predictor.to(f"cuda:{local_rank}")
    return predictor


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


def process_video(video_file, video_predictor, prompt):
    decoder = VideoDecoder(video_file)
    height, width = decoder.metadata.height, decoder.metadata.width
    fps = float(getattr(decoder.metadata, "average_fps", None) or getattr(decoder.metadata, "frame_rate", None) or 30.0)

    predictor = video_predictor
    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": video_file,
        }
    )
    session_id = response["session_id"]
    outputs = []
    for frame_index in trange(len(decoder), desc=video_file.name, leave=False):
        response = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": prompt,
            }
        )
        output = response["outputs"]
        mask = output["out_binary_masks"]
        if mask.shape[0] == 0:
            if frame_index > 0:
                mask = outputs[-1]
            else:
                mask = np.zeros((height, width), dtype=bool)
        else:
            mask = np.any(mask.astype(bool), axis=0)
        outputs.append(mask.astype(bool))

    return np.stack(outputs, axis=0), fps


def generate_masks(data_root, split, num_videos, video_predictor, rank, world_size, prompt):
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
            mask_frames, fps = process_video(video_file, video_predictor, prompt)
        decoder = VideoDecoder(video_file)
        output_file = mask_dir / f"{video_file.stem}.mp4"
        save_mask_video(mask_frames, output_file, fps, decoder.metadata.width, decoder.metadata.height)
        print(f"Rank {rank}: saved {output_file}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    video_predictor = load_model(local_rank)
    generate_masks(args.data_root, args.split, args.num_videos, video_predictor, rank, world_size, args.prompt)

    if dist.is_initialized():
        dist.destroy_process_group()





