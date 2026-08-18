import subprocess
import gc
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.distributed as dist
from sam3.model_builder import build_sam3_video_predictor
from tqdm import tqdm

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
    parser = ArgumentParser(description="Generate SAM3 masks for a VGGSound split.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing train/ and test/ folders.")
    parser.add_argument("--split", required=True, choices=("train", "test"), help="Split to process.")
    parser.add_argument(
        "--num-videos",
        default="all",
        help='Number of videos to process or "all" for the full split.',
    )
    parser.add_argument(
        "--prompt",
        default="",
        help='Text prompt passed to SAM3. Defaults to empty string ("") for visual prompting style.',
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Zero-based chunk index for job-array style partitioning of the video list.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Total number of chunks used to split the video list across batch jobs.",
    )
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

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
        )

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

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


def select_chunk(video_files, chunk_index, num_chunks):
    if num_chunks <= 0:
        raise ValueError("--num-chunks must be a positive integer.")
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError("--chunk-index must be in the range [0, --num-chunks).")

    return video_files[chunk_index::num_chunks]


def load_model(local_rank):
    predictor = build_sam3_video_predictor()
    if torch.cuda.is_available() and hasattr(predictor, "to"):
        predictor = predictor.to(f"cuda:{local_rank}")
    return predictor


def open_video_capture(video_file):
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_file}")
    return capture


def get_video_metadata(video_file):
    capture = open_video_capture(video_file)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return capture, width, height, fps, frame_count
    except Exception:
        capture.release()
        raise


def iter_video_frames(capture):
    while True:
        success, frame = capture.read()
        if not success:
            break
        yield frame


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


def process_video(video_file, predictor, prompt):
    capture, width, height, fps, frame_count = get_video_metadata(video_file)
    video_path = str(video_file)
    session_id = None
    if prompt is None:
        prompt = ""

    outputs = [None] * frame_count

    def to_binary_mask(output_masks):
        if output_masks.shape[0] == 0:
            return np.zeros((height, width), dtype=bool)
        return np.any(output_masks.astype(bool), axis=0)

    try:
        response = predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": video_path,
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            }
        )
        session_id = response["session_id"]
        # SAM3 requires at least one prompt; use empty text to match SAM-Audio visual prompting style.
        add_prompt_response = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            }
        )
        prompted_frame_index = int(add_prompt_response["frame_index"])
        outputs[prompted_frame_index] = to_binary_mask(add_prompt_response["outputs"]["out_binary_masks"])

        stream = predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
            }
        )
        for response in tqdm(stream, total=frame_count, desc=video_file.name, leave=False):
            frame_index = int(response["frame_index"])
            outputs[frame_index] = to_binary_mask(response["outputs"]["out_binary_masks"])

        for frame_index in range(frame_count):
            if outputs[frame_index] is None:
                if frame_index > 0 and outputs[frame_index - 1] is not None:
                    outputs[frame_index] = outputs[frame_index - 1]
                else:
                    outputs[frame_index] = np.zeros((height, width), dtype=bool)
    finally:
        capture.release()
        if session_id is not None:
            try:
                predictor.handle_request(
                    request={
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.stack(outputs, axis=0), fps, width, height


def generate_masks(data_root, split, num_videos, rank, world_size, local_rank, prompt, chunk_index, num_chunks):
    split_dir = Path(data_root) / split
    video_dir = split_dir / "video"
    mask_dir = split_dir / "mask_sam3"
    if rank == 0:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    video_files = resolve_video_files(video_dir, num_videos)
    chunked_videos = select_chunk(video_files, chunk_index, num_chunks)
    assigned_videos = chunked_videos[rank::world_size]

    if not assigned_videos:
        if rank == 0:
            print("No videos assigned to this run.")
        if dist.is_initialized():
            dist.barrier()
        return

    predictor = load_model(local_rank=local_rank)

    if rank == 0:
        print(
            f"Found {len(video_files)} video(s). "
            f"Chunk {chunk_index + 1}/{num_chunks} has {len(chunked_videos)} video(s)."
        )
        print(f"Writing masks to: {mask_dir}")

    print(f"Rank {rank}: processing {len(assigned_videos)} of {len(chunked_videos)} videos from {video_dir}")
    try:
        for video_file in assigned_videos:
            with torch.inference_mode():
                mask_frames, fps, width, height = process_video(video_file, predictor, prompt)
            output_file = mask_dir / f"{video_file.stem}.mp4"
            save_mask_video(mask_frames, output_file, fps, width, height)
            del mask_frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Rank {rank}: saved {output_file}")
    finally:
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    try:
        generate_masks(
            args.data_root,
            args.split,
            args.num_videos,
            rank,
            world_size,
            local_rank,
            args.prompt,
            args.chunk_index,
            args.num_chunks,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()





