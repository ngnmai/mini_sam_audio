import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

VIDEO_SUFFIX = ".mp4"


def resolve_output_dir(video_dir: Path, mask_dir: Path, output_dir_name: str) -> Path:
    if video_dir.parent != mask_dir.parent:
        raise ValueError(
            "--output-dir-name requires --video-dir and --mask-dir to share the same parent directory, "
            f"got {video_dir.parent} and {mask_dir.parent}."
        )
    return video_dir.parent / output_dir_name


def list_video_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == VIDEO_SUFFIX)


def build_video_mask_pairs(video_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    video_files = list_video_files(video_dir)
    if not video_files:
        raise ValueError(f"No {VIDEO_SUFFIX} files found in video directory: {video_dir}")

    pairs = []
    missing = []
    for video_file in video_files:
        mask_file = mask_dir / video_file.name
        if mask_file.exists():
            pairs.append((video_file, mask_file))
        else:
            missing.append(video_file.name)

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing mask video for {len(missing)} file(s): {preview}")

    return pairs


def open_capture(video_file: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_file}")
    return capture


def apply_mask_to_video(video_file: Path, mask_file: Path, output_file: Path) -> None:
    video_capture = open_capture(video_file)
    mask_capture = open_capture(mask_file)

    try:
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask_width = int(mask_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        mask_height = int(mask_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != (mask_width, mask_height):
            raise ValueError(
                f"Resolution mismatch between {video_file.name} ({width}x{height}) "
                f"and its mask ({mask_width}x{mask_height})."
            )

        fps = float(video_capture.get(cv2.CAP_PROP_FPS) or 30.0)
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
            while True:
                video_ok, frame = video_capture.read()
                mask_ok, mask_frame = mask_capture.read()
                if not video_ok or not mask_ok:
                    break

                mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                mask_normalized = (mask_gray.astype(np.float32) / 255.0)[..., None]
                masked_frame = (frame.astype(np.float32) * mask_normalized).astype(np.uint8)
                writer.write(masked_frame)
        finally:
            writer.release()
    finally:
        video_capture.release()
        mask_capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply segmentation mask videos to their matching source videos.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory with source .mp4 videos.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Directory with mask .mp4 videos (same file names and resolution as the source videos).",
    )
    parser.add_argument(
        "--output-dir-name",
        required=True,
        help="Name of the folder (created next to --video-dir and --mask-dir, which must share a parent) used to store masked videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_dir = args.video_dir.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()

    if not video_dir.exists() or not video_dir.is_dir():
        raise ValueError(f"Invalid video directory: {video_dir}")
    if not mask_dir.exists() or not mask_dir.is_dir():
        raise ValueError(f"Invalid mask directory: {mask_dir}")

    output_dir = resolve_output_dir(video_dir, mask_dir, args.output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_video_mask_pairs(video_dir, mask_dir)
    print(f"Found {len(pairs)} matched video/mask pair(s). Writing masked videos to: {output_dir}")

    for video_file, mask_file in tqdm(pairs, desc="Applying masks"):
        output_file = output_dir / video_file.name
        apply_mask_to_video(video_file, mask_file, output_file)


if __name__ == "__main__":
    main()
