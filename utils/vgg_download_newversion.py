"""Download 10-second VGGSound clips into train/test video and audio folders.

The CSV is expected to contain rows with the columns:
youtube_id, start_seconds, label, split

This script downloads a clipped segment for each available video, sorts the
result by split, and stores both the final video clip and a mono 16 kHz WAV
extraction of that clip. Unavailable videos are skipped.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV_PATH = Path("assets/vgg/vggsound.csv")
DEFAULT_OUTPUT_DIR = Path("vggsound_downloads")
CLIP_DURATION_SECONDS = 10.0
AUDIO_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ClipEntry:
    index: int
    youtube_id: str
    start_seconds: float
    label: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VGGSound clips into split-specific video and audio folders."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the VGGSound CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where train/test folders are created.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Number of successfully downloaded videos to collect. Defaults to 100.",
    )
    return parser.parse_args()


def load_entries(csv_path: Path) -> list[ClipEntry]:
    entries: list[ClipEntry] = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for index, row in enumerate(reader):
            if not row:
                continue

            if len(row) < 4:
                raise ValueError(
                    f"Expected 4 columns in {csv_path}, got {len(row)} on row {index + 1}."
                )

            youtube_id = row[0].strip()
            start_seconds = float(row[1])
            label = row[2].strip()
            split = row[3].strip().lower()

            if split not in {"train", "test"}:
                continue

            entries.append(
                ClipEntry(
                    index=index,
                    youtube_id=youtube_id,
                    start_seconds=start_seconds,
                    label=label,
                    split=split,
                )
            )

    return entries


def ensure_output_layout(output_dir: Path) -> None:
    for split in ("train", "test"):
        (output_dir / split / "video").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "audio").mkdir(parents=True, exist_ok=True)

    (output_dir / "tmp").mkdir(parents=True, exist_ok=True)


def build_stem(entry: ClipEntry) -> str:
    start_token = f"{entry.start_seconds:.3f}".replace(".", "p")
    return f"{entry.index:06d}_{entry.youtube_id}_{start_token}"


def cleanup_temp_files(temp_dir: Path, stem: str) -> None:
    for candidate in temp_dir.glob(f"{stem}*"):
        if candidate.is_file():
            candidate.unlink(missing_ok=True)


def yt_dlp_prefix() -> list[str] | None:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return None

    if shutil.which(sys.executable):
        return [sys.executable, "-m", "yt_dlp"]

    return None


def download_clip_to_temp(entry: ClipEntry, temp_dir: Path, temp_stem: str) -> Path | None:
    prefix = yt_dlp_prefix()
    if prefix is None:
        print("Skipping download because yt-dlp is not available on this system.")
        return None

    start = max(entry.start_seconds, 0.0)
    end = start + CLIP_DURATION_SECONDS
    url = f"https://www.youtube.com/watch?v={entry.youtube_id}"
    temp_template = str(temp_dir / f"{temp_stem}.%(ext)s")

    command = [
        *prefix,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--restrict-filenames",
        "--force-keyframes-at-cuts",
        "--download-sections",
        f"*{start:.3f}-{end:.3f}",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        temp_template,
        url,
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    for suffix in (".mp4", ".mkv", ".webm", ".mov"):
        candidate = temp_dir / f"{temp_stem}{suffix}"
        if candidate.exists():
            return candidate

    candidates = sorted(
        candidate
        for candidate in temp_dir.glob(f"{temp_stem}.*")
        if candidate.is_file() and candidate.suffix not in {".part", ".ytdl"}
    )
    return candidates[0] if candidates else None


def transcode_video_clip(source_video: Path, output_video: Path) -> bool:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-t",
        f"{CLIP_DURATION_SECONDS}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_video),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and output_video.exists()


def extract_audio_clip(source_video: Path, output_audio: Path) -> bool:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        str(output_audio),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and output_audio.exists()


def download_entry(entry: ClipEntry, output_dir: Path) -> bool:
    split_dir = output_dir / entry.split
    video_out = split_dir / "video" / f"{build_stem(entry)}.mp4"
    audio_out = split_dir / "audio" / f"{build_stem(entry)}.wav"

    if video_out.exists() and audio_out.exists():
        return True

    temp_dir = output_dir / "tmp"
    temp_stem = build_stem(entry)
    temp_video = download_clip_to_temp(entry, temp_dir, temp_stem)
    if temp_video is None:
        cleanup_temp_files(temp_dir, temp_stem)
        return False

    try:
        if not transcode_video_clip(temp_video, video_out):
            if video_out.exists():
                video_out.unlink(missing_ok=True)
            if audio_out.exists():
                audio_out.unlink(missing_ok=True)
            return False

        if not extract_audio_clip(video_out, audio_out):
            if video_out.exists():
                video_out.unlink(missing_ok=True)
            if audio_out.exists():
                audio_out.unlink(missing_ok=True)
            return False

        return True
    finally:
        cleanup_temp_files(temp_dir, temp_stem)


def main() -> int:
    args = parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH.")

    ensure_output_layout(args.output_dir)
    entries = load_entries(args.csv_path)

    target_count = 100 if args.target_count is None else args.target_count
    if target_count <= 0:
        raise ValueError("--target-count must be a positive integer.")

    train_count = 0
    test_count = 0
    skipped_count = 0
    downloaded_count = 0

    print(f"Loaded {len(entries)} rows from {args.csv_path}")
    print(f"Output directory: {args.output_dir}")
    print("Output structure:")
    print(f"  {args.output_dir}/train/video/*.mp4")
    print(f"  {args.output_dir}/train/audio/*.wav")
    print(f"  {args.output_dir}/test/video/*.mp4")
    print(f"  {args.output_dir}/test/audio/*.wav")
    print("Starting downloads...")

    for position, entry in enumerate(entries, start=1):
        if downloaded_count >= target_count:
            break

        success = download_entry(entry, args.output_dir)
        if success:
            downloaded_count += 1
            if entry.split == "train":
                train_count += 1
            else:
                test_count += 1
            print(
                f"[{position}/{len(entries)}] saved {entry.split} clip "
                f"for {entry.youtube_id} at {entry.start_seconds:.3f}s"
            )

            if downloaded_count % 10 == 0:
                print(
                    "CHECKPOINT: "
                    f"{downloaded_count} videos downloaded "
                    f"(train={train_count}, test={test_count}, skipped={skipped_count})"
                )
        else:
            skipped_count += 1
            print(
                f"[{position}/{len(entries)}] skipped unavailable clip "
                f"for {entry.youtube_id} at {entry.start_seconds:.3f}s"
            )

    if downloaded_count < target_count:
        print(
            f"Warning: only {downloaded_count} successful downloads were available "
            f"out of the requested target of {target_count}."
        )

    print("Done.")
    print(f"Train clips saved: {train_count}")
    print(f"Test clips saved: {test_count}")
    print(f"Skipped clips: {skipped_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())