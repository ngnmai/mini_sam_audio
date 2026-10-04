"""Recursively validate audio/video files and print paths that cannot be opened.

Examples:
    python utils/check_media_loadability.py --root assets/vgg
    python utils/check_media_loadability.py --root assets/vgg --verbose
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import wave
from pathlib import Path

import cv2

try:
    import torchaudio
except Exception:  # pragma: no cover - optional dependency at runtime
    torchaudio = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency at runtime
    sf = None


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".flv",
    ".wmv",
    ".mpeg",
    ".mpg",
}
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a directory recursively, try opening audio/video files, "
            "and print paths that fail to load."
        )
    )
    parser.add_argument("--root", type=Path, required=True, help="Directory to scan recursively.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print short failure reasons to stderr in addition to failed file paths.",
    )
    return parser.parse_args()


def iter_media_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS or suffix in AUDIO_EXTENSIONS:
            yield path


def can_open_video(path: Path) -> tuple[bool, str]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False, "cv2 failed to open"

        ok, _ = capture.read()
        if not ok:
            return False, "cv2 could not decode first frame"

        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        capture.release()


def can_open_audio(path: Path) -> tuple[bool, str]:
    if torchaudio is not None:
        try:
            torchaudio.info(str(path))
            return True, ""
        except Exception as exc:
            return False, f"torchaudio failed: {exc}"

    if sf is not None:
        try:
            sf.info(str(path))
            return True, ""
        except Exception as exc:
            return False, f"soundfile failed: {exc}"

    if path.suffix.lower() == ".wav":
        try:
            with contextlib.closing(wave.open(str(path), "rb")) as wf:
                _ = wf.getnframes()
            return True, ""
        except Exception as exc:
            return False, f"wave failed: {exc}"

    return False, "no audio backend available (install torchaudio or soundfile)"


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"Root is not a directory: {root}", file=sys.stderr)
        return 2

    failed_count = 0
    scanned_count = 0

    for file_path in iter_media_files(root):
        scanned_count += 1
        suffix = file_path.suffix.lower()

        if suffix in VIDEO_EXTENSIONS:
            ok, reason = can_open_video(file_path)
        else:
            ok, reason = can_open_audio(file_path)

        if not ok:
            failed_count += 1
            print(str(file_path))
            if args.verbose:
                print(f"  reason: {reason}", file=sys.stderr)

    if args.verbose:
        print(f"Scanned: {scanned_count}, failed: {failed_count}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
