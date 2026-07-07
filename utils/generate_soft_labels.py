from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from sam_audio import SAMAudio, SAMAudioProcessor


AUDIO_EXTENSIONS = (".wav",)
VIDEO_EXTENSIONS = (".mp4",)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Folder containing audio/, video/, and mask/ subfolders.")
    parser.add_argument("--model", default="facebook/sam-audio-large", help="SAM-Audio checkpoint to use.")
    parser.add_argument(
        "--num-videos",
        default="all",
        help="Number of matched videos to process, or 'all' to process every file.",
    )
    return parser.parse_args()


def parse_num_videos(value: str) -> int | None:
    if value == "all":
        return None

    try:
        num_videos = int(value)
    except ValueError as exc:
        raise ValueError("--num-videos must be an integer or 'all'") from exc

    if num_videos < 1:
        raise ValueError("--num-videos must be at least 1 or 'all'")

    return num_videos


def resolve_files(folder: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    files = {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    }
    if not files:
        raise ValueError(f"No supported files found in {folder}")
    return files


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_soft_labels(data_root: Path, model_name: str, num_videos: int | None = None):
    audio_dir = data_root / "audio"
    video_dir = data_root / "video"
    mask_dir = data_root / "mask"
    output_dir = data_root / "soft_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = resolve_files(audio_dir, AUDIO_EXTENSIONS)
    video_files = resolve_files(video_dir, VIDEO_EXTENSIONS)
    mask_files = resolve_files(mask_dir, VIDEO_EXTENSIONS)

    common_stems = sorted(set(audio_files) & set(video_files) & set(mask_files))
    if not common_stems:
        raise ValueError("No matching filenames found across audio/, video/, and mask/.")

    missing_audio = sorted(set(video_files) | set(mask_files) - set(audio_files))
    missing_video = sorted(set(audio_files) | set(mask_files) - set(video_files))
    missing_mask = sorted(set(audio_files) | set(video_files) - set(mask_files))
    if missing_audio or missing_video or missing_mask:
        missing_parts = []
        if missing_audio:
            missing_parts.append(f"audio: {', '.join(missing_audio)}")
        if missing_video:
            missing_parts.append(f"video: {', '.join(missing_video)}")
        if missing_mask:
            missing_parts.append(f"mask: {', '.join(missing_mask)}")
        raise ValueError("Missing matching files across subfolders (" + "; ".join(missing_parts) + ")")

    device = get_device()
    model = SAMAudio.from_pretrained(model_name).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(model_name)

    stems_to_process = common_stems if num_videos is None else common_stems[:num_videos]

    print(f"Processing {len(stems_to_process)} files from {data_root}")
    for stem in tqdm(stems_to_process, desc="Generating soft labels"):
        audio_file = audio_files[stem]
        video_file = video_files[stem]
        mask_file = mask_files[stem]

        masked_videos = processor.mask_videos([str(video_file)], [str(mask_file)])
        batch = processor(
            audios=[str(audio_file)],
            descriptions=[""],
            masked_videos=masked_videos,
        ).to(device)

        with torch.inference_mode():
            result = model.separate(batch, predict_spans=False, reranking_candidates=1)

        output_file = output_dir / f"{stem}.wav"
        torchaudio.save(output_file.as_posix(), result.target.cpu(), processor.audio_sampling_rate)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    args = parse_args()
    generate_soft_labels(Path(args.data_root), args.model, parse_num_videos(args.num_videos))
