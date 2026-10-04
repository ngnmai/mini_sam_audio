"""Dataset and collate utilities for training mini_sam_audio on soft-label supervision.

Expects a ``data_root`` directory laid out the same way as
``utils/generate_soft_labels.py`` produces/consumes it::

    data_root/audio/<stem>.wav
    data_root/video/<stem>.mp4
    data_root/mask/<stem>.mp4
    data_root/soft_labels/<stem>.wav
    data_root/residual_labels/<stem>.wav  (optional, per-stem)

``soft_labels/`` is mandatory here (it is the supervision signal), unlike
``utils/inference_sam_audio.py`` which only requires audio/video/mask.
``residual_labels/`` is optional: when a per-stem residual file is absent the
residual target is derived as ``mixture - soft_label``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from sam_audio import Batch, SAMAudioProcessor


AUDIO_EXTENSIONS = (".wav",)
VIDEO_EXTENSIONS = (".mp4",)


@dataclass(frozen=True)
class SoftLabelSample:
    stem: str
    audio_path: Path
    video_path: Path
    mask_path: Path
    soft_label_path: Path
    residual_label_path: Optional[Path]


@dataclass
class CollatedBatch:
    batch: "Batch"
    target_wav: torch.Tensor  # [B, T_wav]
    residual_wav: torch.Tensor  # [B, T_wav]


def _collect_files(folder: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    files = {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    }
    if not files:
        raise ValueError(f"No supported files found in {folder}")
    return files


def _collect_files_optional(folder: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    if not folder.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    }


class SoftLabelDataset(Dataset):
    """Discovers matched (audio, video, mask, soft_label) stems under ``data_root``."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

        audio_files = _collect_files(self.data_root / "audio", AUDIO_EXTENSIONS)
        video_files = _collect_files(self.data_root / "video", VIDEO_EXTENSIONS)
        mask_files = _collect_files(self.data_root / "mask", VIDEO_EXTENSIONS)
        soft_label_files = _collect_files(self.data_root / "soft_labels", AUDIO_EXTENSIONS)

        common_stems = sorted(
            set(audio_files) & set(video_files) & set(mask_files) & set(soft_label_files)
        )
        if not common_stems:
            raise ValueError(
                "No matching stems found across audio/, video/, mask/, and soft_labels/ "
                f"under {self.data_root}."
            )

        missing_audio = sorted(
            (set(video_files) | set(mask_files) | set(soft_label_files)) - set(audio_files)
        )
        missing_video = sorted(
            (set(audio_files) | set(mask_files) | set(soft_label_files)) - set(video_files)
        )
        missing_mask = sorted(
            (set(audio_files) | set(video_files) | set(soft_label_files)) - set(mask_files)
        )
        missing_soft_label = sorted(
            (set(audio_files) | set(video_files) | set(mask_files)) - set(soft_label_files)
        )
        if missing_audio or missing_video or missing_mask or missing_soft_label:
            print("Warning: some files were skipped because a matching set was not found.")
            if missing_audio:
                print(f"  Missing audio for {len(missing_audio)} stem(s): {', '.join(missing_audio[:10])}")
            if missing_video:
                print(f"  Missing video for {len(missing_video)} stem(s): {', '.join(missing_video[:10])}")
            if missing_mask:
                print(f"  Missing mask for {len(missing_mask)} stem(s): {', '.join(missing_mask[:10])}")
            if missing_soft_label:
                print(
                    f"  Missing soft_labels for {len(missing_soft_label)} stem(s): "
                    f"{', '.join(missing_soft_label[:10])}"
                )

        residual_files = _collect_files_optional(self.data_root / "residual_labels", AUDIO_EXTENSIONS)

        self.samples = [
            SoftLabelSample(
                stem=stem,
                audio_path=audio_files[stem],
                video_path=video_files[stem],
                mask_path=mask_files[stem],
                soft_label_path=soft_label_files[stem],
                residual_label_path=residual_files.get(stem),
            )
            for stem in common_stems
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SoftLabelSample:
        return self.samples[index]


def _load_mono_resampled(path: Path, target_sampling_rate: int) -> torch.Tensor:
    wav, sampling_rate = torchaudio.load(str(path))
    if sampling_rate != target_sampling_rate:
        wav = torchaudio.functional.resample(wav, sampling_rate, target_sampling_rate)
    return wav.mean(0)  # [T]


def _match_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.size(-1) == length:
        return wav
    if wav.size(-1) > length:
        return wav[..., :length]
    return torch.nn.functional.pad(wav, (0, length - wav.size(-1)))


def make_collate_fn(
    processor: "SAMAudioProcessor", audio_sampling_rate: int
) -> Callable[[list[SoftLabelSample]], CollatedBatch]:
    """Builds a collate function that turns a list of ``SoftLabelSample`` into a
    processor ``Batch`` plus mixture-aligned ``target_wav``/``residual_wav`` tensors.
    """

    def collate(samples: list[SoftLabelSample]) -> CollatedBatch:
        audio_paths = [str(sample.audio_path) for sample in samples]
        video_paths = [str(sample.video_path) for sample in samples]
        mask_paths = [str(sample.mask_path) for sample in samples]
        descriptions = [""] * len(samples)

        masked_videos = processor.mask_videos(video_paths, mask_paths)
        batch = processor(audios=audio_paths, descriptions=descriptions, masked_videos=masked_videos)

        mixture_wav = batch.audios.squeeze(1)  # [B, T_wav]
        t_wav = mixture_wav.size(-1)

        target_wavs = []
        residual_wavs = []
        for sample, mixture in zip(samples, mixture_wav, strict=False):
            target = _match_length(
                _load_mono_resampled(sample.soft_label_path, audio_sampling_rate), t_wav
            )
            target_wavs.append(target)

            if sample.residual_label_path is not None:
                residual = _match_length(
                    _load_mono_resampled(sample.residual_label_path, audio_sampling_rate), t_wav
                )
            else:
                residual = mixture - target
            residual_wavs.append(residual)

        target_wav = torch.stack(target_wavs, dim=0)
        residual_wav = torch.stack(residual_wavs, dim=0)

        return CollatedBatch(batch=batch, target_wav=target_wav, residual_wav=residual_wav)

    return collate


__all__ = [
    "SoftLabelSample",
    "CollatedBatch",
    "SoftLabelDataset",
    "make_collate_fn",
]
