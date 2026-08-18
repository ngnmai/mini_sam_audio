"""Compute batch SDR, SIR, and SAR metrics for separated audio files.

The script expects two directory roots:

* ``mixture_root`` with mixture files named ``<name>.wav``
* ``isolated_root`` with paired reference files named ``<name>__target.wav``
  and ``<name>_residual.wav``

For every mixture stem that has both isolated files, the script loads the
audio, aligns sample rate and length, computes source-separation metrics by
treating the mixture as the evaluated signal against the isolated target and
residual references, and saves the per-file results as both parquet and JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


MIXTURE_SUFFIX = ".wav"
TARGET_SUFFIX = "__target.wav"
RESIDUAL_SUFFIX = "_residual.wav"
DEFAULT_OUTPUT_DIRNAME = "audio_metrics"
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SDR, SIR, and SAR for batch audio separation outputs."
    )
    parser.add_argument(
        "--audio-root",
        "--mixture-root",
        dest="mixture_root",
        type=Path,
        required=True,
        help="Folder containing mixture .wav files.",
    )
    parser.add_argument(
        "--isolated-audio-root",
        "--isolated-root",
        dest="isolated_root",
        type=Path,
        required=True,
        help="Folder containing paired target and residual .wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for parquet and JSON outputs. Defaults to <isolated-root.parent>/audio_metrics.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Optional explicit parquet output path.",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="Optional explicit JSON output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a mixture file does not have both isolated counterparts.",
    )
    return parser.parse_args()


def collect_wav_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    return sorted(path for path in root.rglob(f"*{MIXTURE_SUFFIX}") if path.is_file())


def index_mixtures(mixture_root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    duplicates: list[Path] = []

    for path in collect_wav_files(mixture_root):
        if path.name.endswith(TARGET_SUFFIX) or path.name.endswith(RESIDUAL_SUFFIX):
            continue

        stem = path.stem
        if stem in indexed:
            duplicates.append(path)
            continue
        indexed[stem] = path

    if duplicates:
        duplicate_list = "\n".join(str(path) for path in duplicates)
        raise ValueError(f"Duplicate mixture stems detected under {mixture_root}:\n{duplicate_list}")

    if not indexed:
        raise ValueError(f"No mixture wav files were found in {mixture_root}")

    return indexed


def index_isolated_files(isolated_root: Path) -> dict[str, dict[str, Path]]:
    indexed: dict[str, dict[str, Path]] = {}
    duplicates: list[Path] = []

    for path in collect_wav_files(isolated_root):
        if path.name.endswith(TARGET_SUFFIX):
            stem = path.name[: -len(TARGET_SUFFIX)]
            role = "target"
        elif path.name.endswith(RESIDUAL_SUFFIX):
            stem = path.name[: -len(RESIDUAL_SUFFIX)]
            role = "residual"
        else:
            continue

        roles = indexed.setdefault(stem, {})
        if role in roles:
            duplicates.append(path)
            continue
        roles[role] = path

    if duplicates:
        duplicate_list = "\n".join(str(path) for path in duplicates)
        raise ValueError(f"Duplicate isolated outputs detected under {isolated_root}:\n{duplicate_list}")

    if not indexed:
        raise ValueError(
            f"No isolated wav files named *{TARGET_SUFFIX} or *{RESIDUAL_SUFFIX} were found in {isolated_root}"
        )

    return indexed


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    array = waveform.to(dtype=torch.float64).cpu().numpy()
    if array.ndim == 2:
        array = array.mean(axis=0)
    elif array.ndim != 1:
        raise ValueError(f"Unsupported audio shape for {path}: {array.shape}")
    return array.astype(np.float64, copy=False), sample_rate


def resample_audio(array: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return array

    waveform = torch.from_numpy(array).to(dtype=torch.float64).unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return resampled.squeeze(0).cpu().numpy()


def align_audio(reference: np.ndarray, *signals: np.ndarray) -> tuple[np.ndarray, ...]:
    lengths = [len(reference), *[len(signal) for signal in signals]]
    if not lengths:
        raise ValueError("At least one audio signal is required")

    min_length = min(lengths)
    trimmed = (reference[:min_length], *[signal[:min_length] for signal in signals])
    return trimmed


def safe_db(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10((numerator + EPSILON) / (denominator + EPSILON))


def compute_three_source_metrics(reference: np.ndarray, estimate: np.ndarray, interference: np.ndarray) -> dict[str, float]:
    reference, estimate, interference = align_audio(reference, estimate, interference)

    stacked = np.stack([reference, interference], axis=1)
    coeffs, *_ = np.linalg.lstsq(stacked, estimate, rcond=None)
    target_component = coeffs[0] * reference
    interference_component = coeffs[1] * interference
    artifact_component = estimate - target_component - interference_component

    target_energy = float(np.sum(target_component**2))
    interference_energy = float(np.sum(interference_component**2))
    artifact_energy = float(np.sum(artifact_component**2))

    return {
        "sdr": safe_db(target_energy, interference_energy + artifact_energy),
        "sir": safe_db(target_energy, interference_energy),
        "sar": safe_db(target_energy + interference_energy, artifact_energy),
    }


def match_samples(mixture_root: Path, isolated_root: Path, strict: bool) -> list[dict[str, Path | str]]:
    mixtures = index_mixtures(mixture_root)
    isolated = index_isolated_files(isolated_root)

    sample_rows: list[dict[str, Path | str]] = []
    missing: list[str] = []

    for stem, mixture_path in mixtures.items():
        isolated_pair = isolated.get(stem)
        if isolated_pair is None or "target" not in isolated_pair or "residual" not in isolated_pair:
            missing.append(stem)
            continue

        sample_rows.append(
            {
                "stem": stem,
                "mixture_path": mixture_path,
                "target_path": isolated_pair["target"],
                "residual_path": isolated_pair["residual"],
            }
        )

    if missing:
        missing_preview = ", ".join(missing[:10])
        message = f"Missing isolated target/residual pairs for {len(missing)} mixture file(s): {missing_preview}"
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}")

    if not sample_rows:
        raise ValueError("No fully matched mixture/isolated file pairs were found")

    return sample_rows


def evaluate_sample(sample: dict[str, Path | str]) -> dict[str, object]:
    stem = str(sample["stem"])
    mixture_path = Path(sample["mixture_path"])
    target_path = Path(sample["target_path"])
    residual_path = Path(sample["residual_path"])

    mixture, mixture_rate = load_audio(mixture_path)
    target, target_rate = load_audio(target_path)
    residual, residual_rate = load_audio(residual_path)

    target = resample_audio(target, target_rate, mixture_rate)
    residual = resample_audio(residual, residual_rate, mixture_rate)
    mixture, target, residual = align_audio(mixture, target, residual)

    target_metrics = compute_three_source_metrics(target, mixture, residual)
    residual_metrics = compute_three_source_metrics(residual, mixture, target)

    reconstruction = target + residual
    mixture, reconstruction = align_audio(mixture, reconstruction)
    mix_error = mixture - reconstruction

    return {
        "stem": stem,
        "mixture_path": str(mixture_path),
        "target_path": str(target_path),
        "residual_path": str(residual_path),
        "sample_rate": mixture_rate,
        "duration_seconds": len(mixture) / float(mixture_rate),
        "target_sdr": target_metrics["sdr"],
        "target_sir": target_metrics["sir"],
        "target_sar": target_metrics["sar"],
        "residual_sdr": residual_metrics["sdr"],
        "residual_sir": residual_metrics["sir"],
        "residual_sar": residual_metrics["sar"],
        "mixture_reconstruction_sdr": safe_db(float(np.sum(mixture**2)), float(np.sum(mix_error**2))),
    }


def build_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.isolated_root.parent / DEFAULT_OUTPUT_DIRNAME

    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = args.parquet_path or (output_dir / "source_metrics.parquet")
    json_path = args.json_path or (output_dir / "source_metrics.json")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return parquet_path, json_path


def compute_metrics(args: argparse.Namespace) -> pd.DataFrame:
    samples = match_samples(args.mixture_root, args.isolated_root, args.strict)
    records = [evaluate_sample(sample) for sample in tqdm(samples, desc="Computing audio metrics")]
    return pd.DataFrame.from_records(records)


def build_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    numeric_columns = [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]
    aggregate: dict[str, float] = {}

    for column in numeric_columns:
        aggregate[f"mean_{column}"] = float(dataframe[column].mean())
        aggregate[f"median_{column}"] = float(dataframe[column].median())

    return {
        "num_files": int(len(dataframe)),
        "metrics": aggregate,
    }


def save_results(dataframe: pd.DataFrame, parquet_path: Path, json_path: Path) -> None:
    dataframe.to_parquet(parquet_path, index=False)

    payload = {
        "summary": build_summary(dataframe),
        "records": dataframe.to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def main() -> None:
    args = parse_args()
    dataframe = compute_metrics(args)
    parquet_path, json_path = build_output_paths(args)
    save_results(dataframe, parquet_path, json_path)
    print(f"Saved parquet metrics to {parquet_path}")
    print(f"Saved JSON metrics to {json_path}")


if __name__ == "__main__":
    main()