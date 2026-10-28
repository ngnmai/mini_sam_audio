from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import random
import shutil

DATA_SUBDIRS = ("audio", "video", "mask", "soft_labels")
SOFT_LABEL_SUFFIXES = ("residual", "target")


def parse_args():
    parser = ArgumentParser(
        description="Split a dataset of audio/video/mask/soft_labels folders into train and validation folders."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing the source training data with audio, video, mask and soft_labels subfolders.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        required=True,
        help="Fraction of samples to place in the train folder. Validation gets the remainder.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root directory. Defaults to <input-dir-name>_split next to the input folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before shuffling samples.",
    )
    return parser.parse_args()


def validate_train_ratio(train_ratio: float) -> float:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be a float between 0 and 1, exclusive.")
    return train_ratio


def resolve_output_dir(input_dir: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    return input_dir.parent / f"{input_dir.name}_split"


def soft_label_sample_name(stem: str) -> str:
    # soft_labels filenames carry a "_residual"/"_target" suffix that must be stripped to recover the sample name
    for suffix in SOFT_LABEL_SUFFIXES:
        if stem.endswith(f"_{suffix}"):
            return stem[: -(len(suffix) + 1)]

    raise ValueError(
        f"soft_labels file '{stem}' does not end with '_residual' or '_target'."
    )


def collect_samples(input_dir: Path) -> dict[str, dict[str, list[Path]]]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    samples: dict[str, dict[str, list[Path]]] = {}

    for subdir_name in DATA_SUBDIRS:
        subdir = input_dir / subdir_name
        if not subdir.is_dir():
            raise FileNotFoundError(f"Expected subfolder not found: {subdir}")

        files = sorted(path for path in subdir.rglob("*") if path.is_file())
        if not files:
            raise ValueError(f"No files found under {subdir}")

        for file in files:
            if subdir_name == "soft_labels":
                sample_name = soft_label_sample_name(file.stem)
            else:
                sample_name = file.stem

            samples.setdefault(sample_name, {}).setdefault(subdir_name, []).append(file)

    return samples


def validate_samples(samples: dict[str, dict[str, list[Path]]]) -> None:
    errors: list[str] = []

    for sample_name, folders in samples.items():
        for subdir_name in ("audio", "video", "mask"):
            count = len(folders.get(subdir_name, []))
            if count != 1:
                errors.append(f"Sample '{sample_name}' has {count} file(s) in '{subdir_name}', expected 1.")

        soft_label_files = folders.get("soft_labels", [])
        if len(soft_label_files) != len(SOFT_LABEL_SUFFIXES):
            errors.append(
                f"Sample '{sample_name}' has {len(soft_label_files)} file(s) in 'soft_labels', "
                f"expected one per suffix {SOFT_LABEL_SUFFIXES}."
            )

    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))


def split_samples(
    sample_names: list[str], train_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    shuffled_names = sample_names.copy()
    random.Random(seed).shuffle(shuffled_names)

    train_count = int(len(shuffled_names) * train_ratio)
    train_names = shuffled_names[:train_count]
    validation_names = shuffled_names[train_count:]
    return train_names, validation_names


def copy_split(
    sample_names: list[str],
    samples: dict[str, dict[str, list[Path]]],
    input_dir: Path,
    output_dir: Path,
    split_name: str,
) -> None:
    for sample_name in sample_names:
        for files in samples[sample_name].values():
            for source_file in files:
                relative_path = source_file.relative_to(input_dir)
                destination_file = output_dir / split_name / relative_path
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination_file)


def split_train_validation(input_dir: Path, train_ratio: float, output_dir: Path, seed: int) -> None:
    samples = collect_samples(input_dir)
    validate_samples(samples)

    sample_names = sorted(samples.keys())
    train_names, validation_names = split_samples(sample_names, train_ratio, seed)

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation").mkdir(parents=True, exist_ok=True)

    copy_split(train_names, samples, input_dir, output_dir, "train")
    copy_split(validation_names, samples, input_dir, output_dir, "validation")

    print(f"Source samples: {len(sample_names)}")
    print(f"Train samples: {len(train_names)}")
    print(f"Validation samples: {len(validation_names)}")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir)
    train_ratio = validate_train_ratio(args.train_ratio)
    output_dir = resolve_output_dir(input_dir, args.output_dir)

    split_train_validation(input_dir, train_ratio, output_dir, args.seed)