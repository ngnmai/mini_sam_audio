from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import random
import shutil


def parse_args():
    parser = ArgumentParser(
        description="Split a folder of training data into train and validation folders."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing the source training data.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        required=True,
        help="Fraction of files to place in the train folder. Validation gets the remainder.",
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
        help="Random seed used before shuffling files.",
    )
    return parser.parse_args()


def validate_train_ratio(train_ratio: float) -> float:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be a float between 0 and 1, exclusive.")
    return train_ratio


def collect_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = sorted(path for path in input_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No files found under {input_dir}")

    return files


def resolve_output_dir(input_dir: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    return input_dir.parent / f"{input_dir.name}_split"


def split_files(files: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    shuffled_files = files.copy()
    random.Random(seed).shuffle(shuffled_files)

    train_count = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:train_count]
    validation_files = shuffled_files[train_count:]
    return train_files, validation_files


def copy_split(files: list[Path], input_dir: Path, output_dir: Path, split_name: str) -> None:
    for source_file in files:
        relative_path = source_file.relative_to(input_dir)
        destination_file = output_dir / split_name / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)


def split_train_validation(input_dir: Path, train_ratio: float, output_dir: Path, seed: int) -> None:
    files = collect_files(input_dir)
    train_files, validation_files = split_files(files, train_ratio, seed)

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation").mkdir(parents=True, exist_ok=True)

    copy_split(train_files, input_dir, output_dir, "train")
    copy_split(validation_files, input_dir, output_dir, "validation")

    print(f"Source files: {len(files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(validation_files)}")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir)
    train_ratio = validate_train_ratio(args.train_ratio)
    output_dir = resolve_output_dir(input_dir, args.output_dir)

    split_train_validation(input_dir, train_ratio, output_dir, args.seed)