from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

DATA_SUBDIRS = ("audio", "video", "mask", "soft_labels")
SOFT_LABEL_SUFFIXES = ("residual", "target")


def parse_args():
    parser = ArgumentParser(
        description=(
            "Remove data samples that are missing a corresponding file in any of the "
            "audio, video, mask or soft_labels folders. Soft labels are expected to be "
            "the limiting set, so any sample without a residual/target pair is dropped "
            "from audio, video and mask as well."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing the audio, video, mask and soft_labels subfolders.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the incomplete samples. Without this flag, only a dry-run report is printed.",
    )
    return parser.parse_args()


def resolve_input_dir(input_dir: str) -> Path:
    path = Path(input_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {path}")
    return path


def soft_label_sample_name(stem: str) -> str:
    # soft_labels filenames carry a "_residual"/"_target" suffix that must be stripped to recover the sample name
    for suffix in SOFT_LABEL_SUFFIXES:
        if stem.endswith(f"_{suffix}"):
            return stem[: -(len(suffix) + 1)]

    raise ValueError(
        f"soft_labels file '{stem}' does not end with '_residual' or '_target'."
    )


def collect_samples(input_dir: Path) -> dict[str, dict[str, list[Path]]]:
    samples: dict[str, dict[str, list[Path]]] = {}

    for subdir_name in DATA_SUBDIRS:
        subdir = input_dir / subdir_name
        if not subdir.is_dir():
            raise FileNotFoundError(f"Expected subfolder not found: {subdir}")

        files = sorted(path for path in subdir.rglob("*") if path.is_file())

        for file in files:
            if subdir_name == "soft_labels":
                sample_name = soft_label_sample_name(file.stem)
            else:
                sample_name = file.stem

            samples.setdefault(sample_name, {}).setdefault(subdir_name, []).append(file)

    return samples


def is_sample_complete(folders: dict[str, list[Path]]) -> bool:
    for subdir_name in ("audio", "video", "mask"):
        if len(folders.get(subdir_name, [])) != 1:
            return False

    return len(folders.get("soft_labels", [])) == len(SOFT_LABEL_SUFFIXES)


def split_complete_incomplete(
    samples: dict[str, dict[str, list[Path]]]
) -> tuple[list[str], list[str]]:
    complete_names = []
    incomplete_names = []

    for sample_name, folders in samples.items():
        if is_sample_complete(folders):
            complete_names.append(sample_name)
        else:
            incomplete_names.append(sample_name)

    return sorted(complete_names), sorted(incomplete_names)


def delete_samples(sample_names: list[str], samples: dict[str, dict[str, list[Path]]]) -> int:
    deleted_count = 0

    for sample_name in sample_names:
        for files in samples[sample_name].values():
            for file in files:
                file.unlink()
                deleted_count += 1

    return deleted_count


def filter_incomplete_samples(input_dir: Path, execute: bool) -> None:
    samples = collect_samples(input_dir)
    complete_names, incomplete_names = split_complete_incomplete(samples)

    print(f"Total samples: {len(samples)}")
    print(f"Complete samples (kept): {len(complete_names)}")
    print(f"Incomplete samples (removed): {len(incomplete_names)}")

    if incomplete_names:
        print("Incomplete sample names:")
        for sample_name in incomplete_names:
            print(f"  - {sample_name}")

    if not execute:
        print("Dry run only. Re-run with --execute to delete the files listed above.")
        return

    deleted_count = delete_samples(incomplete_names, samples)
    print(f"Deleted {deleted_count} file(s) across {len(incomplete_names)} incomplete sample(s).")


if __name__ == "__main__":
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)

    filter_incomplete_samples(input_dir, args.execute)
