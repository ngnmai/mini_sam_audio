import argparse
import fcntl
import json
from pathlib import Path

import pandas as pd

RESULTS_JSON_NAME = "judge_results.json"
RESULTS_PARQUET_NAME = "judge_results.parquet"
CATEGORIES = ["overall", "recall", "precision", "faithfulness"]


def compute_statistics(df: pd.DataFrame) -> dict:
    statistics = {}
    for category in CATEGORIES:
        if category not in df.columns:
            continue
        values = df[category]
        statistics[category] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "range": [float(values.min()), float(values.max())],
        }
    return statistics


def write_summary(json_path: Path, statistics: dict) -> list:
    lock_path = json_path.with_suffix(json_path.suffix + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            with open(json_path) as f:
                existing = json.load(f)
            records = existing["records"] if isinstance(existing, dict) and "records" in existing else existing

            with open(json_path, "w") as f:
                json.dump({"statistics": statistics, "records": records}, f, indent=2)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean/median/range statistics from judge.py results and prepend them to the results JSON."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help=f"Directory containing {RESULTS_PARQUET_NAME} and {RESULTS_JSON_NAME} produced by judge.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.expanduser().resolve()

    parquet_path = result_dir / RESULTS_PARQUET_NAME
    json_path = result_dir / RESULTS_JSON_NAME

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {RESULTS_PARQUET_NAME} in {result_dir}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {RESULTS_JSON_NAME} in {result_dir}")

    df = pd.read_parquet(parquet_path)
    statistics = compute_statistics(df)
    records = write_summary(json_path, statistics)

    print(f"Wrote statistics for {len(records)} record(s) to {json_path}")
    for category, stats in statistics.items():
        print(
            f"  {category}: mean={stats['mean']:.3f} median={stats['median']:.3f} "
            f"range=[{stats['range'][0]:.3f}, {stats['range'][1]:.3f}]"
        )


if __name__ == "__main__":
    main()
