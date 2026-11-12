"""Resample a folder of 16 kHz wav files to 11.025 kHz (files with another source rate are skipped)."""
import argparse
from math import gcd
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly

SRC_SR = 16000
DST_SR = 11025


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = input_dir.parent / f"{input_dir.name}_11kHz"

    files = sorted(input_dir.glob("*.wav"))
    if not files:
        parser.error(f"no .wav files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output folder: {output_dir}")
    g = gcd(SRC_SR, DST_SR)
    up, down = DST_SR // g, SRC_SR // g

    done = skipped_sr = skipped_exist = failed = 0
    for path in files:
        out_path = output_dir / path.name
        if out_path.exists() and not args.overwrite:
            skipped_exist += 1
            continue
        try:
            info = sf.info(path)
            if info.samplerate != SRC_SR:
                print(f"SKIP {path.name}: sample rate {info.samplerate} Hz, expected {SRC_SR} Hz")
                skipped_sr += 1
                continue
            audio, _ = sf.read(path, dtype="float32", always_2d=True)
            resampled = resample_poly(audio, up, down, axis=0)
            sf.write(out_path, resampled, DST_SR, subtype="PCM_16")
            done += 1
        except Exception as e:
            print(f"FAIL {path.name}: {e}")
            failed += 1

    print(
        f"resampled={done} skipped_wrong_sr={skipped_sr} "
        f"skipped_existing={skipped_exist} failed={failed} total={len(files)}"
    )


if __name__ == "__main__":
    main()
