import argparse
import csv
import glob
import os

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--frame_root", type=str, required=True)
    parser.add_argument("--out_list", type=str, required=True)
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--min_duration", type=float, default=9.0)  # 0 disables
    args = parser.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    rows = []
    missing = short_frames = short_audio = 0

    for wav in wavs:
        stem = os.path.splitext(os.path.basename(wav))[0]
        frame_dir = os.path.join(args.frame_root, stem)
        if not os.path.isdir(frame_dir):
            print(f"missing frames: {stem}")
            missing += 1
            continue

        n_frames = len(glob.glob(os.path.join(frame_dir, "*.jpg")))
        if n_frames < args.min_frames:
            short_frames += 1
            continue

        # _load_audio tiles short audio, so the separated output would repeat
        if args.min_duration > 0 and sf.info(wav).duration < args.min_duration:
            short_audio += 1
            continue

        rows.append([f"{stem}.wav", stem, n_frames])

    rows.sort(key=lambda r: r[1])
    out_dir = os.path.dirname(os.path.abspath(args.out_list))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_list, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"wavs: {len(wavs)}, missing frames: {missing}, "
          f"short (frames): {short_frames}, short (duration): {short_audio}, "
          f"rows written: {len(rows)}")


if __name__ == "__main__":
    main()
