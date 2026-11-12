import argparse
import glob
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor


def extract_frames(video, out_dir, overwrite):
    if not overwrite and glob.glob(os.path.join(out_dir, "*.jpg")):
        return "skipped"
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-r", "1",
           os.path.join(out_dir, "%06d.jpg")]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"failed: {video}: {e}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return "failed"
    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--frame_root", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    jobs = [(v, os.path.join(args.frame_root, os.path.splitext(os.path.basename(v))[0]))
            for v in videos]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda j: extract_frames(j[0], j[1], args.overwrite), jobs))

    print(f"videos: {len(videos)}, done: {results.count('done')}, "
          f"skipped: {results.count('skipped')}, failed: {results.count('failed')}")


if __name__ == "__main__":
    main()
