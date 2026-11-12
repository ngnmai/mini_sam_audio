#!/bin/bash
# Single-clip DAVIS-FM separation on VGGSound: each wav is the mixture, the paired mp4 gives the frames.
# Usage: AUDIO_DIR=... VIDEO_DIR=... OUT_DIR=... WEIGHTS_FRAME=... WEIGHTS_UNET=... ./run_vgg.sh
set -euo pipefail

AUDIO_DIR="${AUDIO_DIR:-}"
VIDEO_DIR="${VIDEO_DIR:-}"
OUT_DIR="${OUT_DIR:-}"
WEIGHTS_FRAME="${WEIGHTS_FRAME:-}"
WEIGHTS_UNET="${WEIGHTS_UNET:-}"
GPU_ID="${GPU_ID:-0}"

AUDIO_DIR=assets/vgg/vgg_test/test/audio 
VIDEO_DIR=assets/vgg/vgg_test/test/video 
OUT_DIR=outputs/davis_vgg_test 
WEIGHTS_FRAME=assets/davisflow-ckpt/frame_best.pth
WEIGHTS_UNET=assets/davisflow-ckpt/unet_best.pth
GPU_ID=0

for var in AUDIO_DIR VIDEO_DIR OUT_DIR WEIGHTS_FRAME WEIGHTS_UNET; do
    if [[ -z "${!var}" ]]; then
        echo "Error: $var is not set. Export it or pass it inline, e.g. $var=/path ./run_vgg.sh" >&2
        exit 1
    fi
    # absolute paths, because the script cd's to its own directory below
    printf -v "$var" '%s' "$(realpath -m "${!var}")"
done
cd "$(dirname "$0")"

FRAME_ROOT="$OUT_DIR/frames"
LIST_VAL="$OUT_DIR/list.csv"

echo "[0/2] Extracting frames"
python extract_frames_vgg.py --video_dir "$VIDEO_DIR" --frame_root "$FRAME_ROOT"

echo "[1/2] Building clip list"
python build_vgg_lists.py --audio_dir "$AUDIO_DIR" --frame_root "$FRAME_ROOT" --out_list "$LIST_VAL"

# Model and data flags must match the AVE-trained weights.
OPTS=(
    --list_val "$LIST_VAL"
    --audio_root "$AUDIO_DIR"
    --frame_root "$FRAME_ROOT"
    --out_dir "$OUT_DIR"
    --weights_frame "$WEIGHTS_FRAME"
    --weights_unet "$WEIGHTS_UNET"

    # models
    --img_pool maxpool
    --num_channels 64
    --loss l1
    --weighted_loss 0
    --log_freq 1

    # frames-related
    --arch_frame clip
    --num_frames 11
    --stride_frames 1
    --frameRate 1

    # audio-related
    --audLen 110000
    --audRate 11000

    # runtime
    --num_gpus 1
    --gpu_ids "$GPU_ID"
    --workers 4
    --batch_size_per_gpu 4

    # display, viz
    --disp_iter 10
    --num_vis 40
)

echo "[2/2] Separating"
python -u main_vgg.py "${OPTS[@]}"
