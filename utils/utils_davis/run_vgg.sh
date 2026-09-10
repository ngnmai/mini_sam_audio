#!/bin/bash

OPTS=""
OPTS+="--id VGG-clip-FM "

# Input directories (replace placeholders before running)
OPTS+="--audio_dir AUDIO_DIR "
OPTS+="--video_dir VIDEO_DIR "

# Models
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 0 "

# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--arch_frame clip " # [resnet18, clip]
OPTS+="--num_frames 11 "
OPTS+="--stride_frames 2 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# runtime params (Roihu/CSC-friendly defaults)
OPTS+="--num_gpus 1 "
OPTS+="--gpu_ids 0 "
OPTS+="--workers 8 "
OPTS+="--batch_size_per_gpu 4 "

# where to load checkpoints from
OPTS+="--ckpt YOUR_CKPT "

# display
OPTS+="--disp_iter 200 "
OPTS+="--num_vis 40 "

OPTS+="--split test "
OPTS+="--mode eval "

python -u ../main_vgg.py $OPTS