# This script was modified from https://github.com/facebookresearch/sam-audio/blob/main/examples/visual_prompting.ipynb
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import sam3.visualization_utils as utils
import torch
import torchvision
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from IPython.display import Audio, Video

from sam3.model_builder import build_sam3_video_predictor
from torchcodec.decoders import VideoDecoder
from tqdm import trange

from sam_audio import SAMAudio, SAMAudioProcessor

# Setting up DPP 
dist.init_process_group(backend='nccl')

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# ------------------------------------
# SAM3 video predictor in DPP 
# video_predictor = build_sam3_video_predictor()
video_predictor = DistributedDataParallel(video_predictor, device_ids=[local_rank])

# TODO: REPLACE WITH VGG DIRECTORY
# video_file = "assets/office.mp4"
# Video(video_file, embed=True, width=640, height=360)
video_dir = "assets/vgg/"

class VideoDirDataset(Dataset):
    def __init__(self, root_dir, exts=(".mp4", ".mov", ".mkv", ".avi", ".webm")):
        self.files = sorted(
            str(p)
            for p in Path(root_dir).iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        if not self.files:
            raise ValueError(f"No video files found in: {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]


def init_dataloader(video_dir):
    video_dataset = VideoDirDataset(video_dir)
    video_sampler = DistributedSampler(
        video_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False,
    )
    video_loader = DataLoader(video_dataset, batch_size=1, sampler=video_sampler, num_workers=2)
    return video_loader


def process_video(video_file):
    decoder = VideoDecoder(video_file)
    height, width = decoder.metadata.height, decoder.metadata.width

    response = video_predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": video_file,
        }
    )
    session_id = response["session_id"]
    outputs = []
    for frame_index in trange(len(decoder)):
        response = video_predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": "The person on the left",
            }
        )
        output = response["outputs"]
        mask = output["out_binary_masks"]
        if mask.shape[0] == 0:
            if frame_index > 0:
                mask = outputs[-1]
            else:
                mask = np.zeros((1, height, width), dtype=bool)
        outputs.append(mask)
    
    # Return both frames and mask
    frames = decoder[:]
    mask = torch.from_numpy(np.concatenate(outputs)).unsqueeze(1)
    return frames, mask

def generate_mask(video_dir):
    video_loader = init_dataloader(video_dir)
    results = {}  # Store all results: {video_path: (frames, mask)}
    for batch in video_loader:
        vid_file_dir = batch[0]
        frames, mask = process_video(vid_file_dir)
        results[vid_file_dir] = (frames, mask)
        print(f"Rank {dist.get_rank()}: Processed {vid_file_dir}, frames: {frames.shape}, mask: {mask.shape}")

    return results


if __name__ == "__main__":
    # for testing with a single video file
    results = generate_mask(video_dir)
    




