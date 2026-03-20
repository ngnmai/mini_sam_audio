# This script was modified from https://github.com/facebookresearch/sam-audio/blob/main/examples/visual_prompting.ipynb
import tempfile
from io import BytesIO

import cv2
import numpy as np
import sam3.visualization_utils as utils
import torch
import torchvision
from IPython.display import Audio, Video

from sam3.model_builder import build_sam3_video_predictor
from torchcodec.decoders import VideoDecoder
from tqdm import trange

from sam_audio import SAMAudio, SAMAudioProcessor

video_predictor = build_sam3_video_predictor()

# TODO: replace the video file path to VGG sound
video_file = "assets/office.mp4"
Video(video_file, embed=True, width=640, height=360)


decoder = VideoDecoder(video_file)
height, width = decoder.metadata.height, decoder.metadata.width

# TODO: change to multiple frames and files processing, parallel for Mahti
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


# ----- TEST WITH RANDOM FRAMS -----
# TODO: change to multiple frames and files processing, parallel for Mahti
def draw_masks_to_frame(
    frame: np.ndarray, masks: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    masked_frame = frame
    for mask, color in zip(masks, colors, strict=False):
        curr_masked_frame = np.where(mask[..., None], color, masked_frame)
        masked_frame = cv2.addWeighted(masked_frame, 0.75, curr_masked_frame, 0.25, 0)
        contours, _ = cv2.findContours(
            np.array(mask, dtype=np.uint8).copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE,
        )
        cv2.drawContours(masked_frame, contours, -1, (255, 255, 255), 1)
        cv2.drawContours(masked_frame, contours, -1, (0, 0, 0), 1)
        cv2.drawContours(masked_frame, contours, -1, color.tolist(), 1)
    return masked_frame


# Actual output 
# TODO: change the whole file to a function that can output @frames and @mask 
frames = decoder[:]
mask = torch.from_numpy(np.concatenate(outputs)).unsqueeze(1)

