from sam_audio import SAMAudio, SAMAudioProcessor
import torchaudio
import torch

model = SAMAudio.from_pretrained("facebook/sam-audio-large")
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
model = model.eval().cuda()

file = "<audio file>" # audio file path or torch tensor
description = "" # no taking text descriptions 
masked_video = processor.mask_videos([frames], [mask]) 

batch = processor(
    audios=[file],
    descriptions=[description],
    masked_videos=[masked_video]
).to("cuda")


with torch.inference_mode():
    # NOTE: `predict_spans` and `reranking_candidates` have a large impact on performance.
    # Setting `predict_span=True` and `reranking_candidates=8` will give you better results at the cost of
    # latency and memory. See the "Span Prediction" section below for more details
    # => NOTE We are not working with text and span currently
   result = model.separate(batch, predict_spans=False, reranking_candidates=1)

# Save separated audio
sample_rate = processor.audio_sampling_rate
torchaudio.save("target.wav", result.target.cpu(), sample_rate)      # The isolated sound
torchaudio.save("residual.wav", result.residual.cpu(), sample_rate)  # Everything else
