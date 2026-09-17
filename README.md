# Mini SAM Audio

Lightweight compressed version of SAM Audio. 

Mini SAM Audio currently only focuses on visual prompt. 
Inference and training scripts are written to be run parallel on CSC's Roihu.

## About 

This project features scripts that can do inference on SAM Audio, other models to be compared to and Mini SAM Audio's architecture. 

Mini SAM Audio still keeps the overall structure of SAM Audio but with lower levels of DiT blocks. 
Segmentation model is switched out for Yolo 11. Audio codec and Visual codec are all kept the same. 
Training and evaluation dataset are extracted from VGGSound.

## Installation

### Prerequisites
- Python 3.11+
- CUDA compatible GPU
- Optional: uv 

### Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from mini_sam_audio import SAMAudio, SAMAudioProcessor

# TODO add inference script
```