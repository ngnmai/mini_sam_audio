# Mini-SAM-Audio Project Structure

Project setup for the Mini-SAM-Audio model compression pipeline.

## Directory Structure

```
mini_sam_audio/
├── sam-audio/           # SAM-Audio base model (submodule)
├── src/                 # Main source code
│   └── mini_sam_audio/
│       ├── __init__.py
│       ├── compression/       # Model compression pipeline
│       │   └── __init__.py
│       └── utils/             # Utility functions
│           └── __init__.py
├── pyproject.toml       # Project metadata and dependencies
├── requirements.txt     # Detailed dependency list
└── README.md           # This file
```

## Installation

### Prerequisites
- Python 3.11+
- pip or conda

### Setup

1. Install the project in development mode with SAM-Audio:
```bash
cd /home/mai/STUDY/msc_thesis/mini_sam_audio
pip install -e .
```

This will install:
- The `mini_sam_audio` package
- SAM-Audio from the local submodule with all its dependencies

## Project Goals

This project implements a model compression pipeline to:
1. Load SAM-Audio models
2. Compress SAM-Audio through knowledge distillation
3. Train mini-SAM-Audio through KD
4. Evaluate compression quality and performance

## Next Steps

- [ ] Implement model loading utilities
- [ ] Set up student model
- [ ] Add evaluation metrics
- [ ] Create example compression scripts
- [ ] Revisit installation requirements based on actual PyTorch needs
