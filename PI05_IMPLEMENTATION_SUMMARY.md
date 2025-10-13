# Pi0.5 Integration Guide

## Overview

Policy Loom integrates Physical Intelligence's **π0.5 (Pi0.5)** vision-language-action model for robot manipulation. This guide covers architecture, installation, and usage.

## Architecture

Pi0.5 is a flow matching VLA model that combines:
- **PaliGemma** vision-language backbone (SigLIP vision encoder + Gemma 2B/7B language model)
- **Flow Matching** for continuous action generation
- **Multi-camera** RGB image processing
- **Language conditioning** for task specification

### Data Flow

```
MP4 Videos + MCAP → LeRobot Format → Pi0.5 Preprocessor → Model Training
                                     (224x224 images,
                                      tokenized prompts,
                                      32-dim actions)
```

### Key Components

1. **Model Adapter** (`src/loom/training/adapters/pi05.py`): Implements training interface
2. **Data Conversion** (`scripts/convert_mp4_mcap_to_lerobot.py`): Converts raw data to LeRobot format
3. **Configuration** (`configs/pi05_minimal.yaml`): Training hyperparameters

## Installation

### Prerequisites
- Python 3.11+
- CUDA 12.2+ (for GPU training)
- 16GB+ RAM

### Setup

```bash
# Create virtual environment
python -m venv venv-pi05
source venv-pi05/bin/activate  # Linux/Mac
# venv-pi05\Scripts\activate  # Windows

# Install dependencies (skip LFS for faster install)
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05

# Apply required transformers patches
cp -r .venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

### Verify Installation

```bash
python scripts/verify_pi05_installation.py
```

Expected output: `✅ SUCCESS: Pi0.5 Integration Verified!`

## Data Preparation

Convert your robot data (MP4 videos + MCAP) to LeRobot format:

```bash
python scripts/convert_mp4_mcap_to_lerobot.py \
    --input /path/to/raw/data \
    --output /path/to/lerobot/output \
    --task "pick and place manipulation" \
    --repo-id "robot-manipulation" \
    --fps 30
```

Expected directory structure:
```
/path/to/raw/data/
├── run0/
│   ├── videos/
│   │   ├── left_arm.perception_interface.left_cam.state.mp4
│   │   ├── right_arm.perception_interface.right_cam.state.mp4
│   │   └── torso.perception_interface.middle_cam.state.mp4
│   └── run0_0.mcap
├── run1/
└── ...
```

## Training

### Configure Training

Edit `configs/pi05_minimal.yaml`:

```yaml
model:
  type: pi05
  action_dim: 32  # Fixed by OpenPI
  action_horizon: 10
  paligemma_variant: "gemma_2b"  # or "gemma_7b"

training:
  batch_size: 8
  epochs: 100
  learning_rate: 1.0e-4

data:
  dataset: "/path/to/lerobot/output/robot-manipulation__run0"
  eval_dataset: "/path/to/lerobot/output/robot-manipulation__run1"
```

### Start Training

```bash
python -m loom.cli train configs/pi05_minimal.yaml
```

### Monitor Progress

Logs are saved to `logs/training.log` and checkpoints to `checkpoints/pi05_minimal/`.

## Model Configuration

### PaliGemma Variants
- `gemma_2b`: Faster, lower memory (recommended for testing)
- `gemma_7b`: Better performance, requires more VRAM

### Action Dimensions
OpenPI hardcodes action dimension to 32. Actions are automatically padded/truncated.

### Image Processing
All cameras are resized to 224×224 (PaliGemma requirement).

## Troubleshooting

### GPU Compatibility
If you see "no kernel image available", your GPU may not be supported by PyTorch 2.7.1. Run on CPU by setting `CUDA_VISIBLE_DEVICES=""`.

### Missing Videos
The conversion script automatically skips runs with missing video files.

### Memory Issues
Reduce `batch_size` in config or use `gemma_2b` instead of `gemma_7b`.

## API Reference

### Model Adapter Interface

```python
from loom.training.adapters.pi05 import Pi05Adapter

adapter = Pi05Adapter({
    'action_dim': 32,
    'action_horizon': 10,
    'max_token_len': 256
})

model = adapter.create_model()
optimizer = adapter.create_optimizer(model, lr=1e-4, weight_decay=0.01)
```

### Data Loader

```python
from loom.io.lerobot import LeRobotDatasetLoader

loader = LeRobotDatasetLoader(
    repo_id="/path/to/lerobot/dataset",
    split="train"
)
dataset = loader.to_torch_dataset()
```

## Known Limitations

1. **GPU Compatibility**: RTX 50-series requires PyTorch 2.8+
2. **Action Dimension**: Fixed at 32 (OpenPI constraint)
3. **Multi-GPU**: Not yet supported
4. **Mixed Precision**: Not supported in PyTorch implementation

## Further Reading

- [OpenPI GitHub](https://github.com/physical-intelligence/openpi)
- [PaliGemma Documentation](https://ai.google.dev/gemma/docs/paligemma)
- [Policy Loom Architecture](README.md)
