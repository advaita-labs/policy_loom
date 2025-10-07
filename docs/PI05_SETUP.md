# Pi0.5 Training Setup Guide

This guide explains how to set up and train Physical Intelligence's Pi0.5 model using policy_loom.

## Overview

policy_loom integrates Physical Intelligence's official openpi implementation to train Pi0.5 models. Key features:

- ✅ Train from scratch or fine-tune pretrained checkpoints
- ✅ LoRA support for efficient fine-tuning
- ✅ Accepts LeRobot format datasets as input
- ✅ Isolated environment to avoid dependency conflicts
- ✅ GPU/CUDA training support

`★ Architecture ─────────────────────────────────`
**Data Flow:**
LeRobot Dataset → Transform → OpenPI Format → Pi0.5 Model → Training

**Key Components:**
1. `Pi05Adapter` - Integrates openpi models with policy_loom
2. `OpenPITransform` - Converts LeRobot data to openpi format
3. Isolated venv - Prevents dependency conflicts
`─────────────────────────────────────────────────`

## Installation

### Prerequisites

- Python 3.11+ (required by the project)
- NVIDIA GPU with CUDA support
- Git and Git LFS

### Step 1: Install policy_loom with Pi0.5 Support

```bash
# Navigate to policy_loom directory
cd policy_loom

# Install with pi05 extra (skip Git LFS files we don't need)
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
```

**Note:** We set `GIT_LFS_SKIP_SMUDGE=1` to skip downloading large test dataset files from the lerobot dependency. We only need the Python code, not the test data.

This installs:
- Physical Intelligence's openpi package
- All required dependencies (PyTorch, transformers, JAX, etc.)
- policy_loom training infrastructure

### Step 2: Verify Installation

```bash
# Check openpi is installed
python -c "import openpi; print('✓ openpi installed')"

# Check policy_loom can load adapter
python -c "from loom.training.adapters.pi05 import Pi05Adapter; print('✓ Pi05Adapter available')"
```

**Troubleshooting Installation:**

If you see Git LFS errors during installation:
```bash
# Solution: Skip LFS files (we don't need test datasets)
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
```

If openpi import fails, ensure you're using Python 3.11+:
```bash
python --version  # Should be 3.11 or higher
```

## Quick Start

### Training from Scratch

```bash
# Train with minimal config
loom train configs/pi05_minimal.yaml
```

### Fine-Tuning with LoRA

```bash
# Fine-tune with LoRA
loom train configs/pi05_lora.yaml
```

## Configuration

### Basic Configuration (pi05_minimal.yaml)

```yaml
model:
  type: pi05
  action_dim: 7
  action_horizon: 10
  pretrained_path: null  # Train from scratch
  use_lora: false
  freeze_backbone: false

data:
  type: lerobot
  dataset: "lerobot/pusht"
  train_split: train

training:
  epochs: 50
  batch_size: 8
  learning_rate: 1e-4
```

### LoRA Fine-Tuning (pi05_lora.yaml)

```yaml
model:
  type: pi05
  action_dim: 7
  action_horizon: 10
  pretrained_path: "gs://openpi-assets/checkpoints/pi05_base"
  use_lora: true
  lora_rank: 16
  freeze_backbone: true

data:
  type: lerobot
  dataset: "lerobot/aloha_sim_insertion_human"
  train_split: train
  eval_split: test

training:
  epochs: 20
  batch_size: 16
  learning_rate: 5e-5
```

## Configuration Options

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | - | Must be "pi05" |
| `action_dim` | int | 7 | Action space dimension |
| `action_horizon` | int | 10 | Number of future actions to predict |
| `pretrained_path` | str/null | null | Path or GCS URL to openpi checkpoint |
| `use_lora` | bool | false | Enable LoRA fine-tuning |
| `lora_rank` | int | 8 | LoRA rank (if use_lora=true) |
| `freeze_backbone` | bool | false | Freeze vision/language backbone |
| `image_size` | list | [224, 224] | Input image size [H, W] |
| `default_prompt` | str/null | null | Optional default task prompt |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 50 | Number of training epochs |
| `batch_size` | int | 8 | Training batch size |
| `learning_rate` | float | 1e-4 | Learning rate |
| `weight_decay` | float | 1e-5 | Weight decay coefficient |
| `gradient_clip_norm` | float | 1.0 | Gradient clipping norm |
| `num_workers` | int | 4 | DataLoader workers |

### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | str | "lerobot" | Dataset type |
| `dataset` | str | - | HuggingFace dataset repo ID |
| `train_split` | str | "train" | Training split name |
| `eval_split` | str/null | null | Evaluation split name |
| `local_dir` | str/null | null | Local cache directory |

## Data Format

### LeRobot Input Format

policy_loom accepts LeRobot format datasets from HuggingFace Hub:

```python
# Dataset structure
{
    "observation": {
        "state": [...]  # Proprioceptive state
        "images": {
            "cam0": [...],  # RGB image (H, W, 3) uint8
            "cam1": [...],  # Additional cameras
        }
    },
    "action": [...],  # Robot action
    "episode_index": int,
    "frame_index": int,
}
```

### OpenPI Transform

The adapter automatically transforms LeRobot data to openpi format:

```python
# Transformed to OpenPI format
obs_dict = {
    "images": {"cam0": float32[-1,1]},  # Normalized to [-1, 1]
    "image_masks": {"cam0": bool},      # All True for real images
    "state": float32,                   # Proprio state
    "tokenized_prompt": int32,          # Optional prompts
}
actions = float32
```

## Training Modes

### 1. Train from Scratch

Create a new Pi0.5 model with random initialization:

```yaml
model:
  pretrained_path: null
  use_lora: false
  freeze_backbone: false
```

**Use when:**
- You have large custom dataset (>10k episodes)
- Task is very different from pretraining distribution
- You want full model control

### 2. Fine-Tune Full Model

Load pretrained checkpoint and fine-tune all parameters:

```yaml
model:
  pretrained_path: "gs://openpi-assets/checkpoints/pi05_base"
  use_lora: false
  freeze_backbone: false
```

**Use when:**
- You have moderate dataset (1k-10k episodes)
- Task is related but not identical to pretraining
- You have sufficient GPU memory (requires ~40GB+)

### 3. LoRA Fine-Tuning

Load pretrained checkpoint and fine-tune using LoRA:

```yaml
model:
  pretrained_path: "gs://openpi-assets/checkpoints/pi05_base"
  use_lora: true
  lora_rank: 16
  freeze_backbone: true
```

**Use when:**
- You have small dataset (<1k episodes)
- Task is similar to pretraining distribution
- Limited GPU memory (requires ~16GB+)
- Fast iteration needed

## Available Checkpoints

Physical Intelligence provides pretrained checkpoints via GCS:

```yaml
# Base model (general robotics)
pretrained_path: "gs://openpi-assets/checkpoints/pi05_base"

# Fine-tuned for specific platforms
pretrained_path: "gs://openpi-assets/checkpoints/pi05_droid"
pretrained_path: "gs://openpi-assets/checkpoints/pi05_aloha"
```

Or use local path:
```yaml
pretrained_path: "./checkpoints/my_pi05_model"
```

## Multiple Model Support

policy_loom supports multiple model types through optional extras:

```bash
# Install diffusion policy support
uv sync --extra diffusion

# Install Pi0.5 support (requires GIT_LFS_SKIP_SMUDGE)
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05

# Install both
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra diffusion --extra pi05
```

**Note:** All models now require Python 3.11+.

## Troubleshooting

### ImportError: No module named 'openpi'

**Solution:** Ensure pi05 extra is installed:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
python -c "import openpi"
```

### CUDA out of memory

**Solutions:**
1. Reduce batch size: `batch_size: 4`
2. Enable LoRA: `use_lora: true`
3. Freeze backbone: `freeze_backbone: true`
4. Use gradient checkpointing (if available in openpi)

### Dependency conflicts

**Solution:** Reinstall with clean environment:
```bash
# Remove virtual environment
rm -rf .venv

# Reinstall
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
```

### Git LFS download errors

**Error:** `remote missing object` or `smudge filter lfs failed`

**Solution:** Skip LFS files during installation:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
```

These are large test dataset files from lerobot that we don't need.

### Model creation fails

**Solution:** Verify openpi installation:
```bash
python -c "from openpi.models.pi0 import Pi0Config; print('openpi models available')"
```

If error persists, openpi API may have changed. Check [openpi GitHub](https://github.com/Physical-Intelligence/openpi) for updates.

### Training crashes on data loading

**Solution:** Check dataset format:
```python
from datasets import load_dataset
ds = load_dataset("lerobot/pusht", split="train")
print(ds[0].keys())  # Should have 'observation', 'action', etc.
```

## Performance Tips

### GPU Utilization

- **Batch size:** Start with 8, increase if GPU memory allows
- **Num workers:** Set to number of CPU cores (typically 4-8)
- **Mixed precision:** Enabled by default in openpi

### Training Speed

- **LoRA:** 2-3x faster than full fine-tuning
- **Frozen backbone:** Saves ~30-40% memory and compute
- **Multi-GPU:** Currently requires manual setup with openpi's distributed training

### Data Loading

- **Cache datasets:** Set `local_dir: ./data/cache` to avoid re-downloading
- **Prefetch:** DataLoader with `num_workers > 0` prefetches batches
- **Pin memory:** Automatically enabled for GPU training

## Next Steps

- See [Training Guide](./TRAINING_GUIDE.md) for advanced training techniques
- See [Configuration Examples](../configs/README.md) for more config options
- See [Testing Guide](./TESTING_GUIDE.md) for validation strategies
- Check [openpi documentation](https://github.com/Physical-Intelligence/openpi) for model details

## Support

- **Issues:** [policy_loom GitHub Issues](https://github.com/your-org/policy_loom/issues)
- **Discussions:** [policy_loom Discussions](https://github.com/your-org/policy_loom/discussions)
- **OpenPI Issues:** [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi/issues)
