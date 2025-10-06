# Pi0.5 Training Guide

This guide explains how to train Physical Intelligence's pi0.5 VLA model using policy_loom.

## Overview

Pi0.5 is the latest Vision-Language-Action model from Physical Intelligence with:
- Improved open-world generalization
- Dual decoding pathways (high-level + low-level actions)
- 50Hz action chunking
- Trained on 100+ diverse environments

## Installation

### ⚠️ Important: Separate Environment Required

Pi0.5 requires a custom transformers branch that conflicts with SmolVLA and other models.
**Always install pi0.5 in a dedicated virtual environment:**

```bash
# Create dedicated venv for pi0.5
python -m venv venv-pi05
source venv-pi05/bin/activate  # Windows: venv-pi05\Scripts\activate

# Clone and install policy_loom
git clone https://github.com/advaita-labs/policy_loom.git
cd policy_loom
uv sync --extra pi05
```

###Requirements

- Python >= 3.10
- CUDA-capable GPU (>22GB VRAM recommended for full fine-tuning)
- PyTorch >= 2.2.1

## Quick Start

### 1. Train on a LeRobot Dataset

```bash
# Activate pi0.5 environment
source venv-pi05/bin/activate

# Train on Koch test dataset
python scripts/train_pi05.py \\
    --dataset lerobot/koch_test \\
    --output checkpoints/pi05_koch \\
    --epochs 50 \\
    --batch-size 8

# Train with W&B logging
python scripts/train_pi05.py \\
    --dataset lerobot/aloha_sim_transfer_cube_scripted \\
    --output checkpoints/pi05_aloha \\
    --wandb \\
    --wandb-project my_pi05_training
```

### 2. Fine-tune from Pretrained Checkpoint

```bash
# Fine-tune pi05_base on custom dataset
python scripts/train_pi05.py \\
    --dataset lerobot/your_custom_dataset \\
    --model lerobot/pi05_base \\
    --epochs 100 \\
    --batch-size 16 \\
    --lr 1e-5 \\
    --output checkpoints/pi05_finetuned
```

### 3. Training Options

```bash
python scripts/train_pi05.py --help

# Common options:
  --dataset REPO_ID          LeRobot dataset (e.g., 'lerobot/koch_test')
  --model PATH               Pretrained model (default: lerobot/pi05_base)
  --freeze-backbone          Freeze VLM backbone (faster training, less memory)
  --epochs N                 Number of epochs (default: 100)
  --batch-size N             Batch size (default: 8)
  --lr FLOAT                 Learning rate (default: 1e-4)
  --output DIR               Checkpoint directory
  --wandb                    Enable W&B logging
```

## Advanced Usage

### Using Custom Datasets

If you have your own robot data, first convert it to LeRobot format:

```python
from loom.io.synchronized import SynchronizedVideoMCAPReader
from loom.pipeline import merge_streams

# 1. Load your data using policy_loom readers
left_cam = SynchronizedVideoMCAPReader(...)
right_cam = SynchronizedVideoMCAPReader(...)

# 2. Merge streams
samples = list(merge_streams(left_cam, right_cam))

# 3. Export to LeRobot format (TODO: implement exporter)
# This will be added in future versions

# 4. Upload to HuggingFace Hub or use locally
# Then train with: python scripts/train_pi05.py --dataset your_dataset
```

### Memory Optimization

Pi0.5 is a large model. To reduce memory usage:

```bash
# Option 1: Freeze backbone (recommended for fine-tuning)
python scripts/train_pi05.py \\
    --dataset lerobot/your_dataset \\
    --freeze-backbone \\
    --batch-size 16

# Option 2: Reduce batch size
python scripts/train_pi05.py \\
    --dataset lerobot/your_dataset \\
    --batch-size 4

# Option 3: Use gradient accumulation (coming soon)
```

### Checkpointing

By default, policy_loom saves:
- Checkpoint every 1000 steps
- Top 3 best checkpoints (by eval loss)
- Last 2 checkpoints

Customize with:
```bash
python scripts/train_pi05.py \\
    --dataset lerobot/your_dataset \\
    --save-every 500 \\
    --output checkpoints/pi05
```

## Available Pretrained Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `lerobot/pi05_base` | Base pi0.5 model | General fine-tuning |
| `lerobot/pi05_libero` | Fine-tuned on LIBERO | Simulated manipulation |
| `lerobot/pi05_droid` | Fine-tuned on DROID | Real-world manipulation |

## Troubleshooting

### Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch-size 4`
2. Freeze backbone: `--freeze-backbone`
3. Use smaller model or fewer workers

### Transformers Version Conflict

**Error**: `ImportError` or key mismatch in GemmaForCausalLM

**Solution**: Ensure you're in the pi05 venv:
```bash
source venv-pi05/bin/activate
pip list | grep transformers  # Should show custom branch
```

### Dataset Loading Issues

**Error**: `Dataset not found` or `Connection error`

**Solutions**:
1. Check internet connection (HuggingFace Hub access)
2. Verify dataset name: `lerobot/dataset_name`
3. Try downloading manually first:
```python
from datasets import load_dataset
ds = load_dataset("lerobot/koch_test")  # Test download
```

## Performance Tips

1. **Use fast storage**: Place checkpoints on SSD/NVMe
2. **Monitor GPU utilization**: `nvidia-smi -l 1`
3. **Enable W&B**: Track metrics in real-time with `--wandb`
4. **Use multiple workers**: Dataloader workers are set to 4 by default

## Next Steps

- [LeRobot Dataset Format](https://huggingface.co/docs/lerobot)
- [Pi0.5 Paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [Physical Intelligence Blog](https://www.physicalintelligence.company/blog/pi05)

## FAQ

**Q: Can I train pi0.5 alongside DiffusionPolicy?**
A: No, install them in separate virtual environments due to transformers conflicts.

**Q: How long does training take?**
A: Depends on dataset size and GPU. ~2-4 hours for 50 epochs on koch_test with A100.

**Q: Can I use multiple GPUs?**
A: Distributed training support coming soon.

**Q: How do I use the trained model for inference?**
A: Load the checkpoint with LeRobot's Pi0Policy.from_pretrained() method.
