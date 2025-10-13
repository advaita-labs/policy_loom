# Pi0.5 Training Guide

This guide explains how to train Physical Intelligence's π0.5 (Pi0.5) vision-language-action model using Policy Loom.

## Overview

**π0.5** is a flow-matching based VLA model that predicts robot actions conditioned on:
- **Vision**: Multi-camera RGB images processed by PaliGemma (SigLIP + Gemma)
- **Language**: Natural language task descriptions tokenized with PaliGemma tokenizer
- **State**: Robot proprioception (optional, can be embedded in language tokens)

**Key Features:**
- 32-dimensional action space (hardcoded in OpenPI)
- 10-50 action horizon (configurable)
- Flow matching for smooth action trajectories
- Supports fine-tuning on custom robot data

## Prerequisites

### Hardware Requirements
- **GPU Memory**: Minimum 22.5GB VRAM for LoRA fine-tuning (RTX 4090 or better)
- **Full Fine-tuning**: 70GB+ VRAM (A100 80GB / H100)
- **Inference Only**: 8GB+ VRAM

### Software Requirements
- Python 3.11+
- CUDA 12.2+ (for GPU support)
- Ubuntu 22.04 (tested, other Linux distros may work)

## Installation

### Step 1: Install Policy Loom with Pi0.5 Support

**CRITICAL**: Pi0.5 requires `transformers==4.53.2` which conflicts with DiffusionPolicy's requirements. You MUST use a separate virtual environment.

```bash
# Clone Policy Loom
git clone https://github.com/advaita-labs/policy_loom
cd policy_loom

# Create separate venv for Pi0.5
python -m venv venv-pi05
source venv-pi05/bin/activate  # On Windows: venv-pi05\Scripts\activate

# Install with Pi0.5 dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
```

### Step 2: Apply Transformers Patches

OpenPI requires modified transformers library for:
- AdaRMS normalization (adaptive RMS norm)
- Precision control (bfloat16/float32)
- KV cache without updates

```bash
# Apply patches
cp -r .venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/

# Verify patches
uv run python -c "from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly()"
```

### Step 3: Verify Installation

```bash
uv run python -c "
from loom.training.adapters.pi05 import Pi05Adapter
from loom.training.adapter import list_adapters

print('Available adapters:', list_adapters())
assert 'pi05' in list_adapters(), 'Pi05Adapter not registered!'

config = {'action_dim': 32, 'action_horizon': 10, 'max_token_len': 256}
adapter = Pi05Adapter(config)
model = adapter.create_model()
print(f'✓ Pi0.5 model created: {type(model).__name__}')
"
```

## Data Preparation

### Step 1: Convert Robot Data to LeRobot Format

Policy Loom uses LeRobot dataset format. Convert your MP4/MCAP recordings:

```bash
# Example: Convert single recording
uv run python scripts/convert_mp4_mcap_to_lerobot.py \
    --input /path/to/recordings \
    --output data/lerobot_output \
    --task "pick and place cube" \
    --repo-id "my-robot-data"
```

### Step 2: Verify Data Structure

LeRobot datasets should have this structure:

```
data/lerobot_output/
├── train/
│   ├── data/
│   │   ├── chunk-000/
│   │   │   ├── observation.images.left_cam-*.png
│   │   │   ├── observation.images.right_cam-*.png
│   │   │   └── observation.state-*.npy
│   │   └── ...
│   └── meta/
│       └── info.json
└── eval/
    └── (same structure)
```

### Step 3: Configure Camera Mapping

Pi0.5 expects specific camera names. Map your cameras in `configs/pi05_minimal.yaml`:

```yaml
data:
  camera_name_mapping:
    "observation.images.left_cam": "left"
    "observation.images.right_cam": "right"
    "observation.images.middle_cam": "top"
```

## Training Configuration

### Minimal Configuration

See `configs/pi05_minimal.yaml` for a working example:

```yaml
model:
  type: pi05
  action_dim: 32  # REQUIRED by OpenPI
  action_horizon: 10
  max_token_len: 256

training:
  learning_rate: 1.0e-4
  batch_size: 8
  num_epochs: 100

data:
  train_path: "data/lerobot_output/train"
  eval_path: "data/lerobot_output/eval"
  image_size: [224, 224]
```

### Advanced Configuration Options

#### Model Architecture

```yaml
model:
  paligemma_variant: "gemma_2b"  # or "gemma_7b" for larger model
  action_expert_variant: "gemma_300m"  # Action decoder size
  dtype: "bfloat16"  # or "float32" for full precision
```

#### Training Hyperparameters

```yaml
training:
  learning_rate: 1.0e-4  # OpenPI default
  weight_decay: 0.01
  batch_size: 8  # Adjust based on GPU memory
  gradient_accumulation_steps: 4  # Simulate larger batches
  num_workers: 4  # DataLoader parallelism
```

#### Loading Pretrained Weights

```yaml
checkpoints:
  # Option 1: Load from OpenPI JAX checkpoint
  pretrained_path: "gs://openpi-assets/checkpoints/pi05_droid"
  
  # Option 2: Load from converted PyTorch checkpoint
  pytorch_weight_path: "checkpoints/pi05_droid_pytorch"
```

## Training

### Start Training

```bash
# Basic training
uv run loom train configs/pi05_minimal.yaml

# With Weights & Biases logging
uv run loom train configs/pi05_minimal.yaml --wandb
```

### Resume Training

```bash
uv run loom train configs/pi05_minimal.yaml \
    --resume checkpoints/pi05_minimal/checkpoint-epoch-50.pt
```

### Multi-GPU Training (Coming Soon)

```bash
# FSDP not yet supported in PyTorch implementation
# Use single GPU for now
```

## Monitoring

### Local Logging

Training logs are saved to:
- Console output: Real-time loss/metrics
- TensorBoard: `checkpoints/pi05_minimal/logs/`
- Checkpoints: `checkpoints/pi05_minimal/`

```bash
# View tensorboard
tensorboard --logdir checkpoints/pi05_minimal/logs/
```

### Weights & Biases

Enable in config:

```yaml
logging:
  use_wandb: true
  wandb_project: "policy-loom-pi05"
  wandb_run_name: "experiment-1"
```

## Inference

### Load Trained Model

```python
from loom.training.adapters.pi05 import Pi05Adapter
import torch

# Load config
config = {
    'action_dim': 32,
    'action_horizon': 10,
    'max_token_len': 256,
}

# Create adapter and model
adapter = Pi05Adapter(config)
model = adapter.create_model()

# Load checkpoint
checkpoint = torch.load('checkpoints/pi05_minimal/best_model.pt')
model.load_state_dict(checkpoint['model'])
model.eval()
```

### Run Inference

```python
import torch
from PIL import Image

# Prepare observation
observation = {
    'images': {
        'left': torch.tensor(...),   # (B, 3, 224, 224)
        'right': torch.tensor(...),
    },
    'tokenized_prompt': torch.tensor(...),  # (B, 256)
    'tokenized_prompt_mask': torch.tensor(...),  # (B, 256)
}

# Predict actions
with torch.no_grad():
    actions = model.sample_actions(
        device='cuda',
        observation=observation,
        num_steps=10  # Flow matching diffusion steps
    )
# actions shape: (B, action_horizon, 32)
```

## Troubleshooting

### Common Issues

#### 1. Import Error: "No module named 'openpi'"

**Solution**: Install Pi0.5 extras in separate venv:
```bash
python -m venv venv-pi05 && source venv-pi05/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
```

#### 2. Transformers Version Mismatch

**Error**: `check_whether_transformers_replace_is_installed_correctly() returns False`

**Solution**: Reapply patches:
```bash
cp -r .venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

#### 3. CUDA Out of Memory

**Solutions**:
- Reduce `batch_size` (try 4, 2, or 1)
- Increase `gradient_accumulation_steps`
- Use smaller model: `paligemma_variant: "gemma_2b"`
- Reduce `action_horizon`

#### 4. Action Dimension Warning

**Warning**: `OpenPI requires action_dim=32, but config has X`

**Explanation**: OpenPI hardcodes action_dim to 32. Actions are automatically padded/truncated.

**Solution**: This is expected. Ensure your action space can be mapped to 32 dimensions.

### Performance Tips

1. **Batch Size**: Start with 8, adjust based on GPU memory
2. **Learning Rate**: OpenPI default 1e-4 works well
3. **Image Size**: Must be 224x224 for PaliGemma
4. **Action Horizon**: 10-50 typical, smaller = faster training
5. **Precision**: bfloat16 uses less memory but may have higher losses than float32

## Architecture Details

### Data Flow

```
MP4/MCAP → LeRobot Dataset → DataLoader → Collate → OpenPITransform → Pi05Adapter → Training
```

1. **Reader**: Converts MP4/MCAP to LeRobot format
2. **DataLoader**: Batches samples with `collate_lerobot_batch`
3. **OpenPITransform**: Converts LeRobot batch to OpenPI format
4. **Pi05Adapter**: Implements training_step/eval_step
5. **Trainer**: Generic training loop

### Model Components

- **Vision Encoder**: SigLIP (PaliGemma's vision tower)
- **Language Model**: Gemma 2B/7B (PaliGemma's text decoder)
- **Action Expert**: Gemma 300M (flow matching decoder)
- **Tokenizer**: PaligemmaTokenizer (SentencePiece)

### Action Prediction

Pi0.5 uses **flow matching** instead of diffusion:
1. Sample noise: `z_T ~ N(0, I)`
2. Denoise with learned flow: `z_0 = f(z_T, t, observation)`
3. Predict actions: `a = z_0`

## References

- **OpenPI GitHub**: https://github.com/physical-intelligence/openpi
- **π0.5 Paper**: [Coming soon]
- **PaliGemma**: https://huggingface.co/google/paligemma-3b-pt-224
- **LeRobot**: https://github.com/huggingface/lerobot

## Support

For issues specific to Policy Loom Pi0.5 integration:
- GitHub Issues: https://github.com/advaita-labs/policy_loom/issues
- Documentation: https://github.com/advaita-labs/policy_loom/tree/main/docs

For OpenPI model questions:
- OpenPI Issues: https://github.com/physical-intelligence/openpi/issues
