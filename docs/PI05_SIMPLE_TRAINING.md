# Pi0.5 Training - Simplified Approach

## Overview

Pi0.5 training in policy_loom uses OpenPI's native training system directly. **No complex adapters or converters needed** - OpenPI already handles everything!

## Quick Start

```bash
# Install with Pi0.5 support
uv sync --extra pi05

# Train on any LeRobot dataset
uv run python -m loom.cli train-pi05 physical-intelligence/libero

# With custom settings
uv run python -m loom.cli train-pi05 gauravpradeep/t02_piper_pick_and_place_bimanual \
    --config-name pi05_libero \
    --batch-size 256 \
    --steps 30000 \
    --lr 5e-5
```

## What OpenPI Handles Automatically

When you use `train-pi05`, OpenPI takes care of:

- ✅ **LeRobot dataset loading** from HuggingFace Hub
- ✅ **Action padding** to 32 dimensions (Pi0.5 requirement)
- ✅ **Image preprocessing** and normalization
- ✅ **Multi-camera handling** and transforms
- ✅ **Observation formatting** into Pi0.5's expected format
- ✅ **Action sequencing** (action horizons)
- ✅ **Batch collation** and data loading

You don't need to write any of this - it's built into OpenPI!

## Available Configs

OpenPI provides pre-configured setups for different robots:

- **`pi05_libero`** - LIBERO simulation (7-dim actions)
- **`pi05_droid`** - DROID robot (7-dim actions)
- **`pi0_aloha_real`** - ALOHA bimanual robot (14-dim actions)
- **`pi0_fast_libero`** - Fast variant for LIBERO

Use `--config-name` to select.

## Action Dimension Handling

**Important**: Pi0.5's model architecture hardcodes action_dim=32. OpenPI automatically:

1. **Pads** your dataset actions to 32 dimensions (fills with zeros)
2. **Validates** that your dataset has ≤ 32 action dimensions
3. **Errors** if your dataset has > 32 dimensions

Example:
- Dataset has 7-dim actions (bimanual pick)
- OpenPI pads to [action₁, ..., action₇, 0, 0, ..., 0] (32 dims total)
- Model trains on 32-dim actions
- At inference, you use first 7 dimensions

## Command Options

```bash
uv run python -m loom.cli train-pi05 DATASET [OPTIONS]
```

**Arguments:**
- `DATASET` - HuggingFace LeRobot dataset repo (e.g., `physical-intelligence/libero`)

**Options:**
- `--config-name` - OpenPI config (default: `pi05_libero`)
- `--batch-size` - Training batch size (default: 256)
- `--steps` - Number of training steps (default: 30000)
- `--output-dir` - Checkpoint save directory (default: `./checkpoints`)
- `--lr` - Learning rate (default: uses OpenPI config's learning rate)

## Examples

### Train on LIBERO dataset

```bash
uv run python -m loom.cli train-pi05 physical-intelligence/libero \
    --config-name pi05_libero \
    --steps 30000
```

### Train on custom bimanual dataset

```bash
uv run python -m loom.cli train-pi05 gauravpradeep/t02_piper_pick_and_place_bimanual \
    --config-name pi05_libero \
    --batch-size 128 \
    --steps 50000 \
    --lr 1e-4
```

### Quick test run (2 steps)

```bash
uv run python -m loom.cli train-pi05 physical-intelligence/libero \
    --steps 2 \
    --batch-size 2
```

## Checkpoints

Checkpoints are saved to `--output-dir` (default: `./checkpoints`):

- `checkpoint_step_1000.pt` - Every 1000 steps
- `checkpoint_step_2000.pt`
- ...
- `final_checkpoint.pt` - Final model

Each checkpoint contains:
```python
{
    'step': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'loss': float,
    'config': OpenPI TrainConfig,
}
```

## Loading Checkpoints

```python
import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

# Load checkpoint
ckpt = torch.load('checkpoints/final_checkpoint.pt')

# Recreate model
model = PI0Pytorch(ckpt['config'].model)
model.load_state_dict(ckpt['model_state_dict'])
```

## GPU Requirements

- **Minimum**: 24GB VRAM (RTX 3090, A5000)
- **Recommended**: 48GB VRAM (A6000, A40)
- **Model size**: ~3.6B parameters for Pi0.5

Smaller GPUs: Consider using `pi0_fast` configs with LoRA fine-tuning.

## Troubleshooting

### Dataset not found

```
❌ Data loader creation failed: Dataset 'xyz' not found
```

**Solution**: Check dataset exists on HuggingFace Hub and is in LeRobot format.

### OOM (Out of Memory)

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce `--batch-size` (try 128, 64, 32)
- OpenPI handles optimizations automatically
- Use LoRA fine-tuning configs (e.g., `pi0_libero_low_mem_finetune`)

### Transformers patches not installed

```
ValueError: transformers_replace is not installed correctly
```

**Solution**: OpenPI requires custom transformers patches:
```bash
# Reinstall with patches
uv sync --extra pi05

# Verify
python -c "from openpi.models_pytorch import pi0_pytorch"
```

### Dataset format incompatible

```
TypeError: stack(): argument 'tensors' must be tuple of Tensors
```

**Solution**: Dataset may not be properly formatted for LeRobot. Check:
- Dataset has required fields: `observation`, `action`, `images`
- Timestamps are in correct format
- Images are properly encoded

## Advanced: Using OpenPI Directly

For more control, you can use OpenPI's training system directly:

```python
import dataclasses
from openpi.training import config as openpi_config
from openpi.training.data_loader import create_data_loader
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import torch

# Load config
config = openpi_config.get_config("pi05_libero")

# Override settings
config = dataclasses.replace(
    config,
    data=dataclasses.replace(
        config.data,
        repo_id="your/dataset"
    ),
    batch_size=256,
    num_train_steps=30000,
)

# Create model and data loader
model = PI0Pytorch(config.model)
data_loader = create_data_loader(config, framework="pytorch")

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for observation, actions in data_loader:
    loss = model.forward(observation, actions)
    loss.backward()
    optimizer.step()
```

See OpenPI documentation for full training options: https://github.com/physical-intelligence/openpi

## Why This Approach?

Previously, policy_loom built custom adapters, converters, and transforms for Pi0.5. This was **unnecessary complexity** because:

1. OpenPI already has LeRobot dataset support
2. OpenPI handles all preprocessing and transforms
3. OpenPI's API is simple and well-tested
4. Less code for us to maintain

Now we just provide a thin CLI wrapper around OpenPI's excellent training system!
