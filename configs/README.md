# Training Configuration Files

This directory contains example configuration files for training different VLA models with policy_loom.

## Available Configurations

### DiffusionPolicy

| File | Description | Use Case |
|------|-------------|----------|
| `diffusion_policy_example.yaml` | Comprehensive config with all options and comments | Reference for all available settings |
| `diffusion_minimal.yaml` | Minimal working configuration | Quick start, simple datasets |

### Pi0.5 (Physical Intelligence VLA)

| File | Description | Use Case |
|------|-------------|----------|
| `pi05_example.yaml` | Comprehensive config with all options and comments | Reference for all available settings |
| `pi05_minimal.yaml` | Minimal working configuration | Quick start, fine-tuning |

## Quick Start

### Training DiffusionPolicy

```bash
# Activate diffusion environment
source .venv-diffusion/bin/activate

# Using minimal config
python scripts/train_diffusion_eval_t10.py --config configs/diffusion_minimal.yaml

# Or create your own config based on the example
cp configs/diffusion_policy_example.yaml configs/my_diffusion_config.yaml
# Edit my_diffusion_config.yaml
python scripts/train_diffusion_eval_t10.py --config configs/my_diffusion_config.yaml
```

### Training Pi0.5

```bash
# Activate pi05 environment
source venv-pi05/bin/activate

# Using minimal config
python scripts/train_pi05.py --config configs/pi05_minimal.yaml

# Or create your own config based on the example
cp configs/pi05_example.yaml configs/my_pi05_config.yaml
# Edit my_pi05_config.yaml
python scripts/train_pi05.py --config configs/my_pi05_config.yaml
```

## Data Format Examples

The configuration files support three main data formats:

### 1. LeRobot Datasets (Recommended)

```yaml
data:
  type: lerobot
  dataset: "lerobot/koch_test"  # HuggingFace dataset ID
  train_split: train
  eval_split: test
```

**Dataset structure:** Automatically downloaded from HuggingFace Hub.

**Browse datasets:** https://huggingface.co/lerobot

### 2. MCAP Datasets (ROS2 bags)

```yaml
data:
  type: mcap
  train_path: "data/train_recordings/*.mcap"
  eval_path: "data/eval_recordings/*.mcap"
  topics:
    cameras:
      - name: wrist_camera
        topic: /camera/wrist/image_raw
    state_topic: /robot/joint_states
    action_topic: /robot/action
```

**Directory structure:**
```
data/
├── train_recordings/
│   ├── demo_001.mcap
│   ├── demo_002.mcap
│   └── ...
└── eval_recordings/
    ├── eval_001.mcap
    └── ...
```

### 3. MP4 Datasets (Video files)

```yaml
data:
  type: mp4
  train_path: "data/train_videos"
  eval_path: "data/eval_videos"
  camera_names: ["camera_front", "camera_wrist"]
  state_filename: "state.npy"
  action_filename: "action.npy"
```

**Directory structure:**
```
data/
├── train_videos/
│   ├── episode_001/
│   │   ├── camera_front.mp4
│   │   ├── camera_wrist.mp4
│   │   ├── state.npy      # Shape: (num_timesteps, state_dim)
│   │   ├── action.npy     # Shape: (num_timesteps, action_dim)
│   │   └── instruction.txt  # (Pi0.5 only) Task description
│   ├── episode_002/
│   └── ...
└── eval_videos/
    └── ...
```

**Data format requirements:**
- `state.npy`: NumPy array of shape `(T, state_dim)` with proprioceptive states (joint positions, velocities, etc.)
- `action.npy`: NumPy array of shape `(T, action_dim)` with robot actions
- Video files: Must have same number of frames as state/action arrays
- Frame rate: Should match the robot control frequency

## Key Configuration Sections

### Model Configuration

Defines the model architecture and hyperparameters:

```yaml
model:
  type: diffusion_policy  # or "pi05"
  obs_dim: 7             # Observation dimension
  action_dim: 7          # Action dimension
  action_horizon: 8      # Number of future actions to predict
  hidden_dim: 256        # Neural network hidden dimension
```

### Training Parameters

Controls the training process:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  lr_scheduler:
    type: cosine          # Learning rate schedule
    warmup_steps: 1000
```

### Checkpointing

Manages model checkpoints:

```yaml
checkpoints:
  dir: ./checkpoints/my_model
  save_every_steps: 1000
  keep_top_k: 3          # Keep best 3 checkpoints
  resume_from: null       # Path to resume from
```

### Logging & Monitoring

Configure logging and W&B integration:

```yaml
logging:
  log_every_steps: 10
  wandb:
    enabled: true
    project: robot_learning
    name: my_experiment
```

## Tips for Creating Your Own Configs

### 1. Start from Minimal Configs

Copy one of the minimal configs and modify only what you need:

```bash
cp configs/diffusion_minimal.yaml configs/my_config.yaml
```

### 2. Adjust Batch Size for Your Hardware

| Hardware | DiffusionPolicy Batch Size | Pi0.5 Batch Size |
|----------|---------------------------|------------------|
| Apple M1/M2 (16GB) | 4-8 | 2-4 |
| RTX 3090 (24GB) | 32-64 | 8-16 |
| A100 (40GB) | 128-256 | 32-64 |

### 3. Tune Learning Rate

- **DiffusionPolicy:** Typically `1e-4` to `1e-3`
- **Pi0.5 (fine-tuning):** Lower rates like `1e-5` to `1e-4`
- **Pi0.5 (frozen backbone):** Higher rates like `5e-4`

### 4. Use Gradient Accumulation for Larger Effective Batch Size

If your GPU memory is limited, use gradient accumulation:

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size = 32
```

### 5. Enable W&B for Experiment Tracking

Weights & Biases is highly recommended for tracking experiments:

```yaml
logging:
  wandb:
    enabled: true
    project: my_project
    name: experiment_001
    tags: ["diffusion_policy", "real_robot"]
```

## Common Configuration Patterns

### Quick Testing Setup (Fast iteration)

```yaml
training:
  epochs: 1
  batch_size: 4
  num_workers: 0  # For debugging
checkpoints:
  save_every_steps: 100
evaluation:
  eval_every_epochs: 1
logging:
  log_every_steps: 1
  wandb:
    enabled: false
```

### Production Training Setup

```yaml
training:
  epochs: 100
  batch_size: 64
  gradient_accumulation_steps: 2
checkpoints:
  save_every_steps: 5000
  keep_top_k: 3
evaluation:
  eval_every_epochs: 5
logging:
  log_every_steps: 50
  wandb:
    enabled: true
```

### Memory-Constrained Setup

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8
  num_workers: 2
  mixed_precision: "fp16"  # Pi0.5 only
model:
  freeze_backbone: true  # Pi0.5 only
checkpoints:
  keep_top_k: 1
  keep_last_k: 1
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `num_workers`
4. Enable `mixed_precision: "fp16"` (Pi0.5)
5. Set `freeze_backbone: true` (Pi0.5)
6. Reduce `hidden_dim` or `action_horizon`

### Training Too Slow

1. Increase `batch_size` (if memory allows)
2. Increase `num_workers`
3. Enable `mixed_precision: "fp16"` (Pi0.5)
4. Reduce logging frequency: increase `log_every_steps`
5. Enable `torch_compile: true` (Pi0.5, advanced)

### Not Converging

1. Lower `learning_rate`
2. Increase `warmup_steps`
3. Try different `lr_scheduler` (e.g., cosine vs linear)
4. Increase `epochs` or dataset size
5. Check data quality and preprocessing
6. For Pi0.5: try `freeze_backbone: true` first

## Additional Resources

- **Example scripts:** See `scripts/train_*.py` for training script examples
- **Documentation:** `docs/PI05_TRAINING.md` for Pi0.5-specific guidance
- **Testing:** `docs/TESTING_GUIDE.md` for environment setup and testing
- **LeRobot datasets:** https://huggingface.co/lerobot
- **Physical Intelligence:** https://www.physicalintelligence.company

## Contributing

When adding new configuration options:

1. Add them to the `*_example.yaml` files with detailed comments
2. Update this README with usage examples
3. Test the configuration with actual training runs
4. Document any hardware-specific considerations
