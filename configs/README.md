# Training Configuration Files

This directory contains example configuration files for running policy_loom training pipelines.

## Available Configurations

### DiffusionPolicy

| File | Description | Use Case |
|------|-------------|----------|
| `diffusion_policy_example.yaml` | Comprehensive config with detailed comments | Reference for all available settings |
| `diffusion_minimal.yaml` | Minimal working configuration | Quick start on simple datasets |

## Quick Start

```bash
# Activate diffusion environment
source venv-diffusion/bin/activate

# Train with minimal config
python scripts/train_diffusion_eval_t10.py --config configs/diffusion_minimal.yaml

# Or customise a copy of the example config
cp configs/diffusion_policy_example.yaml configs/my_diffusion_config.yaml
# Edit configs/my_diffusion_config.yaml, then run:
python scripts/train_diffusion_eval_t10.py --config configs/my_diffusion_config.yaml
```

## Data Format Examples

Configuration files support three primary dataset formats.

### 1. LeRobot Datasets (Recommended)

```yaml
data:
  type: lerobot
  dataset: "lerobot/koch_test"
  train_split: train
  eval_split: test
```

**Dataset structure:** automatically downloaded from the HuggingFace Hub.  
Browse available datasets at <https://huggingface.co/lerobot>.

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

**Directory layout**

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

**Directory layout**

```
data/
├── train_videos/
│   ├── episode_001/
│   │   ├── camera_front.mp4
│   │   ├── camera_wrist.mp4
│   │   ├── state.npy      # Shape: (num_timesteps, state_dim)
│   │   └── action.npy     # Shape: (num_timesteps, action_dim)
│   ├── episode_002/
│   └── ...
└── eval_videos/
    └── ...
```

**Requirements**
- `state.npy`: `(T, state_dim)` proprioceptive states.
- `action.npy`: `(T, action_dim)` actions aligned with states.
- Each episode’s video frames must align in count/timing with state/action arrays.
- Frame rate should match the control frequency.

## Key Configuration Sections

### Model Configuration

```yaml
model:
  type: diffusion_policy
  obs_dim: 7
  action_dim: 7
  action_horizon: 8
  hidden_dim: 256
```

### Training Parameters

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  lr_scheduler:
    type: cosine
    warmup_steps: 1000
```

### Checkpointing

```yaml
checkpoints:
  dir: ./checkpoints/my_model
  save_every_steps: 1000
  keep_top_k: 3
  resume_from: null
```

### Logging & Monitoring

```yaml
logging:
  log_every_steps: 10
  wandb:
    enabled: true
    project: robot_learning
    name: my_experiment
```

## Tips for Creating Your Own Configs

1. **Start from Minimal Configs** – copy `diffusion_minimal.yaml` and modify only what you need.
2. **Adjust Batch Size for Your Hardware** – e.g. 4–8 on Apple M-series, 32–64 on RTX 3090, 128+ on A100.
3. **Tune Learning Rate** – diffusion policies typically use `1e-4` to `1e-3`.
4. **Use Gradient Accumulation** when GPU memory is limited:
   ```yaml
   training:
     batch_size: 8
     gradient_accumulation_steps: 4  # Effective batch size = 32
   ```
5. **Enable W&B Tracking** to monitor experiments over time.

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `batch_size`
2. Increase `gradient_accumulation_steps`
3. Lower `num_workers`
4. Enable mixed precision (`mixed_precision: "fp16"`)
5. Reduce model size (e.g. smaller `hidden_dim`)

### Training Too Slow
1. Increase `batch_size` if memory allows
2. Increase `num_workers`
3. Reduce logging frequency (`log_every_steps`)
4. Consider enabling `torch_compile`

### Not Converging
1. Lower `learning_rate`
2. Increase `warmup_steps`
3. Try a different `lr_scheduler`
4. Train for more epochs or collect more data
5. Double-check preprocessing and data alignment

## Additional Resources

- Example scripts: see `scripts/train_diffusion_eval_t10.py`
- Testing instructions: `docs/TESTING_GUIDE.md`
- LeRobot datasets: <https://huggingface.co/lerobot>
