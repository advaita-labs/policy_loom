# Pi0.5 Configuration Guide

Comprehensive guide to Pi0.5 training configuration parameters for policy_loom.

## Configuration Structure

```yaml
model:          # Model architecture and initialization
training:       # Training hyperparameters
data:           # Dataset configuration
checkpoints:    # Model checkpointing
evaluation:     # Evaluation frequency
logging:        # Logging and experiment tracking
```

---

## Model Configuration

### `type`
- **Type:** `string`
- **Required:** Yes
- **Value:** `"pi05"`
- **Description:** Specifies Pi0.5 model adapter
- **Example:** `type: pi05`

### `config_name`
- **Type:** `string`
- **Default:** `"pi05_libero"`
- **Options:** `pi05_libero`, `pi05_droid`, (pi05_aloha may exist)
- **Description:** OpenPI configuration variant. Each has preset architecture for specific robot platforms.
  - `pi05_libero`: For LIBERO simulation benchmark
  - `pi05_droid`: For DROID robot platform (Franka arm)
- **Impact:** **Overrides action_dim/action_horizon** with openpi's preset values
- **Example:** `config_name: pi05_libero`

⚠️ **Important:** openpi configs have fixed architectures. Your action_dim/action_horizon settings may be ignored!

### `action_dim`
- **Type:** `integer`
- **Default:** `7`
- **Range:** Platform-dependent
- **Description:** Robot action space dimensionality
  - LIBERO/Franka: 7 DoF (6 arm joints + 1 gripper)
  - ALOHA: 14 DoF (7 per arm)
- **Impact:** Must match your robot's control interface
- **May be overridden:** By openpi config_name
- **Example:** `action_dim: 7`

### `action_horizon`
- **Type:** `integer`
- **Default:** `10`
- **Range:** 1-50 typical
- **Description:** Number of future action steps model predicts per forward pass
  - Higher = smoother trajectories, more GPU memory
  - Lower = faster inference, less temporal coherence
- **Impact:**
  - Training: Affects loss computation and memory usage
  - Inference: Policy outputs N-step action sequence
- **May be overridden:** By openpi config_name
- **Example:** `action_horizon: 10`

📝 **Note:** LeRobot datasets provide single-step actions. policy_loom currently doesn't implement temporal windowing, so action_horizon > 1 requires the model to learn multi-step predictions from single-step supervision.

### `image_size`
- **Type:** `list[int, int]`
- **Default:** `[224, 224]`
- **Format:** `[height, width]`
- **Description:** Input image resolution for vision encoder
- **Impact:**
  - Larger = better visual detail, more GPU memory
  - Must match pretrained checkpoint if loading weights
- **Typical Values:**
  - `[224, 224]`: Standard for PaliGemma backbone
  - `[256, 256]`: Higher resolution option
- **Example:** `image_size: [224, 224]`

### `pretrained_path`
- **Type:** `string | null`
- **Default:** `null`
- **Format:** Local path or GCS URL
- **Description:** Path to pretrained Pi0.5 checkpoint for fine-tuning
  - `null`: Train from scratch (random initialization)
  - Local: `"./checkpoints/my_model"`
  - GCS: `"gs://openpi-assets/checkpoints/pi05_base"`
- **Available Checkpoints:**
  - `gs://openpi-assets/checkpoints/pi05_base`: Base model
  - `gs://openpi-assets/checkpoints/pi05_droid`: DROID-tuned
  - `gs://openpi-assets/checkpoints/pi05_libero`: LIBERO-tuned
- **Impact:** Fine-tuning is faster and needs less data than training from scratch
- **Example:** `pretrained_path: "gs://openpi-assets/checkpoints/pi05_base"`

### `use_lora`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enable LoRA (Low-Rank Adaptation) parameter-efficient fine-tuning
- **Status:** ⚠️ **Not yet implemented** in policy_loom Pi0.5 adapter
- **Benefits:** Reduces trainable parameters, GPU memory, training time
- **Example:** `use_lora: false`

### `lora_rank`
- **Type:** `integer`
- **Default:** `8`
- **Range:** 4-64 typical
- **Description:** LoRA rank (if use_lora=true)
  - Higher rank = more capacity but more parameters
- **Impact:** Trade-off between adaptation quality and efficiency
- **Example:** `lora_rank: 16`

### `freeze_backbone`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Freeze vision/language backbone parameters during training
  - Only trains action prediction head
- **Benefits:** Reduces GPU memory, prevents catastrophic forgetting
- **Use When:** Fine-tuning with small dataset (<1K episodes)
- **Example:** `freeze_backbone: true`

### `default_prompt`
- **Type:** `string | null`
- **Default:** `null`
- **Description:** Optional natural language task instruction
  - Conditions model behavior via language
  - Useful for multi-task learning
- **Examples:**
  - `"Pick up the red cube"`
  - `"Navigate to the goal"`
  - `null`: No language conditioning
- **Example:** `default_prompt: "Pick up the object"`

---

## Training Configuration

### `epochs`
- **Type:** `integer`
- **Default:** `50`
- **Description:** Number of complete passes through training dataset
- **Guidance:**
  - From scratch: 50-100 epochs
  - Fine-tuning: 10-30 epochs
- **Impact:** More epochs = better convergence but risk of overfitting
- **Example:** `epochs: 50`

### `batch_size`
- **Type:** `integer`
- **Default:** `8`
- **Range:** 1-32 typical (depends on GPU memory)
- **Description:** Number of samples per training batch
- **GPU Memory Usage:**
  - Batch 4: ~8GB VRAM
  - Batch 8: ~16GB VRAM
  - Batch 16: ~32GB VRAM
- **Impact:**
  - Larger = more stable gradients, faster training
  - Smaller = fits on smaller GPUs, may need lower learning rate
- **Example:** `batch_size: 8`

### `learning_rate`
- **Type:** `float`
- **Default:** `1e-4` (0.0001)
- **Range:** 1e-5 to 1e-3 typical
- **Description:** Optimizer step size
- **Guidance:**
  - From scratch: 1e-4
  - Fine-tuning: 5e-5 (lower to preserve pretrained weights)
  - With small batch: reduce proportionally
- **Example:** `learning_rate: 1e-4`

### `weight_decay`
- **Type:** `float`
- **Default:** `1e-5`
- **Range:** 0.0 to 1e-3
- **Description:** L2 regularization strength
- **Impact:** Prevents overfitting by penalizing large weights
- **Example:** `weight_decay: 1e-5`

### `gradient_clip_norm`
- **Type:** `float`
- **Default:** `1.0`
- **Range:** 0.1 to 10.0
- **Description:** Maximum gradient norm (prevents exploding gradients)
- **Impact:** Stabilizes training, especially early on
- **Example:** `gradient_clip_norm: 1.0`

### `num_workers`
- **Type:** `integer`
- **Default:** `4`
- **Range:** 0 to CPU cores
- **Description:** Number of parallel data loading workers
- **Guidance:**
  - 0: Single-threaded (debugging)
  - 4-8: Typical for training
  - Match CPU cores for maximum throughput
- **Impact:** Faster data loading reduces training time
- **Example:** `num_workers: 4`

### `lr_scheduler.type`
- **Type:** `string`
- **Default:** `"cosine"`
- **Options:** `constant`, `cosine`, `step`, `plateau`
- **Description:** Learning rate scheduling strategy
  - `constant`: No decay
  - `cosine`: Smooth cosine decay (recommended)
  - `step`: Step decay at intervals
  - `plateau`: Reduce on metric plateau
- **Example:** `type: cosine`

### `lr_scheduler.warmup_steps`
- **Type:** `integer`
- **Default:** `500`
- **Description:** Number of steps to linearly increase LR from 0 to target
- **Impact:** Prevents unstable early training
- **Guidance:**
  - Short datasets: 100-500 steps
  - Long datasets: 1000-5000 steps
- **Example:** `warmup_steps: 500`

### `lr_scheduler.min_lr`
- **Type:** `float`
- **Default:** `1e-6`
- **Description:** Minimum learning rate for cosine schedule
- **Impact:** Maintains slow learning at end of training
- **Example:** `min_lr: 1e-6`

---

## Data Configuration

### `type`
- **Type:** `string`
- **Required:** Yes
- **Options:** `lerobot`, `file`
- **Description:** Dataset type/format
  - `lerobot`: HuggingFace LeRobot format datasets
  - `file`: Local .pt files (legacy)
- **Example:** `type: lerobot`

### `dataset`
- **Type:** `string`
- **Required:** Yes (for type=lerobot)
- **Description:** HuggingFace dataset repository ID
- **Format:** `"username/dataset-name"` or `"org/dataset-name"`
- **Examples:**
  - `"lerobot/pusht"`: 2D pushing task
  - `"lerobot/aloha_sim_insertion_human"`: ALOHA bimanual task
  - `"lerobot/libero_spatial_no_noops"`: LIBERO benchmark
- **Example:** `dataset: "lerobot/pusht"`

### `train_split`
- **Type:** `string`
- **Default:** `"train"`
- **Description:** Dataset split for training
- **Example:** `train_split: train`

### `eval_split`
- **Type:** `string | null`
- **Default:** `null`
- **Description:** Dataset split for evaluation
  - `null`: No evaluation
  - `"test"` or `"val"`: Standard splits
- **Impact:** Enables periodic evaluation during training
- **Example:** `eval_split: test`

### `local_dir`
- **Type:** `string | null`
- **Default:** `null`
- **Description:** Local directory to cache downloaded datasets
- **Benefits:** Avoids re-downloading on subsequent runs
- **Example:** `local_dir: ./data/cache`

---

## Checkpoints Configuration

### `dir`
- **Type:** `string`
- **Default:** `./checkpoints/pi05`
- **Description:** Directory to save model checkpoints
- **Example:** `dir: ./checkpoints/pi05`

### `save_every_steps`
- **Type:** `integer | null`
- **Default:** `null`
- **Description:** Save checkpoint every N training steps
  - `null`: Only save based on epochs
- **Use When:** Very long training runs
- **Example:** `save_every_steps: 1000`

### `save_every_epochs`
- **Type:** `integer | null`
- **Default:** `5`
- **Description:** Save checkpoint every N epochs
- **Example:** `save_every_epochs: 5`

### `keep_top_k`
- **Type:** `integer`
- **Default:** `2`
- **Description:** Keep top K checkpoints by validation performance
- **Impact:** Saves disk space while keeping best models
- **Example:** `keep_top_k: 2`

### `keep_last_k`
- **Type:** `integer`
- **Default:** `1`
- **Description:** Keep last K checkpoints (most recent)
- **Impact:** Enables resuming from latest state
- **Example:** `keep_last_k: 1`

### `resume_from`
- **Type:** `string | null`
- **Default:** `null`
- **Description:** Path to checkpoint to resume training from
- **Example:** `resume_from: ./checkpoints/pi05/checkpoint_epoch_10.pt`

---

## Evaluation Configuration

### `eval_every_steps`
- **Type:** `integer | null`
- **Default:** `null`
- **Description:** Run evaluation every N training steps
- **Example:** `eval_every_steps: 500`

### `eval_every_epochs`
- **Type:** `integer | null`
- **Default:** `10`
- **Description:** Run evaluation every N epochs
- **Example:** `eval_every_epochs: 10`

---

## Logging Configuration

### `log_every_steps`
- **Type:** `integer`
- **Default:** `10`
- **Description:** Log metrics every N training steps
- **Impact:** More frequent = detailed logs but slower
- **Example:** `log_every_steps: 10`

### `log_dir`
- **Type:** `string`
- **Default:** `./logs/pi05`
- **Description:** Directory for TensorBoard/local logs
- **Example:** `log_dir: ./logs/pi05`

### `wandb.enabled`
- **Type:** `boolean`
- **Default:** `false`
- **Description:** Enable Weights & Biases experiment tracking
- **Example:** `enabled: true`

### `wandb.project`
- **Type:** `string`
- **Default:** `"robot_learning"`
- **Description:** W&B project name
- **Example:** `project: robot_learning`

### `wandb.name`
- **Type:** `string`
- **Default:** `"pi05_training"`
- **Description:** W&B run name
- **Example:** `name: pi05_pusht_experiment_1`

### `wandb.entity`
- **Type:** `string | null`
- **Default:** `null`
- **Description:** W&B team/user entity
- **Example:** `entity: my_org`

### `wandb.tags`
- **Type:** `list[string]`
- **Default:** `["pi05", "openpi"]`
- **Description:** Tags for organizing W&B experiments
- **Example:** `tags: ["pi05", "libero", "baseline"]`

### `wandb.notes`
- **Type:** `string`
- **Default:** `"Training Pi0.5 with openpi"`
- **Description:** Human-readable experiment notes
- **Example:** `notes: "Baseline run with default hyperparameters"`

---

## Common Configuration Patterns

### Training from Scratch (Small Dataset)
```yaml
model:
  config_name: pi05_libero
  pretrained_path: null
training:
  epochs: 100
  batch_size: 8
  learning_rate: 1e-4
```

### Fine-Tuning Pretrained Model
```yaml
model:
  config_name: pi05_droid
  pretrained_path: "gs://openpi-assets/checkpoints/pi05_base"
training:
  epochs: 20
  batch_size: 16
  learning_rate: 5e-5  # Lower for fine-tuning
```

### Memory-Constrained Training (Low GPU Memory)
```yaml
model:
  freeze_backbone: true  # Reduces memory
training:
  batch_size: 4  # Smaller batch
  gradient_clip_norm: 0.5
```

### Fast Iteration / Debugging
```yaml
training:
  epochs: 5
  batch_size: 4
  num_workers: 0  # Single-threaded for debugging
logging:
  log_every_steps: 1
checkpoints:
  save_every_epochs: 1
```

---

## Important Notes

### ⚠️ Config Overrides
openpi's `config_name` (pi05_libero, pi05_droid) has **preset architecture values** that **override** your yaml settings for:
- `action_dim`
- `action_horizon`
- Internal model architecture

**Workaround:** Match your yaml values to openpi defaults, or modify openpi config source code.

### 📦 Required Setup Steps
Before training, run:
```bash
# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05

# Patch transformers (CRITICAL!)
cp -r .venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

### 🔄 Action Horizon Limitation
LeRobot datasets provide **single-step actions**, but Pi0.5 is designed for **multi-step prediction**.

**Current Status:** policy_loom doesn't implement temporal windowing yet. With `action_horizon > 1`, the model must learn to predict sequences from single-step supervision (challenging but possible).

**Future:** Implement sliding window over action sequences from dataset.

---

## Troubleshooting

### Issue: "Config action_dim/horizon may override user settings"
**Cause:** openpi configs have fixed values
**Solution:** Check openpi source for actual values, adjust yaml to match

### Issue: "transformers_replace is not installed"
**Cause:** Missing transformers patch
**Solution:** Run the `cp` command from setup section above

### Issue: CUDA out of memory
**Solution:** Reduce `batch_size` or set `freeze_backbone: true`

### Issue: Training loss not decreasing
**Causes:**
1. Learning rate too high/low
2. Missing normalization stats
3. Action dimension mismatch
4. Dataset incompatible with model expectations

---

## References

- [openpi GitHub](https://github.com/Physical-Intelligence/openpi)
- [Pi0.5 Paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
