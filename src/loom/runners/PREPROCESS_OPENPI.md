# Preprocess → OpenPI Runner

## Purpose

The primary runner for `policy_loom`: convert raw data (mp4, mcap, image directories) into OpenPI format for VLA training.

## Usage

### Command Line

```bash
# From config file
uv run python -m loom.runners.preprocess --config configs/preprocess_mp4_to_openpi.yaml

# Inline arguments
uv run python -m loom.runners.preprocess \
    --input-type mp4 \
    --input-path ./data/demo.mp4 \
    --output-path ./output/demo_processed \
    --fps 10 \
    --resolution 224x224

# Multiple files
uv run python -m loom.runners.preprocess \
    --input-type mp4 \
    --input-path "./data/*.mp4" \
    --output-path ./output/all_demos \
    --fps 10
```

### Python API

```python
from loom.runners import PreprocessRunner

runner = PreprocessRunner(
    input_type="mp4",
    input_path="./data/demo.mp4",
    output_path="./output/demo_processed",
    transforms=[
        {"type": "resample_fps", "target_fps": 10},
        {"type": "resize", "height": 224, "width": 224},
        {"type": "normalize", "preset": "imagenet"},
    ],
)

manifest = runner.run()
print(f"Processed {manifest['num_episodes']} episodes")
```

## Configuration

### YAML Config Format

```yaml
# preprocess_mp4_to_openpi.yaml
input:
  type: mp4  # mp4, mcap, images
  path: "./data/demo.mp4"
  # type-specific options
  fps: 30  # original fps (for mp4)

transforms:
  - type: resample_fps
    target_fps: 10
  - type: resize
    height: 224
    width: 224
  - type: normalize
    preset: imagenet

output:
  type: openpi
  path: "./output/demo_processed"
  overwrite: false  # fail if output exists
  validate: true    # validate output after writing

metadata:
  robot_type: "franka_panda"
  task: "pick_and_place"
  source: "teleop"

logging:
  level: INFO
  file: "./logs/preprocess.log"
```

## Inputs

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `input.type` | str | Yes | - | Data source type ("mp4", "mcap", "images") |
| `input.path` | str | Yes | - | Path to input file(s) (supports globs) |
| `output.path` | str | Yes | - | Output directory for OpenPI dataset |
| `transforms` | list | No | `[]` | List of transforms to apply |
| `output.overwrite` | bool | No | `false` | Overwrite existing output |
| `output.validate` | bool | No | `true` | Validate output schema |
| `metadata.*` | any | No | - | Additional metadata for manifest |

## Outputs

### Directory Structure

```
{output.path}/
├── manifest.json
├── episodes/
│   ├── episode_000000/
│   ├── episode_000001/
│   └── ...
└── logs/
    └── preprocess_20250115_103000.log
```

### Manifest Contents

```json
{
  "version": "0.1.0",
  "created_at": "2025-01-15T10:30:00Z",
  "input_path": "./data/demo.mp4",
  "input_type": "mp4",
  "output_path": "./output/demo_processed",
  "num_episodes": 1,
  "total_timesteps": 450,
  "config_hash": "a1b2c3d4",
  "transforms": [
    {"type": "resample_fps", "target_fps": 10},
    {"type": "resize", "height": 224, "width": 224},
    {"type": "normalize", "preset": "imagenet"}
  ],
  "metadata": {
    "robot_type": "franka_panda",
    "task": "pick_and_place"
  },
  "statistics": {
    "processing_time_seconds": 42.3,
    "samples_per_second": 10.6,
    "dropped_samples": 0,
    "errors": 0
  },
  "versions": {
    "policy_loom": "0.1.0",
    "python": "3.10.13",
    "numpy": "1.26.0"
  }
}
```

## Failure Policies

### Input Errors

| Error | Behavior | Exit Code |
|-------|----------|-----------|
| File not found | Abort, log error | 1 |
| Malformed file | Skip file, continue | 0 (with warning) |
| Unsupported format | Abort, log error | 1 |

### Transform Errors

| Error | Behavior | Exit Code |
|-------|----------|-----------|
| Invalid sample | Skip sample, log warning | 0 |
| Transform exception | Abort pipeline | 1 |

### Output Errors

| Error | Behavior | Exit Code |
|-------|----------|-----------|
| Disk full | Abort, clean up temp files | 1 |
| Permission denied | Abort, log error | 1 |
| Output exists (no overwrite) | Abort, log error | 1 |
| Validation failed | Abort, keep output for debugging | 1 |

## Validation

After processing, the runner validates:
- ✅ All episodes have required files (metadata.json, observations.parquet, etc.)
- ✅ Parquet files are readable
- ✅ Video files have matching frame counts
- ✅ Timestamps are monotonic
- ✅ No missing steps

If validation fails, the runner exits with code 1 and logs the errors.

## Progress Tracking

```
Processing: ./data/demo.mp4
  Reading frames... 100% ━━━━━━━━━━━━━━━━━━━━ 450/450 [00:05<00:00, 85.2 frames/s]
  Applying transforms...
    ✓ resample_fps (10 fps)
    ✓ resize (224x224)
    ✓ normalize (imagenet)
  Writing episode_000000... ━━━━━━━━━━━━━━━━━━━━ 450/450 [00:08<00:00, 53.1 samples/s]

Summary:
  Episodes: 1
  Timesteps: 450
  Duration: 45.0s
  Processing time: 13.2s
  Throughput: 34.1 samples/s
  Output: ./output/demo_processed
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOOM_NUM_WORKERS` | 1 | Number of parallel workers |
| `LOOM_BUFFER_SIZE` | 1000 | Writer buffer size |
| `LOOM_TEMP_DIR` | `/tmp/loom` | Temporary file directory |
| `LOOM_LOG_LEVEL` | `INFO` | Logging level |

## Examples

### Single MP4 File

```bash
uv run python -m loom.runners.preprocess \
    --config configs/preprocess_mp4_to_openpi.yaml
```

### Multiple MCAP Files

```bash
uv run python -m loom.runners.preprocess \
    --input-type mcap \
    --input-path "./data/*.mcap" \
    --output-path ./output/all_mcap \
    --config configs/mcap_topic_map.yaml
```

### Image Directory

```bash
uv run python -m loom.runners.preprocess \
    --input-type images \
    --input-path "./data/episode_001/" \
    --output-path ./output/episode_001_processed \
    --fps 10
```

## Troubleshooting

**Issue**: "Output directory already exists"
**Solution**: Use `--overwrite` flag or delete output directory

**Issue**: "Transform failed: Invalid image shape"
**Solution**: Check input data, ensure RGB images are (H, W, 3)

**Issue**: "Disk full error"
**Solution**: Free up disk space or use a different output path

**Issue**: "Validation failed: Missing observations.parquet"
**Solution**: Check logs for write errors, may need to rerun
