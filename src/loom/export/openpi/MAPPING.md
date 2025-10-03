# Sample → OpenPI Mapping

## Overview

This document defines how canonical `Sample` objects map to OpenPI files.

## Field Mapping

### Sample → observations.parquet

| Sample Field | OpenPI Column | Type | Notes |
|--------------|---------------|------|-------|
| `timestamp` | `timestamp` | `float64` | Normalized to start at 0.0 |
| `metadata['frame_idx']` | `step` | `int64` | Sequential index |
| `proprio[0]` | `proprio_0` | `float32` | First proprio dimension |
| `proprio[1]` | `proprio_1` | `float32` | Second proprio dimension |
| ... | ... | `float32` | Additional dimensions |

**Column names**: `proprio_{i}` where `i` is 0-based index.

If `metadata['proprio_spec']` exists, use those names instead:
- `metadata['proprio_spec'] = ["joint_0", "joint_1", ...]`
- Columns: `joint_0`, `joint_1`, ...

### Sample → actions.parquet

| Sample Field | OpenPI Column | Type | Notes |
|--------------|---------------|------|-------|
| `timestamp` | `timestamp` | `float64` | Normalized to start at 0.0 |
| `metadata['frame_idx']` | `step` | `int64` | Sequential index |
| `action[0]` | `action_0` | `float32` | First action dimension |
| `action[1]` | `action_1` | `float32` | Second action dimension |
| ... | ... | `float32` | Additional dimensions |

**Column names**: `action_{i}` where `i` is 0-based index.

If `metadata['action_spec']` exists, use those names instead:
- `metadata['action_spec'] = ["dx", "dy", "dz", "gripper"]`
- Columns: `dx`, `dy`, `dz`, `gripper`

### Sample → videos/camera_0.mp4

| Sample Field | Video Property | Notes |
|--------------|----------------|-------|
| `rgb` | Frame data | Encoded as H.264 |
| `timestamp` | Frame PTS | Presentation timestamp |
| `metadata['fps']` | Video FPS | Default: 10 |

**Encoding**:
- Codec: H.264
- Pixel format: YUV420p (8-bit)
- CRF: 23 (high quality)
- Preset: medium

### Sample → metadata.json

| Sample Field | Metadata Key | Notes |
|--------------|-------------|-------|
| `metadata['episode_id']` | `episode_id` | Required |
| `metadata['fps']` | `fps` | Default: 10 |
| `metadata['task']` | `task` | Optional task label |
| `metadata['source']` | `source` | Data source ("mp4", "mcap", etc.) |
| - | `length` | Computed: number of steps |
| - | `duration_seconds` | Computed: last_timestamp - first_timestamp |
| - | `timestamp` | ISO 8601 creation time |

## Timestamp Normalization

**Problem**: Input timestamps may be Unix epoch (large numbers) or relative (small numbers).

**Solution**: Normalize to episode-relative timestamps starting at 0.0.

```python
def normalize_timestamps(samples: list[Sample]) -> list[Sample]:
    t0 = samples[0].timestamp
    for sample in samples:
        sample.timestamp -= t0
    return samples
```

**Result**:
- First sample: `timestamp = 0.0`
- Last sample: `timestamp = duration_seconds`

## Missing Fields

### No `rgb`

**Behavior**: Skip video creation, write observations and actions only.

```
episode_000/
├── metadata.json
├── observations.parquet
└── actions.parquet
```

### No `proprio`

**Behavior**: Write only `timestamp` and `step` columns.

```
timestamp,step
0.000,0
0.100,1
```

### No `action`

**Behavior**: Skip `actions.parquet` entirely (observation-only dataset).

```
episode_000/
├── metadata.json
├── observations.parquet
└── videos/
    └── camera_0.mp4
```

## Multi-Camera Mapping

If `metadata['camera_name']` is present, use it for video filename:

```python
camera_name = sample.metadata.get('camera_name', 'camera_0')
video_path = episode_dir / "videos" / f"{camera_name}.mp4"
```

**Example**:
- `sample.metadata['camera_name'] = "front"` → `videos/front.mp4`
- `sample.metadata['camera_name'] = "wrist"` → `videos/wrist.mp4`

## Data Type Conversion

| Sample Type | OpenPI Type | Conversion |
|-------------|-------------|------------|
| `np.uint8` | `float32` | Normalize to [0, 1]: `value / 255.0` |
| `np.float32` | `float32` | Direct copy |
| `np.float64` | `float64` (timestamps only) | Direct copy |
| `int` (timestamp) | `float64` | Convert to seconds: `value / 1e9` (if nanoseconds) |

## Manifest Generation

After writing all episodes, generate `manifest.json`:

```python
manifest = {
    "version": "0.1.0",
    "created_at": datetime.utcnow().isoformat() + "Z",
    "num_episodes": num_episodes,
    "total_timesteps": sum(episode_lengths),
    "robot_type": config.get("robot_type", "unknown"),
    "action_space": infer_action_space(first_episode),
    "observation_space": infer_observation_space(first_episode),
}
```

## Example: Complete Mapping

**Input Sample**:
```python
Sample(
    timestamp=1234567890.123,
    rgb=np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    proprio=np.array([0.1, -0.2, 0.3, 0.0, 0.0, 1.57], dtype=np.float32),
    action=np.array([0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    metadata={
        "episode_id": "ep001",
        "frame_idx": 42,
        "fps": 10,
        "camera_name": "front",
        "proprio_spec": ["j0", "j1", "j2", "j3", "j4", "j5"],
        "action_spec": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
    }
)
```

**Output**:
- **observations.parquet**: `timestamp=42.0, step=42, j0=0.1, j1=-0.2, j2=0.3, j3=0.0, j4=0.0, j5=1.57`
- **actions.parquet**: `timestamp=42.0, step=42, dx=0.01, dy=-0.01, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, gripper=1.0`
- **videos/front.mp4**: Frame 42 encoded as H.264
- **metadata.json**: `{"episode_id": "ep001", "fps": 10, ...}`
