# OpenPI Schema

## Overview

OpenPI (Open Policy Interface) is a standardized format for storing robotics demonstration data. It's designed for training Vision-Language-Action (VLA) models.

## Directory Structure

```
openpi_dataset/
├── manifest.json
├── episodes/
│   ├── episode_000/
│   │   ├── metadata.json
│   │   ├── observations.parquet
│   │   ├── actions.parquet
│   │   └── videos/
│   │       └── camera_0.mp4
│   ├── episode_001/
│   │   ├── metadata.json
│   │   ├── observations.parquet
│   │   ├── actions.parquet
│   │   └── videos/
│   │       ├── camera_0.mp4
│   │       └── camera_1.mp4
│   └── ...
└── statistics.json
```

## File Formats

### `manifest.json` (Dataset-level)

Top-level metadata about the entire dataset.

```json
{
  "version": "0.1.0",
  "created_at": "2025-01-15T10:30:00Z",
  "num_episodes": 100,
  "total_timesteps": 45000,
  "robot_type": "franka_panda",
  "action_space": {
    "type": "continuous",
    "dims": 7,
    "fields": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
  },
  "observation_space": {
    "cameras": ["camera_0"],
    "proprio_dims": 6,
    "proprio_fields": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
  }
}
```

### `episodes/episode_NNN/metadata.json` (Episode-level)

Per-episode metadata.

```json
{
  "episode_id": "episode_000",
  "length": 450,
  "duration_seconds": 45.0,
  "fps": 10,
  "task": "pick_and_place",
  "success": true,
  "source": "teleop",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### `observations.parquet`

Tabular data for observations (proprioception, timestamps).

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `float64` | Time in seconds |
| `step` | `int64` | Step index (0-based) |
| `proprio_0` | `float32` | First proprio dimension |
| `proprio_1` | `float32` | Second proprio dimension |
| ... | `float32` | Additional proprio dimensions |

**Example**:
```
timestamp,step,proprio_0,proprio_1,proprio_2,proprio_3,proprio_4,proprio_5
0.000,0,0.1,-0.2,0.3,0.0,0.0,1.57
0.100,1,0.11,-0.19,0.31,0.01,0.0,1.57
0.200,2,0.12,-0.18,0.32,0.02,0.0,1.57
```

### `actions.parquet`

Tabular data for actions.

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `float64` | Time in seconds |
| `step` | `int64` | Step index (0-based) |
| `action_0` | `float32` | First action dimension |
| `action_1` | `float32` | Second action dimension |
| ... | `float32` | Additional action dimensions |

**Example**:
```
timestamp,step,action_0,action_1,action_2,action_3,action_4,action_5,action_6
0.000,0,0.01,-0.01,0.0,0.0,0.0,0.0,1.0
0.100,1,0.01,-0.01,0.0,0.0,0.0,0.0,1.0
0.200,2,0.0,0.0,0.01,0.0,0.0,0.0,1.0
```

### `videos/camera_N.mp4`

Video files for each camera view.

**Properties**:
- **Codec**: H.264
- **FPS**: Matches episode fps (typically 10 or 30)
- **Resolution**: Preprocessed resolution (e.g., 224x224 or 480x640)
- **Pixel format**: YUV420p

## Data Types

- **Timestamps**: `float64`, seconds since episode start (0.0 = first frame)
- **Indices**: `int64`, 0-based
- **Proprio/Actions**: `float32`, normalized or raw (document in manifest)
- **Videos**: H.264 encoded, 8-bit color

## Validation Rules

1. **Synchronized lengths**: `len(observations) == len(actions) == num_video_frames`
2. **Monotonic timestamps**: `timestamp[i] < timestamp[i+1]`
3. **Step indices**: `step == range(len(observations))`
4. **No missing files**: All paths in manifest exist
5. **Valid parquet**: Files can be read with `pyarrow` or `pandas`

## Design Principles

- **Tabular + Video**: Separate tables for fast random access, videos for large blobs
- **Standard formats**: Parquet (efficient, typed), MP4 (widely supported)
- **Flat episodes**: One directory per episode (easy to shard, delete, copy)
- **Self-contained**: Each episode is independent (no cross-episode references)
- **Metadata everywhere**: Easy to filter/query without loading data
