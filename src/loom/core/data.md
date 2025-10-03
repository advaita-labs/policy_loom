# Canonical Sample Format

## Overview

The `Sample` dataclass is the unified representation that flows through the entire `policy_loom` pipeline. Every component (readers, transforms, writers) operates on `Sample` objects.

## Fields

### `timestamp: float | int`
**Required**

Time at which this sample was captured. Can be:
- **float**: Seconds since epoch (Unix timestamp)
- **int**: Nanoseconds since epoch (for high-precision timing)

All samples in a stream should use consistent units.

### `rgb: NDArray[uint8] | NDArray[float32] | None`
**Optional**

Image data with shape `(H, W, C)` where:
- `H`: Height in pixels
- `W`: Width in pixels
- `C`: Channels (typically 3 for RGB, 1 for grayscale)

**Dtype conventions:**
- `uint8`: Raw pixel values [0, 255]
- `float32`: Normalized values [0.0, 1.0] or [-1.0, 1.0] (document in metadata)

### `proprio: NDArray[float32] | None`
**Optional**

Proprioceptive sensor data as 1D array. Common contents:
- Joint positions (rad or deg)
- Joint velocities (rad/s)
- End-effector pose (x, y, z, roll, pitch, yaw)
- Gripper state (open/close percentage)

**Important:** Document the exact fields and units in `metadata['proprio_spec']`.

### `action: NDArray[float32] | None`
**Optional**

Action commanded at this timestep as 1D array. Structure depends on robot:
- Joint space actions (delta or absolute)
- Cartesian space actions (dx, dy, dz, droll, dpitch, dyaw)
- Gripper command

**Important:** Document the action space in `metadata['action_spec']`.

### `metadata: dict[str, Any]`
**Optional**

Arbitrary metadata. Common keys:

- `episode_id: str` - Unique episode identifier
- `frame_idx: int` - Frame number within episode
- `source: str` - Data source identifier ("mp4", "mcap", "teleop")
- `proprio_spec: list[str]` - Names of proprio fields (e.g., `["joint_0", "joint_1", ...]`)
- `action_spec: list[str]` - Names of action dimensions
- `camera_name: str` - Which camera captured the RGB image
- `dropped_frames: int` - Number of dropped frames before this sample
- `sync_error_ms: float` - Synchronization error between modalities

## Design Principles

1. **Minimal but sufficient**: Only fields that >80% of VLA models need
2. **Validation at construction**: `__post_init__` catches shape errors early
3. **Typed arrays**: Use numpy for efficient processing and type safety
4. **Metadata escape hatch**: Uncommon fields go in metadata, not new attributes
5. **No dependencies**: Only numpy (ubiquitous in ML)

## Example Usage

```python
from loom.core import Sample
import numpy as np

# Video-only sample
sample = Sample(
    timestamp=1234567890.123,
    rgb=np.zeros((480, 640, 3), dtype=np.uint8),
    metadata={"episode_id": "ep001", "frame_idx": 42}
)

# Full robotics sample
sample = Sample(
    timestamp=1234567890.456,
    rgb=np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    proprio=np.array([0.1, -0.2, 0.3, 0.0, 0.0, 1.57], dtype=np.float32),
    action=np.array([0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    metadata={
        "episode_id": "ep002",
        "frame_idx": 100,
        "proprio_spec": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
        "action_spec": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
    }
)
```

## Migration Notes

If your data doesn't fit this schema:
1. **First**, try to map it cleanly (most robotics data can)
2. **Second**, use metadata for extra fields
3. **Last resort**, propose a schema extension (requires strong justification)
