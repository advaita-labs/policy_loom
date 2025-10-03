# MCAP/ROS Bag Ingestion

## Purpose

Read MCAP files (or ROS bags) and yield synchronized `Sample` objects containing images, proprioception, and actions from multiple ROS topics.

## Output Schema

Each `Sample` contains:
- **timestamp**: Message timestamp (from ROS header), in seconds
- **rgb**: Image from configured camera topic, `(H, W, 3)`, dtype `uint8`
- **proprio**: Joint states from configured topic, dtype `float32`
- **action**: Actions from configured topic, dtype `float32`
- **metadata**:
  - `source`: `"mcap"`
  - `episode_id`: Filename stem
  - `frame_idx`: Sequential index after synchronization
  - `topics`: List of topics used for this sample
  - `sync_error_ms`: Max time difference between synchronized messages (milliseconds)

## Topic Mapping

The reader requires a **topic map** configuration:

```yaml
topics:
  image: "/camera/color/image_raw"
  proprio: "/joint_states"
  action: "/action_commands"
```

## Synchronization Policy

**Problem**: ROS topics publish at different rates and may not be perfectly aligned.

**Solution**: Use an **approximate time synchronizer**:
1. Buffer messages from all topics
2. Find the closest messages across topics within a **time tolerance** (default: 50ms)
3. Yield a `Sample` when all required topics have matching messages
4. Drop unmatched messages

**Configuration**:
- `time_tolerance_ms`: Max time difference for messages to be considered synchronized (default: 50)
- `required_topics`: Topics that must be present (default: all)
- `queue_size`: Buffer size for each topic (default: 10)

## Dependencies

**Required**:
- `mcap` for reading MCAP files
- `mcap-ros2-support` for ROS 2 message deserialization
- `numpy`

Install with:
```bash
uv pip install mcap mcap-ros2-support
```

## Usage Example

```python
from loom.io.mcap import MCAPReader

config = {
    "topics": {
        "image": "/camera/image_raw",
        "proprio": "/joint_states",
        "action": "/cmd_vel"
    },
    "time_tolerance_ms": 50,
}

with MCAPReader("recording.mcap", config=config) as reader:
    for sample in reader.read():
        print(f"Timestamp: {sample.timestamp:.3f}s")
        print(f"  RGB: {sample.rgb.shape}")
        print(f"  Proprio: {sample.proprio.shape}")
        print(f"  Action: {sample.action.shape}")
        print(f"  Sync error: {sample.metadata['sync_error_ms']:.2f}ms")
```

## Limitations

- **Synchronization drops data**: Messages that can't be matched within tolerance are discarded
- **ROS 2 only**: Currently no ROS 1 support (use `rosbag2mcap` converter)
- **Fixed schema**: Topic mapping must be provided upfront
- **No compression**: MCAP compression is handled by the library, not by us

## Future Extensions

- ROS 1 bag support
- Multi-camera synchronization
- Depth image support
- IMU data ingestion
- Configurable sync policies (exact time, nearest, interpolation)
