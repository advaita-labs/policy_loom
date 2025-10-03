# MP4 Video Ingestion

## Purpose

Read video files (mp4, avi, mov) and yield a stream of `Sample` objects containing only RGB frames and timestamps.

## Output Schema

Each `Sample` contains:
- **timestamp**: Frame timestamp in seconds (computed from frame index and fps)
- **rgb**: Frame as numpy array `(H, W, 3)`, dtype `uint8`
- **proprio**: `None`
- **action**: `None`
- **metadata**:
  - `source`: `"mp4"`
  - `episode_id`: Filename stem (e.g., `"demo_001"` from `demo_001.mp4`)
  - `frame_idx`: 0-based frame index
  - `fps`: Frames per second from video metadata
  - `width`: Original video width
  - `height`: Original video height
  - `codec`: Video codec (e.g., `"h264"`)

## Timing Rules

- **Frame timestamps** are computed as `frame_idx / fps`
- Assumes **constant frame rate** (CFR)
- For variable frame rate (VFR) videos, use frame presentation timestamps if available
- Dropped frames in the source video are **not** detected or filled

## Dependencies

**Required**:
- `av` (PyAV) for video decoding
- `numpy`

**Optional**:
- `ffmpeg` system library (usually installed with PyAV)

Install with:
```bash
uv pip install av
```

## Usage Example

```python
from loom.io.mp4 import MP4Reader

with MP4Reader("demo.mp4") as reader:
    for sample in reader.read():
        print(f"Frame {sample.metadata['frame_idx']}: {sample.rgb.shape}")
```

## Limitations

- **No audio**: Audio tracks are ignored
- **No actions/proprio**: This is video-only ingestion
- **RGB only**: Grayscale videos are converted to RGB
- **No multi-view**: One video = one camera stream (use multiple readers for multi-camera)

## Future Extensions

- Multi-view support (read multiple videos with synchronized timestamps)
- Depth map ingestion (from RGBD videos)
- Audio feature extraction
