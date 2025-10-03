# Time Transforms

## Purpose

Temporal transformations that modify the **sequence** of samples, not individual samples. Unlike vision transforms, time transforms:
- May drop or duplicate samples
- May reorder samples (if explicitly documented)
- Operate on streams, not single samples
- May be stateful (buffer recent samples)

## Available Transforms

### `ResampleFPS`

**Purpose**: Change the frame rate of a sample stream.

**Config**:
```yaml
- type: resample_fps
  target_fps: 10
  method: nearest  # nearest, interpolate, drop, duplicate
```

**Methods**:
- `nearest`: Keep samples closest to target timestamps
- `interpolate`: Interpolate between samples (for proprio/action)
- `drop`: Drop excess samples
- `duplicate`: Repeat samples to increase fps

**Behavior**:
- Updates `metadata['fps']`
- Adds `metadata['resampled']: true`
- Preserves temporal order

---

### `Window`

**Purpose**: Extract a sliding window of samples.

**Config**:
```yaml
- type: window
  size: 16  # number of frames per window
  stride: 1  # shift by N frames
  mode: stack  # stack, list
```

**Output**:
- `stack`: Sample with `rgb` shape `(T, H, W, C)`
- `list`: List of `T` samples (for non-uniform processing)

**Use case**: Create temporal windows for video models (3D CNNs, transformers).

---

### `Align`

**Purpose**: Align samples from multiple sources to a common timeline.

**Config**:
```yaml
- type: align
  reference: camera_0
  tolerance_ms: 50
```

**Behavior**:
- Synchronizes samples from different streams (multi-camera, multi-sensor)
- Drops samples that can't be aligned within tolerance
- Adds `metadata['sync_error_ms']`

---

### `Subsample`

**Purpose**: Keep every Nth sample.

**Config**:
```yaml
- type: subsample
  factor: 2  # keep every 2nd sample
  offset: 0  # start from this index
```

**Use case**: Reduce dataset size, test on sparse data.

---

### `TemporalShift`

**Purpose**: Shift timestamps by a fixed offset.

**Config**:
```yaml
- type: temporal_shift
  offset_seconds: -0.1  # shift backwards by 100ms
```

**Use case**: Correct timing errors, align multi-sensor data.

---

### `Deduplicate`

**Purpose**: Remove duplicate frames (same timestamp or same pixels).

**Config**:
```yaml
- type: deduplicate
  mode: timestamp  # timestamp, pixel, both
  tolerance_ms: 1  # for timestamp deduplication
```

---

## Dependencies

- `numpy` (required)
- `scipy` (optional, for interpolation)

Install with:
```bash
uv pip install scipy
```

## Usage Example

```python
from loom.transforms.time import ResampleFPS
from loom.io.mp4 import MP4Reader

resample = ResampleFPS(target_fps=10)

with MP4Reader("video.mp4") as reader:
    for sample in resample(reader.read()):
        print(f"Resampled timestamp: {sample.timestamp}")
```

## Design Notes

- **Stream-aware**: Operate on `Iterator[Sample]`, not single `Sample`
- **Stateful when needed**: May buffer samples for interpolation/alignment
- **Order-preserving**: Unless explicitly documented (e.g., shuffle)
- **Document drops**: Log or track dropped frames in metadata
- **Configurable**: Expose all parameters (no magic defaults)
