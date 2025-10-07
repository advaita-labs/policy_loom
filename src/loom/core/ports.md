# Pipeline Component Contracts

## Overview

`policy_loom` uses three core abstractions (ports) to keep components decoupled and composable:

1. **Reader** - Ingest raw data into `Sample` streams
2. **Transform** - Stateless modifications to `Sample` objects
3. **Writer** - Output `Sample` streams to disk

An optional fourth port, **Exporter**, groups format-specific logic.

---

## Reader

**Purpose**: Turn a data source (mp4, mcap, image directory) into a stream of `Sample` objects.

**Contract**:
```python
class Reader(ABC):
    def read(self) -> Iterator[Sample]:
        """Yield samples in temporal order."""
        ...

    def close(self) -> None:
        """Release resources (file handles, etc.)."""
        pass
```

**Responsibilities**:
- Open and parse the data source
- Yield `Sample` objects in **temporal order**
- Handle missing fields gracefully (e.g., video-only → `proprio=None`)
- Populate `metadata` with provenance info (`source`, `episode_id`, etc.)
- Manage resources (use context manager pattern)

**Error Handling**:
- Raise `IOError` for file access issues
- Raise `ValueError` for malformed data
- Log warnings for recoverable issues (missing frames, etc.)

**Example Implementations**:
- `loom.io.mp4.MP4Reader` - reads video files
- `loom.io.mcap.MCAPReader` - reads ROS bag data

---

## Transform

**Purpose**: Apply a stateless transformation to a `Sample`.

**Contract**:
```python
class Transform(ABC):
    def __call__(self, sample: Sample) -> Sample:
        """Return transformed sample."""
        ...
```

**Responsibilities**:
- Modify one or more fields of a `Sample`
- Remain **stateless** (no memory between calls)
- Preserve temporal order (unless explicitly documented as reordering)
- Validate inputs and raise `ValueError` for invalid samples
- Document side effects (e.g., "drops frames", "changes resolution")

**Key Properties**:
- **Composable**: `transform_b(transform_a(sample))` should work
- **Deterministic**: Same input → same output (no randomness unless seeded)
- **Fast**: Avoid I/O, heavy computation (for per-sample transforms)

**Example**: Implement custom transforms by subclassing Transform
```python
from loom.core import Transform, Sample
import cv2

class ResizeTransform(Transform):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, sample: Sample) -> Sample:
        for camera in sample.cameras:
            camera.image = cv2.resize(camera.image, (self.width, self.height))
        return sample
```

---

## Design Principles

1. **Single Responsibility**: Each port does one thing well
2. **Fail Fast**: Validate at construction, not at runtime
3. **Context Managers**: Use `with` statements for resource safety
4. **Typed Interfaces**: Return/accept concrete types (`Sample`, not `dict`)
5. **Documented Contracts**: ABCs encode the "what", docs encode the "why"

---

## Composition Example

```python
from loom.io.mp4 import MP4Reader
from loom.core import Transform, Sample

# Implement custom transforms
class ResizeTransform(Transform):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, sample: Sample) -> Sample:
        # Your resize logic here
        return sample

# Build a pipeline
with MP4Reader("input.mp4") as reader:
    # Compose transforms
    pipeline = [
        ResizeTransform(height=224, width=224),
        # Add more custom transforms here
    ]

    for sample in reader.read():
        for transform in pipeline:
            sample = transform(sample)
        # Process the transformed sample
        print(f"Processed frame at {sample.timestamp}s")
```

Clean, composable, testable. That's the goal.
