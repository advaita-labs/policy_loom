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

**Example Implementations**:
- `loom.transforms.vision.Resize` - resize images
- `loom.transforms.time.ResampleFPS` - temporal resampling
- `loom.transforms.vision.Normalize` - normalize pixel values

---

## Writer

**Purpose**: Write a stream of `Sample` objects to disk in a specific format.

**Contract**:
```python
class Writer(ABC):
    def write(self, sample: Sample) -> None:
        """Write a single sample."""
        ...

    def close(self) -> None:
        """Finalize writes, flush buffers, write manifest."""
        ...
```

**Responsibilities**:
- Create output directory structure
- Write samples to appropriate files (parquet, mp4, csv, etc.)
- Buffer writes for efficiency (flush in `close()`)
- Write a **manifest** with metadata (versions, config, stats)
- Use **atomic writes** (write to temp, rename on success)
- Handle write failures gracefully (log, skip, or abort)

**Lifecycle**:
1. `__init__` - set up paths, create temp directories
2. `write(sample)` - called once per sample (may buffer)
3. `close()` - flush buffers, write manifest, rename temp → final

**Example Implementations**:
- `loom.export.openpi.OpenPIWriter` - writes OpenPI format
- Generic writers in `loom.write.*`

---

## Exporter (Optional)

**Purpose**: Encapsulate format-specific logic (schema, layout, validation).

**Contract**:
```python
class Exporter(ABC):
    def create_writer(self, output_path: Path, **kwargs) -> Writer:
        """Factory for creating a Writer."""
        ...

    def validate_output(self, output_path: Path) -> bool:
        """Validate output conforms to schema."""
        ...
```

**When to Use**:
- For well-defined export formats (OpenPI, RLDS, etc.)
- To group related schema/layout/validation logic
- To provide a stable public API for a format

**Example**:
```python
exporter = OpenPIExporter()
writer = exporter.create_writer(Path("./out"), fps=10)
for sample in reader.read():
    writer.write(sample)
writer.close()
exporter.validate_output(Path("./out"))
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
from loom.transforms.vision import Resize, Normalize
from loom.transforms.time import ResampleFPS
from loom.export.openpi import OpenPIWriter

# Build a pipeline
with MP4Reader("input.mp4") as reader, \
     OpenPIWriter("output/") as writer:

    # Compose transforms
    pipeline = [
        ResampleFPS(target_fps=10),
        Resize(height=224, width=224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    for sample in reader.read():
        for transform in pipeline:
            sample = transform(sample)
        writer.write(sample)
```

Clean, composable, testable. That's the goal.
