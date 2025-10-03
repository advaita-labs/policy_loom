# Design: Ingest and Preprocessing

## Motivation

Training Vision-Language-Action (VLA) models requires:
1. **Diverse data sources**: Videos, ROS bags, image sequences
2. **Consistent preprocessing**: Resize, normalize, temporal alignment
3. **Efficient storage**: For large datasets (100k+ episodes)
4. **Reproducibility**: Track configs, versions, provenance

Existing solutions are fragmented:
- Video tools don't handle robotics data
- ROS tools don't output ML-friendly formats
- Each research group builds custom pipelines

**Goal**: A single, modular toolkit for VLA data preprocessing.

---

## Core Principles

### 1. Separation of Concerns

Three distinct stages:
- **Ingest**: Raw data → canonical `Sample`
- **Transform**: `Sample` → modified `Sample`
- **Export**: `Sample` stream → target format

Each stage is independent and testable.

### 2. Documentation-First

Code is ephemeral, contracts are eternal. We document:
- What each component does (responsibilities)
- What it expects (inputs, preconditions)
- What it produces (outputs, guarantees)
- What can go wrong (failure modes)

Markdown docs live alongside code, not in a separate wiki.

### 3. Composability

Components are **Lego blocks**:
- Readers are interchangeable (mp4, mcap, images)
- Transforms compose: `transform_c(transform_b(transform_a(sample)))`
- Writers are pluggable (OpenPI, RLDS, custom)

No monolithic pipelines, no tightly-coupled code.

### 4. Late Batching

Preprocessing operates on **single samples**, not batches:
- Simpler code (no padding, no shape mismatches)
- Lower memory (stream processing)
- More flexible (variable-length episodes)

Batching happens at training time (PyTorch `DataLoader`).

### 5. Fail Fast, Log Everything

- Validate inputs at construction time
- Raise exceptions on errors (don't silently skip)
- Log warnings for recoverable issues
- Write manifests for every run (reproducibility)

---

## Architecture

### Data Flow

```
┌──────────────┐
│ Raw Data     │  (mp4, mcap, images, ...)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Reader       │  Yields Sample stream
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Transforms   │  Stateless modifications
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Writer       │  Writes to target format
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Output       │  (OpenPI, RLDS, ...)
└──────────────┘
```

### Components

#### Sample (Canonical Type)

```python
@dataclass
class Sample:
    timestamp: float | int
    rgb: NDArray | None
    proprio: NDArray | None
    action: NDArray | None
    metadata: dict[str, Any]
```

**Why this schema?**
- Covers 95% of VLA use cases
- Typed (numpy arrays, not lists)
- Validated (shape checks in `__post_init__`)
- Extensible (metadata for uncommon fields)

#### Reader (Ingest)

```python
class Reader(ABC):
    @abstractmethod
    def read(self) -> Iterator[Sample]:
        ...
```

**Implementations**:
- `MP4Reader`: Video files
- `MCAPReader`: ROS bags
- `ImageDirReader`: Image sequences (future)

**Responsibilities**:
- Open data source
- Parse into `Sample` objects
- Handle missing fields (e.g., video-only → `proprio=None`)
- Close resources (context manager)

#### Transform (Processing)

```python
class Transform(ABC):
    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        ...
```

**Examples**:
- `Resize(height=224, width=224)`: Resize images
- `ResampleFPS(target_fps=10)`: Change frame rate
- `Normalize(preset="imagenet")`: Normalize pixels

**Key property**: Stateless (no memory between calls).

#### Writer (Export)

```python
class Writer(ABC):
    @abstractmethod
    def write(self, sample: Sample) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
```

**Implementations**:
- `OpenPIWriter`: OpenPI format
- `RLDSWriter`: TensorFlow RLDS (future)

**Responsibilities**:
- Create output directory structure
- Buffer samples for efficient I/O
- Write manifest with metadata
- Ensure atomic writes (temp → rename)

---

## Design Decisions

### Why Parquet, Not HDF5?

| Feature | Parquet | HDF5 |
|---------|---------|------|
| Columnar | ✅ Yes | ❌ No |
| Compression | ✅ Built-in | ⚠️ Manual |
| Ecosystem | ✅ Pandas, Arrow, DuckDB | ⚠️ Fewer tools |
| Typed | ✅ Yes | ⚠️ Weakly typed |
| Cloud-native | ✅ Yes (S3, GCS) | ❌ Not designed for cloud |

Parquet is the modern standard for tabular data.

### Why MP4, Not Raw Frames?

| Format | Storage | Speed | Compatibility |
|--------|---------|-------|---------------|
| MP4 (H.264) | 10-20x smaller | Fast decode | ✅ Universal |
| Raw frames | Large | Fastest | ⚠️ Custom loader |
| PNG | 2-5x smaller | Slower | ⚠️ No temporal compression |

MP4 balances size and speed. Use H.264 (not H.265) for compatibility.

### Why Episode Directories, Not Monolithic Files?

**Benefits**:
- Easy to delete/move/shard episodes
- Parallel processing (process episodes independently)
- Atomic writes (write to temp, rename on success)
- Simple validation (check files in directory)

**Trade-off**: More files (but modern filesystems handle this well).

### Why YAML Configs, Not Python?

**Benefits**:
- Language-agnostic (anyone can read/write)
- Version-controllable (git-friendly)
- No code execution (safer)
- Easy to generate/template

**Trade-off**: Less expressive than Python (but configs should be simple).

---

## Alternatives Considered

### Monolithic Preprocessing Script

**Rejected**: Doesn't scale to new data sources or formats. Every new dataset requires rewriting the script.

**Our approach**: Modular components that compose.

### DataFrame-Based Processing

Use pandas DataFrames instead of `Sample` objects.

**Rejected**:
- Hard to represent images (DataFrames are for tabular data)
- Awkward for nested data (videos, multi-camera)
- Type checking is weaker

**Our approach**: Typed dataclasses with numpy arrays.

### Streaming Transforms (e.g., `map`, `filter`)

Use functional-style transforms like `samples.map(resize).filter(predicate)`.

**Rejected**: Overengineering for our use case. Explicit loops are clearer.

**Our approach**: Explicit `for sample in reader.read()` loops.

---

## Extension Points

### Adding a New Data Format

1. Subclass `Reader`
2. Implement `read() -> Iterator[Sample]`
3. Add README documenting the format
4. Add tests with golden outputs

### Adding a New Transform

1. Subclass `Transform`
2. Implement `__call__(sample: Sample) -> Sample`
3. Add to `transforms/CATALOG.md`
4. Add tests

### Adding a New Export Format

1. Subclass `Writer` (or `Exporter`)
2. Implement `write(sample)` and `close()`
3. Document schema and layout
4. Add validation

---

## Future Work

### Parallelization

Current: Sequential processing (one sample at a time).

Future: Parallel processing at the **file level** (process multiple videos in parallel).

**Why not sample-level parallelism?**
- Overhead too high for small samples
- Transforms are fast (resize, normalize = milliseconds)
- I/O is the bottleneck

### Cloud Storage

Support reading from/writing to S3, GCS:
- Use `fsspec` for filesystem abstraction
- Stream data (don't download entire files)

### Multi-Modal Data

- Depth images
- Audio
- IMU data
- Force-torque sensors

Extend `Sample` with optional fields as needed.

### Training Integration

- PyTorch `Dataset` for OpenPI
- TensorFlow `tf.data` pipeline
- Data augmentation at training time

---

## References

- OpenVLA: https://openvla.github.io
- OpenPI (inspired by this toolkit)
- RLDS: https://github.com/google-research/rlds
- LeRobot: https://github.com/huggingface/lerobot
