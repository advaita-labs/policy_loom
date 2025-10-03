# Preprocessing Pipeline Flow

## Overview

The preprocessing pipeline transforms raw data (mp4, mcap, etc.) into a training-ready format (OpenPI, RLDS, etc.). It follows a strict order to ensure reproducibility.

## Pipeline Stages

```
┌──────────┐
│ 1. READ  │  Open data source, yield Sample stream
└────┬─────┘
     │
     ▼
┌──────────────┐
│ 2. TRANSFORM │  Apply transforms (resize, resample, normalize, etc.)
└────┬─────────┘
     │
     ▼
┌──────────┐
│ 3. WRITE │  Write to output format (OpenPI)
└──────────┘
```

## Stage Details

### 1. Read (Ingest)

**Purpose**: Convert raw data source into canonical `Sample` stream.

**Components**:
- `Reader` instance (e.g., `MP4Reader`, `MCAPReader`)
- Context manager for resource safety

**Flow**:
```python
with Reader(input_path) as reader:
    for sample in reader.read():
        yield sample
```

**Failure Policy**:
- **IOError**: Log error, skip file, continue with next
- **ValueError**: Log warning, skip malformed samples
- **Resource leak**: Context manager ensures cleanup

---

### 2. Transform (Processing)

**Purpose**: Apply stateless transformations to each sample.

**Components**:
- List of `Transform` instances
- Applied sequentially (order matters!)

**Flow**:
```python
transforms = [
    ResampleFPS(target_fps=10),
    Resize(height=224, width=224),
    Normalize(preset="imagenet"),
]

for sample in reader.read():
    for transform in transforms:
        sample = transform(sample)
    yield sample
```

**Failure Policy**:
- **ValueError**: Log error, skip sample, continue
- **Unexpected exception**: Abort pipeline (fail fast)

**Optimization**:
- Transforms are **stateless** (no caching needed)
- Can be parallelized with multiprocessing (future)
- Profile to find bottlenecks

---

### 3. Write (Export)

**Purpose**: Write processed samples to disk in target format.

**Components**:
- `Writer` instance (e.g., `OpenPIWriter`)
- Buffering for efficiency
- Atomic writes for safety

**Flow**:
```python
with Writer(output_path) as writer:
    for sample in transformed_samples:
        writer.write(sample)
    # writer.close() called automatically
```

**Failure Policy**:
- **IOError**: Abort pipeline, clean up temp files
- **DiskFullError**: Abort, log error, alert user
- **Partial write**: Use atomic rename to prevent corruption

---

## End-to-End Example

```python
from loom.io.mp4 import MP4Reader
from loom.transforms.vision import Resize, Normalize
from loom.transforms.time import ResampleFPS
from loom.export.openpi import OpenPIWriter

def preprocess_pipeline(input_path: str, output_path: str) -> None:
    """Preprocess mp4 to OpenPI format."""

    # Stage 1: Read
    reader = MP4Reader(input_path)

    # Stage 2: Transform
    transforms = [
        ResampleFPS(target_fps=10),
        Resize(height=224, width=224),
        Normalize(preset="imagenet"),
    ]

    # Stage 3: Write
    with OpenPIWriter(output_path) as writer:
        for sample in reader.read():
            # Apply transforms
            for transform in transforms:
                sample = transform(sample)

            # Write to output
            writer.write(sample)

    print(f"Preprocessing complete: {output_path}")
```

## Config-Driven Pipelines

For production, use YAML configs:

```yaml
# preprocess_mp4_to_openpi.yaml
input:
  type: mp4
  path: "./input/demo.mp4"

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
```

Execute with:
```bash
uv run python -m loom.runners.preprocess --config preprocess_mp4_to_openpi.yaml
```

## Parallelization (Future)

For large datasets, parallelize at the **file level**:

```python
# Pseudocode
files = glob("input/*.mp4")
with Pool(num_workers) as pool:
    pool.map(preprocess_pipeline, files)
```

**Not** at the sample level (overhead too high).

## Observability

Every pipeline run produces:
- **Manifest**: Input/output paths, config hash, versions
- **Logs**: Warnings, errors, dropped samples
- **Statistics**: Total samples, processing time, throughput

See `observability/MANIFEST.md` for details.

## Reproducibility Checklist

- ✅ Deterministic transforms (seeded randomness)
- ✅ Version-pinned dependencies
- ✅ Config hashed and stored
- ✅ Input data checksums
- ✅ Output validation

## Common Issues

**Issue**: Out of memory
**Solution**: Process in smaller batches, reduce buffer sizes

**Issue**: Slow preprocessing
**Solution**: Profile transforms, use C++ implementations (OpenCV), parallelize

**Issue**: Dropped samples
**Solution**: Check logs, adjust time tolerance, validate input data

**Issue**: Inconsistent output
**Solution**: Pin random seeds, check for stateful transforms
