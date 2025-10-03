# Table Conventions (Parquet/CSV)

## Overview

Tabular data (observations, actions) is stored in **Parquet** format for efficiency and type safety. CSV is supported as a fallback for human inspection.

## Parquet Benefits

- **Columnar**: Efficient for analytics (read only columns you need)
- **Compressed**: Smaller file sizes than CSV
- **Typed**: Preserves dtypes (float32, int64, etc.)
- **Fast**: Native read/write in pandas, polars, arrow
- **Widely supported**: Works with PyTorch, TensorFlow, DuckDB

## Table Schemas

### observations.parquet

**Purpose**: Store proprioceptive observations and timestamps.

**Schema**:
```
timestamp: float64
step: int64
proprio_0: float32
proprio_1: float32
...
proprio_N: float32
```

**Example**:
```python
import pandas as pd

df = pd.DataFrame({
    "timestamp": [0.0, 0.1, 0.2],
    "step": [0, 1, 2],
    "proprio_0": [0.1, 0.11, 0.12],
    "proprio_1": [-0.2, -0.19, -0.18],
})

df.to_parquet("observations.parquet", index=False, engine="pyarrow")
```

**Alternative with named columns**:
If `metadata['proprio_spec']` is provided:
```python
proprio_spec = ["joint_0", "joint_1", "joint_2"]

df = pd.DataFrame({
    "timestamp": [0.0, 0.1, 0.2],
    "step": [0, 1, 2],
    "joint_0": [0.1, 0.11, 0.12],
    "joint_1": [-0.2, -0.19, -0.18],
    "joint_2": [0.3, 0.31, 0.32],
})
```

---

### actions.parquet

**Purpose**: Store actions taken at each timestep.

**Schema**:
```
timestamp: float64
step: int64
action_0: float32
action_1: float32
...
action_N: float32
```

**Example**:
```python
df = pd.DataFrame({
    "timestamp": [0.0, 0.1, 0.2],
    "step": [0, 1, 2],
    "action_0": [0.01, 0.01, 0.0],
    "action_1": [-0.01, -0.01, 0.0],
    "action_2": [0.0, 0.0, 0.01],
})

df.to_parquet("actions.parquet", index=False, engine="pyarrow")
```

---

## Column Naming

### Reserved Columns

- `timestamp`: Always float64, in seconds
- `step`: Always int64, 0-based index

### Dynamic Columns

- `proprio_N`: Generic indexed columns (N = 0, 1, 2, ...)
- `action_N`: Generic indexed columns
- Custom names: Use `metadata['proprio_spec']` or `metadata['action_spec']`

### Naming Rules

- **Snake_case**: `joint_position`, `gripper_state`
- **No spaces**: Use underscores
- **Descriptive**: `wrist_x` not `w_x`
- **Consistent**: Same names across all episodes

---

## Data Types

| Field | Type | Notes |
|-------|------|-------|
| `timestamp` | `float64` | Seconds, high precision |
| `step` | `int64` | 0-based index |
| `proprio_*` | `float32` | Joint positions, velocities, etc. |
| `action_*` | `float32` | Commanded actions |

**Why float32 for proprio/actions?**
- Sufficient precision for robotics (6-7 significant digits)
- Half the size of float64
- Standard for ML models (PyTorch/TensorFlow default)

---

## Units and Normalization

### Recommended Units

| Field | Unit | Range |
|-------|------|-------|
| Joint positions | radians | [-π, π] or joint limits |
| Joint velocities | rad/s | Depends on robot |
| Cartesian positions | meters | Depends on workspace |
| Gripper state | normalized | [0, 1] (0=open, 1=closed) |
| Actions (delta) | normalized | [-1, 1] per dimension |

### Normalization

**Option 1**: Store raw values, document units in manifest
```json
{
  "proprio_spec": ["joint_0", "joint_1"],
  "proprio_units": ["rad", "rad"],
  "proprio_ranges": [[-3.14, 3.14], [-3.14, 3.14]]
}
```

**Option 2**: Normalize to [-1, 1] or [0, 1], store normalization params
```json
{
  "proprio_spec": ["joint_0_norm", "joint_1_norm"],
  "normalization": {
    "type": "min_max",
    "min": [-3.14, -3.14],
    "max": [3.14, 3.14]
  }
}
```

**Recommendation**: Store raw values + metadata (easier to debug, convert later).

---

## Writing Tables

### Using Pandas

```python
import pandas as pd

# Build DataFrame
data = {
    "timestamp": timestamps,
    "step": steps,
    **{f"proprio_{i}": col for i, col in enumerate(proprio_cols)},
}
df = pd.DataFrame(data)

# Write to parquet
df.to_parquet(
    "observations.parquet",
    index=False,
    engine="pyarrow",
    compression="snappy",  # fast compression
)
```

### Using PyArrow (Faster)

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Build table
schema = pa.schema([
    ("timestamp", pa.float64()),
    ("step", pa.int64()),
    ("proprio_0", pa.float32()),
    ("proprio_1", pa.float32()),
])

table = pa.table({
    "timestamp": pa.array(timestamps, type=pa.float64()),
    "step": pa.array(steps, type=pa.int64()),
    "proprio_0": pa.array(proprio_0, type=pa.float32()),
    "proprio_1": pa.array(proprio_1, type=pa.float32()),
}, schema=schema)

# Write to parquet
pq.write_table(table, "observations.parquet", compression="snappy")
```

---

## Reading Tables

### Using Pandas

```python
import pandas as pd

df = pd.read_parquet("observations.parquet")
print(df.head())
```

### Using PyArrow

```python
import pyarrow.parquet as pq

table = pq.read_table("observations.parquet")
df = table.to_pandas()
```

### Using Polars (Fast)

```python
import polars as pl

df = pl.read_parquet("observations.parquet")
print(df.head())
```

---

## Compression

**Recommended**: `snappy`
- Fast compression/decompression
- Good compression ratio
- Default in many tools

**Alternatives**:
- `gzip`: Better compression, slower
- `zstd`: Modern, balanced
- `uncompressed`: Fastest read/write, largest files

Set compression:
```python
df.to_parquet("observations.parquet", compression="snappy")
```

---

## Buffered Writing

For large episodes, write in chunks:

```python
writer = pq.ParquetWriter("observations.parquet", schema)

for chunk in chunked_samples:
    chunk_table = pa.table(chunk, schema=schema)
    writer.write_table(chunk_table)

writer.close()
```

This avoids loading the entire episode into memory.

---

## CSV Fallback

For debugging or human inspection, export to CSV:

```python
df = pd.read_parquet("observations.parquet")
df.to_csv("observations.csv", index=False)
```

**Warning**: CSV loses type information and is much larger.

---

## Validation

Check table validity:

```python
def validate_table(path):
    df = pd.read_parquet(path)

    # Check required columns
    assert "timestamp" in df.columns
    assert "step" in df.columns

    # Check types
    assert df["timestamp"].dtype == np.float64
    assert df["step"].dtype == np.int64

    # Check monotonicity
    assert df["timestamp"].is_monotonic_increasing
    assert df["step"].equals(pd.Series(range(len(df))))

    # Check no NaNs
    assert not df.isnull().any().any()

    return True
```

---

## Performance Tips

1. **Use PyArrow**: Faster than pandas for large files
2. **Enable compression**: 5-10x size reduction with minimal overhead
3. **Batch writes**: Write in chunks for memory efficiency
4. **Column pruning**: Read only needed columns
   ```python
   df = pd.read_parquet("observations.parquet", columns=["timestamp", "proprio_0"])
   ```
5. **Predicate pushdown**: Filter during read (supported in PyArrow/DuckDB)
   ```python
   table = pq.read_table("observations.parquet", filters=[("step", "<", 100)])
   ```
