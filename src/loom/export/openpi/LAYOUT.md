# OpenPI Directory Layout

## Overview

OpenPI uses a **flat episode structure** where each episode is a self-contained directory. This design supports:
- **Parallel processing**: Process episodes independently
- **Easy deletion**: Remove episodes by deleting directories
- **Simple sharding**: Split dataset across machines by moving directories
- **Atomic writes**: Write to temp directory, rename on success

## Directory Hierarchy

```
{output_path}/
├── manifest.json                 # Dataset metadata
├── statistics.json               # Computed statistics (optional)
├── episodes/
│   ├── episode_000/
│   │   ├── metadata.json         # Episode metadata
│   │   ├── observations.parquet  # Tabular observations
│   │   ├── actions.parquet       # Tabular actions
│   │   └── videos/               # Video directory
│   │       └── camera_0.mp4
│   ├── episode_001/
│   │   ├── metadata.json
│   │   ├── observations.parquet
│   │   ├── actions.parquet
│   │   └── videos/
│   │       ├── camera_0.mp4
│   │       └── camera_1.mp4
│   └── ...
└── .tmp/                         # Temp directory (gitignored)
    └── episode_002_inprogress/
```

## Naming Conventions

### Episode Directories

**Format**: `episode_{index:06d}`

Examples:
- `episode_000000`
- `episode_000001`
- `episode_012345`

**Rules**:
- Zero-padded to 6 digits (supports up to 999,999 episodes)
- Sequential numbering (no gaps)
- Immutable once written (don't renumber)

### Video Files

**Format**: `camera_{name}.mp4` or `{view_name}.mp4`

Examples:
- `camera_0.mp4` (default)
- `camera_front.mp4` (named view)
- `wrist.mp4` (wrist camera)

**Rules**:
- One file per camera/view
- Use consistent naming across episodes
- Document camera names in `manifest.json`

## File Permissions

- **Directories**: `755` (rwxr-xr-x)
- **Data files**: `644` (rw-r--r--)
- **Manifest/metadata**: `644` (rw-r--r--)

## Disk Space Estimates

Typical episode storage (10 Hz, 30 seconds):

| Component | Size | Calculation |
|-----------|------|-------------|
| Video (224x224, H.264) | 1-5 MB | ~0.1-0.5 MB/s |
| Observations (6 dims) | 7 KB | 300 steps × 6 × 4 bytes |
| Actions (7 dims) | 8 KB | 300 steps × 7 × 4 bytes |
| Metadata | 1 KB | JSON overhead |
| **Total per episode** | **~1-5 MB** | |

For 10,000 episodes: **10-50 GB**

## Atomic Write Protocol

To prevent partial writes:

1. **Write to temp**: `.tmp/episode_NNN_inprogress/`
2. **Finalize**: Flush all buffers, close files
3. **Validate**: Check file sizes, parquet headers
4. **Rename**: `mv .tmp/episode_NNN_inprogress episodes/episode_NNN`
5. **Cleanup**: Remove temp directory on error

```python
# Pseudocode
temp_dir = output_path / ".tmp" / f"episode_{idx}_inprogress"
final_dir = output_path / "episodes" / f"episode_{idx:06d}"

try:
    write_episode(temp_dir, samples)
    validate_episode(temp_dir)
    temp_dir.rename(final_dir)
except Exception as e:
    shutil.rmtree(temp_dir)
    raise
```

## Rollover Rules

**When to start a new episode**:
1. Explicit episode boundary in input data (e.g., ROS topic signals "episode_end")
2. Gap in timestamps > `max_gap_seconds` (default: 1.0s)
3. Max episode length reached (default: 1000 steps)
4. Source file changes (e.g., new mp4 file)

**Rollover behavior**:
- Flush current episode writer
- Increment episode counter
- Create new episode directory

## Multi-Camera Layout

For datasets with multiple camera views:

```
episode_000/
├── metadata.json
├── observations.parquet
├── actions.parquet
└── videos/
    ├── camera_front.mp4   # 224x224, 10 fps
    ├── camera_wrist.mp4   # 224x224, 10 fps
    └── camera_side.mp4    # 224x224, 10 fps
```

All videos must have:
- Same number of frames
- Same fps
- Synchronized timestamps (within tolerance)

## Compatibility

OpenPI layout is compatible with:
- **Apache Arrow**: Read parquet files with `pyarrow.parquet.read_table()`
- **Pandas**: `pd.read_parquet("observations.parquet")`
- **PyTorch**: Custom `Dataset` class loads episodes on-demand
- **TensorFlow**: `tf.data.Dataset` with parquet readers
- **DuckDB**: SQL queries over parquet files
