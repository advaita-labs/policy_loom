# Episode Layout and Atomicity

## Overview

Writers in `policy_loom` follow strict rules for episode creation to ensure data integrity and prevent corruption.

## Episode Creation Lifecycle

```
1. Initialize temp directory
   └── .tmp/episode_NNN_inprogress/

2. Write samples incrementally
   ├── Buffer samples in memory
   └── Flush to parquet/mp4 periodically

3. Finalize episode
   ├── Flush remaining buffers
   ├── Write metadata.json
   └── Close all file handles

4. Validate episode
   ├── Check file sizes > 0
   ├── Verify parquet headers
   └── Count video frames

5. Atomic rename
   └── mv .tmp/episode_NNN_inprogress/ episodes/episode_NNN/

6. Cleanup on error
   └── rm -rf .tmp/episode_NNN_inprogress/
```

## Directory Naming

### Episode Numbering

**Format**: `episode_{index:06d}`

Examples:
- `episode_000000`
- `episode_000123`
- `episode_099999`

**Rules**:
- Zero-padded to 6 digits (supports up to 999,999 episodes)
- Sequential numbering (no gaps)
- Start from 0 for each dataset
- Immutable once written

### Temp Directory

**Format**: `.tmp/{name}_inprogress/`

Examples:
- `.tmp/episode_000000_inprogress/`
- `.tmp/episode_000001_inprogress/`

**Location**: Same parent directory as `episodes/`

**Cleanup**: Auto-deleted on error or completion

## Atomic Writes

### Why Atomic?

**Problem**: If writing fails mid-episode, partial data corrupts the dataset.

**Solution**: Write to temp, validate, then atomically rename.

```python
# Pseudocode
def write_episode(samples, output_dir, episode_idx):
    temp_dir = output_dir / ".tmp" / f"episode_{episode_idx:06d}_inprogress"
    final_dir = output_dir / "episodes" / f"episode_{episode_idx:06d}"

    try:
        # Write to temp
        temp_dir.mkdir(parents=True)
        write_observations(temp_dir / "observations.parquet", samples)
        write_actions(temp_dir / "actions.parquet", samples)
        write_videos(temp_dir / "videos", samples)
        write_metadata(temp_dir / "metadata.json", samples)

        # Validate
        validate_episode(temp_dir)

        # Atomic rename
        temp_dir.rename(final_dir)

    except Exception as e:
        # Cleanup on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"Failed to write episode {episode_idx}") from e
```

### Atomicity Guarantees

On Unix systems (Linux, macOS), `rename()` is atomic if:
- Source and destination are on the **same filesystem**
- Destination does not exist (or is atomically replaced)

**Best practice**: Ensure `.tmp/` and `episodes/` are on the same mount point.

## Roll-Over Rules

### When to Start a New Episode

1. **Explicit boundary**: `metadata['episode_end'] == True`
2. **Time gap**: `timestamp_delta > max_gap_seconds` (default: 1.0s)
3. **Max length**: `episode_length > max_episode_length` (default: 10,000 steps)
4. **Source change**: New input file (e.g., `demo_001.mp4` → `demo_002.mp4`)

### Roll-Over Behavior

```python
if should_rollover(current_sample, previous_sample):
    # Finalize current episode
    current_writer.close()

    # Start new episode
    episode_idx += 1
    current_writer = create_writer(episode_idx)
```

**Important**: Always finalize (close) the previous episode before starting a new one.

## File Naming Within Episodes

### Required Files

- `metadata.json` - Episode metadata
- `observations.parquet` - Observations table
- `actions.parquet` - Actions table (optional, if actions present)
- `videos/` - Directory for video files

### Optional Files

- `depth/` - Depth images (future)
- `audio/` - Audio files (future)
- `events.json` - Event annotations (future)

### Video Files

**Naming**: `{camera_name}.mp4` or `camera_{idx}.mp4`

Examples:
- `camera_0.mp4` (default)
- `camera_front.mp4` (named view)
- `camera_wrist.mp4` (named view)

**Rule**: Use consistent naming across all episodes in a dataset.

## Temp Directory Cleanup

### On Success

- Temp directory is renamed to final location
- No cleanup needed

### On Error

- Temp directory is deleted
- Logs should indicate what went wrong

### On Interruption (Ctrl+C, kill)

- Temp directories may remain
- Runner should clean up stale temps on next run

```python
def cleanup_stale_temps(output_dir):
    temp_dir = output_dir / ".tmp"
    for stale in temp_dir.glob("*_inprogress"):
        if stale.stat().st_mtime < time.time() - 3600:  # 1 hour old
            logger.warning(f"Removing stale temp: {stale}")
            shutil.rmtree(stale)
```

## Permissions and Ownership

**Directories**: `755` (rwxr-xr-x)
- Owner: read, write, execute
- Group/others: read, execute

**Files**: `644` (rw-r--r--)
- Owner: read, write
- Group/others: read

**Set permissions**:
```python
episode_dir.mkdir(mode=0o755, parents=True)
metadata_file.chmod(0o644)
```

## Concurrent Writes

**Question**: Can multiple writers write to the same dataset?

**Answer**: Yes, if episode indices don't collide.

**Strategy**: Use file locking or allocate episode ranges per worker.

```python
# Example: Worker-based allocation
worker_id = 0
num_workers = 4
episode_offset = worker_id * 100000  # Worker 0: 0-99999, Worker 1: 100000-199999

episode_idx = episode_offset + local_idx
```

## Validation Checklist

Before finalizing an episode, check:
- ✅ `metadata.json` exists and is valid JSON
- ✅ `observations.parquet` exists and is readable
- ✅ `actions.parquet` exists (if actions present)
- ✅ Video files exist and have >0 frames
- ✅ File sizes > 0 bytes
- ✅ Timestamps are monotonic
- ✅ Step indices are `range(len(observations))`

If any check fails, **do not rename** the temp directory.

## Error Recovery

**Scenario**: Writer crashes mid-episode.

**Recovery**:
1. On restart, detect incomplete episodes in `.tmp/`
2. Delete incomplete episodes (can't resume)
3. Start from last successfully written episode
4. Log the gap for user awareness

**Future**: Implement episode checkpointing for resumable writes.
