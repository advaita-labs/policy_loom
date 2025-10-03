# Logging Conventions

## Overview

`policy_loom` uses Python's standard `logging` module with structured logging practices for debugging and monitoring.

## Log Levels

| Level | When to Use | Examples |
|-------|-------------|----------|
| `DEBUG` | Detailed diagnostic info | "Read frame 42 from video", "Transform took 0.1ms" |
| `INFO` | Confirmation that things are working | "Started processing demo.mp4", "Wrote episode_000" |
| `WARNING` | Something unexpected but recoverable | "Dropped 3 frames due to sync error", "Missing proprio data" |
| `ERROR` | Serious problem, operation failed | "Failed to write episode", "Invalid parquet file" |
| `CRITICAL` | System-level failure | "Disk full, aborting", "Out of memory" |

## Logger Hierarchy

```
loom                              # Root logger
├── loom.io                       # All IO operations
│   ├── loom.io.mp4               # MP4 reader
│   └── loom.io.mcap              # MCAP reader
├── loom.transforms               # All transforms
│   ├── loom.transforms.vision    # Vision transforms
│   └── loom.transforms.time      # Time transforms
├── loom.export                   # Export operations
│   └── loom.export.openpi        # OpenPI writer
├── loom.runners                  # Pipeline runners
└── loom.observability            # Logging itself
```

**Benefit**: Control verbosity per module (e.g., debug transforms but not IO).

## Configuration

### Basic Setup

```python
import logging

# Root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("preprocessing.log"),  # File
    ],
)

logger = logging.getLogger("loom")
```

### Module-Specific Loggers

```python
# In loom/io/mp4/reader.py
logger = logging.getLogger(__name__)  # "loom.io.mp4.reader"

logger.info("Opening video file: %s", video_path)
logger.debug("Video resolution: %dx%d", width, height)
```

### Per-Run Log Files

```python
from datetime import datetime
from pathlib import Path

log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"preprocess_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ],
)
```

## Structured Logging

### Component Tags

Prefix log messages with component tags for easy filtering:

```python
logger.info("[MP4Reader] Opened video: %s", video_path)
logger.info("[ResizeTransform] Resized frame %d to %dx%d", frame_idx, height, width)
logger.info("[OpenPIWriter] Wrote episode %d", episode_idx)
```

**Benefit**: `grep '\[MP4Reader\]' preprocess.log`

### Contextual Information

Include context in log messages:

```python
logger.info(
    "Processed episode %d: %d timesteps, %.2f seconds",
    episode_idx,
    num_timesteps,
    duration,
)
```

### Exception Logging

Always log exceptions with traceback:

```python
try:
    process_episode(samples)
except Exception as e:
    logger.error("Failed to process episode %d", episode_idx, exc_info=True)
    raise
```

## Log Message Guidelines

### Good Messages

✅ **Specific**: "Failed to read frame 42 from demo.mp4"
✅ **Actionable**: "Disk full (95% used), cleaning up temp files"
✅ **Contextual**: "Dropped 3 frames (timestamps: 1.2s, 1.3s, 1.4s) due to sync error"

### Bad Messages

❌ **Vague**: "Error occurred"
❌ **Too verbose**: "Processing frame 1", "Processing frame 2", ... (use DEBUG)
❌ **No context**: "Invalid value" (what value? where?)

## Example Logs

### Successful Run

```
2025-01-15 10:30:00 [INFO] loom.runners: Starting preprocessing: demo.mp4 -> ./output/demo_processed
2025-01-15 10:30:01 [INFO] loom.io.mp4: [MP4Reader] Opened video: demo.mp4 (450 frames, 30 fps)
2025-01-15 10:30:01 [INFO] loom.transforms.time: [ResampleFPS] Target FPS: 10 (3x downsampling)
2025-01-15 10:30:02 [INFO] loom.transforms.vision: [Resize] Resizing frames to 224x224
2025-01-15 10:30:05 [INFO] loom.export.openpi: [OpenPIWriter] Writing episode_000000
2025-01-15 10:30:13 [INFO] loom.export.openpi: [OpenPIWriter] Episode complete: 150 timesteps, 15.0s
2025-01-15 10:30:13 [INFO] loom.runners: Preprocessing complete: 1 episode, 150 timesteps, 13.2s
```

### Run with Warnings

```
2025-01-15 10:30:00 [INFO] loom.runners: Starting preprocessing: recording.mcap -> ./output/recording_processed
2025-01-15 10:30:01 [INFO] loom.io.mcap: [MCAPReader] Opened MCAP: recording.mcap (3 topics)
2025-01-15 10:30:02 [WARNING] loom.io.mcap: [MCAPReader] Dropped 5 messages (sync error > 50ms)
2025-01-15 10:30:05 [WARNING] loom.transforms.time: [ResampleFPS] Interpolated 2 missing frames
2025-01-15 10:30:10 [INFO] loom.export.openpi: [OpenPIWriter] Writing episode_000000
2025-01-15 10:30:18 [INFO] loom.export.openpi: [OpenPIWriter] Episode complete: 445 timesteps, 44.5s
2025-01-15 10:30:18 [INFO] loom.runners: Preprocessing complete: 1 episode, 445 timesteps, 18.1s
2025-01-15 10:30:18 [WARNING] loom.runners: 7 warnings occurred, check logs for details
```

### Run with Errors

```
2025-01-15 10:30:00 [INFO] loom.runners: Starting preprocessing: demo.mp4 -> ./output/demo_processed
2025-01-15 10:30:01 [INFO] loom.io.mp4: [MP4Reader] Opened video: demo.mp4 (450 frames, 30 fps)
2025-01-15 10:30:05 [ERROR] loom.export.openpi: [OpenPIWriter] Failed to write observations.parquet
Traceback (most recent call last):
  File "loom/export/openpi/writer.py", line 123, in write_table
    df.to_parquet(path)
  File "pandas/core/frame.py", line 2345, in to_parquet
    raise IOError("Disk full")
IOError: Disk full
2025-01-15 10:30:05 [ERROR] loom.runners: Preprocessing failed, cleaning up temp files
2025-01-15 10:30:06 [INFO] loom.runners: Removed temp directory: .tmp/episode_000000_inprogress
```

## Progress Logging

For long-running operations, log progress periodically:

```python
total = len(samples)
for i, sample in enumerate(samples):
    process(sample)

    if (i + 1) % 100 == 0:  # Log every 100 samples
        logger.info("Processed %d / %d samples (%.1f%%)", i + 1, total, 100 * (i + 1) / total)
```

**Alternative**: Use `tqdm` for progress bars (logged to stderr).

## Log Filtering

### By Level

```bash
# Show only warnings and errors
grep -E '\[WARNING\]|\[ERROR\]|\[CRITICAL\]' preprocess.log
```

### By Component

```bash
# Show only MP4 reader logs
grep '\[MP4Reader\]' preprocess.log

# Show only transform logs
grep 'loom.transforms' preprocess.log
```

### By Timestamp

```bash
# Show logs between 10:30 and 10:31
awk '/10:30:00/,/10:31:00/' preprocess.log
```

## Log Rotation

For long-running services, rotate logs:

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "preprocess.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,  # Keep 5 old logs
)

logger.addHandler(handler)
```

## Best Practices

1. **Use lazy formatting**: `logger.info("Msg: %s", value)` not `logger.info(f"Msg: {value}")`
2. **Log exceptions with `exc_info=True`**: Captures full traceback
3. **Don't log in hot loops**: Use DEBUG level or sample (log every Nth iteration)
4. **Use structured tags**: `[ComponentName]` for easy filtering
5. **Include context**: Episode index, file path, frame number
6. **Log at boundaries**: Start/end of major operations
7. **Log resource usage**: Memory, disk, time for long operations

## Future: Structured Logging

Migrate to structured logging (JSON logs):

```python
import structlog

logger = structlog.get_logger()
logger.info(
    "episode_written",
    episode_idx=0,
    timesteps=450,
    duration_seconds=45.0,
)

# Output:
# {"event": "episode_written", "episode_idx": 0, "timesteps": 450, "duration_seconds": 45.0, "timestamp": "2025-01-15T10:30:00Z"}
```

**Benefits**:
- Machine-parseable
- Easy to query (jq, Elasticsearch)
- Consistent format
