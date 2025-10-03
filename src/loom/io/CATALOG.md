# Ingest Catalog

## Supported Formats

| Format | Short Name | Module | Status | Dependencies |
|--------|-----------|---------|--------|--------------|
| MP4 Video | `mp4` | `loom.io.mp4` | ✅ Implemented | `av` (PyAV) |
| MCAP/ROS 2 | `mcap` | `loom.io.mcap` | ✅ Implemented | `mcap`, `mcap-ros2-support` |

## Planned Formats

| Format | Short Name | Priority | Notes |
|--------|-----------|----------|-------|
| Image Directory | `images` | High | Sequential images + timestamps.csv |
| HDF5 | `hdf5` | Medium | Common in older datasets |
| ROS 1 Bag | `rosbag1` | Low | Use `rosbag2mcap` converter instead |
| Zarr | `zarr` | Low | For large-scale datasets |

## Adding a New Format

1. Create a new directory under `src/loom/io/<format>/`
2. Implement a `Reader` subclass
3. Add README.md documenting:
   - Output schema (what fields are populated)
   - Timing rules
   - Dependencies
   - Usage examples
   - Limitations
4. Add tests in `tests/ingest/<format>/`
5. Update this CATALOG.md
6. Add to `src/loom/io/__init__.py`

## Format Selection

Readers are selected by:
1. **Explicit type**: `MP4Reader("video.mp4")`
2. **Config-driven**: `create_reader(path, format="mp4")`
3. **Auto-detection**: `auto_reader(path)` (guesses from extension)

Prefer explicit types in code, config-driven in YAML pipelines.
