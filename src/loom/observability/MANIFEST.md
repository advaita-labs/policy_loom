# Manifest Specification

## Purpose

Every preprocessing run produces a **manifest** that records:
- What was processed
- How it was processed
- When it was processed
- What was produced

This enables reproducibility, debugging, and auditing.

## Manifest Location

```
{output_path}/manifest.json
```

One manifest per dataset (not per episode).

## Manifest Schema

```json
{
  "version": "0.1.0",
  "created_at": "2025-01-15T10:30:00Z",
  "input": {
    "type": "mp4",
    "path": "./data/demo.mp4",
    "checksum": "sha256:a1b2c3d4...",
    "size_bytes": 123456789
  },
  "output": {
    "path": "./output/demo_processed",
    "num_episodes": 1,
    "total_timesteps": 450,
    "total_size_bytes": 5242880
  },
  "config": {
    "hash": "sha256:e5f6g7h8...",
    "transforms": [
      {"type": "resample_fps", "target_fps": 10},
      {"type": "resize", "height": 224, "width": 224},
      {"type": "normalize", "preset": "imagenet"}
    ],
    "metadata": {
      "robot_type": "franka_panda",
      "task": "pick_and_place"
    }
  },
  "statistics": {
    "processing_time_seconds": 42.3,
    "samples_per_second": 10.6,
    "dropped_samples": 0,
    "errors": 0,
    "warnings": 3
  },
  "versions": {
    "policy_loom": "0.1.0",
    "python": "3.10.13",
    "numpy": "1.26.0",
    "opencv": "4.8.1",
    "pyarrow": "14.0.1"
  },
  "environment": {
    "hostname": "workstation-01",
    "platform": "Linux-5.15.0-x86_64",
    "user": "researcher"
  }
}
```

## Field Descriptions

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Manifest schema version (semver) |
| `created_at` | string | Yes | ISO 8601 timestamp (UTC) |
| `input` | object | Yes | Input source metadata |
| `output` | object | Yes | Output dataset metadata |
| `config` | object | Yes | Processing configuration |
| `statistics` | object | Yes | Processing statistics |
| `versions` | object | Yes | Dependency versions |
| `environment` | object | No | Execution environment |

### `input` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Input type ("mp4", "mcap", etc.) |
| `path` | string | Yes | Absolute or relative path |
| `checksum` | string | No | File checksum (sha256) |
| `size_bytes` | integer | No | Total input size |

### `output` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | Yes | Output directory path |
| `num_episodes` | integer | Yes | Number of episodes written |
| `total_timesteps` | integer | Yes | Total timesteps across all episodes |
| `total_size_bytes` | integer | No | Total output size |

### `config` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `hash` | string | Yes | Config hash (sha256 of serialized config) |
| `transforms` | array | Yes | List of transforms applied |
| `metadata` | object | No | User-provided metadata |

### `statistics` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `processing_time_seconds` | float | Yes | Total wall-clock time |
| `samples_per_second` | float | Yes | Throughput |
| `dropped_samples` | integer | Yes | Number of dropped samples |
| `errors` | integer | Yes | Number of errors |
| `warnings` | integer | Yes | Number of warnings |

### `versions` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `policy_loom` | string | Yes | policy_loom version |
| `python` | string | Yes | Python version |
| `numpy` | string | Yes | NumPy version |
| Other deps | string | No | Optional dependency versions |

### `environment` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `hostname` | string | No | Machine hostname |
| `platform` | string | No | OS and architecture |
| `user` | string | No | Username who ran the pipeline |

## Generating the Manifest

```python
import hashlib
import json
from datetime import datetime
from pathlib import Path

def generate_manifest(
    input_path: Path,
    output_path: Path,
    config: dict,
    statistics: dict,
) -> dict:
    """Generate a manifest for a preprocessing run."""

    # Compute config hash
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()

    # Get dependency versions
    import numpy as np
    import policy_loom

    manifest = {
        "version": "0.1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input": {
            "type": config["input"]["type"],
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size if input_path.is_file() else None,
        },
        "output": {
            "path": str(output_path),
            "num_episodes": len(list((output_path / "episodes").iterdir())),
            "total_timesteps": statistics.get("total_timesteps", 0),
        },
        "config": {
            "hash": f"sha256:{config_hash}",
            "transforms": config.get("transforms", []),
            "metadata": config.get("metadata", {}),
        },
        "statistics": statistics,
        "versions": {
            "policy_loom": policy_loom.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
        },
    }

    return manifest


def write_manifest(manifest: dict, output_path: Path) -> None:
    """Write manifest to output directory."""
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
```

## Reading the Manifest

```python
import json
from pathlib import Path

def load_manifest(dataset_path: Path) -> dict:
    """Load manifest from a dataset."""
    manifest_path = dataset_path / "manifest.json"
    with manifest_path.open("r") as f:
        return json.load(f)


# Usage
manifest = load_manifest(Path("./output/demo_processed"))
print(f"Dataset created at: {manifest['created_at']}")
print(f"Number of episodes: {manifest['output']['num_episodes']}")
print(f"Config hash: {manifest['config']['hash']}")
```

## Use Cases

### Reproduce a Run

```python
# Load manifest
manifest = load_manifest(Path("./output/demo_processed"))

# Extract config
config = manifest["config"]

# Re-run with same config
runner = PreprocessRunner(config=config)
runner.run()
```

### Compare Runs

```python
manifest_a = load_manifest(Path("./output/run_a"))
manifest_b = load_manifest(Path("./output/run_b"))

if manifest_a["config"]["hash"] != manifest_b["config"]["hash"]:
    print("Configs differ!")
    print("Diff:", diff_configs(manifest_a["config"], manifest_b["config"]))
```

### Audit Trail

```python
# List all datasets
datasets = Path("./output").glob("*/")

for dataset in datasets:
    manifest = load_manifest(dataset)
    print(f"{dataset.name}:")
    print(f"  Created: {manifest['created_at']}")
    print(f"  Episodes: {manifest['output']['num_episodes']}")
    print(f"  Transforms: {len(manifest['config']['transforms'])}")
```

## Validation

Check manifest validity:

```python
def validate_manifest(manifest: dict) -> bool:
    # Required fields
    assert "version" in manifest
    assert "created_at" in manifest
    assert "input" in manifest
    assert "output" in manifest
    assert "config" in manifest
    assert "statistics" in manifest
    assert "versions" in manifest

    # Check types
    assert isinstance(manifest["output"]["num_episodes"], int)
    assert isinstance(manifest["statistics"]["processing_time_seconds"], (int, float))

    # Check timestamps
    datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))

    return True
```

## Future Extensions

- **Dataset lineage**: Track parent datasets (e.g., "derived from dataset X")
- **Diff manifests**: Built-in diff tool for comparing runs
- **Compression**: Store config as compressed blob for large configs
- **Signatures**: Cryptographic signatures for data provenance
