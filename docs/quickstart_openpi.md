# Quickstart: Preprocessing to OpenPI

Get started with `policy_loom` in 5 minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/policy_loom.git
cd policy_loom

# Install with dev dependencies
uv sync --extra dev

# Activate environment
source .venv/bin/activate
```

## Quick Example: MP4 → OpenPI

### 1. Prepare Input Data

```bash
mkdir -p data
# Place your video file in data/demo.mp4
```

### 2. Run Preprocessing

```bash
uv run python -m loom.runners.preprocess \
    --config configs/preprocess_mp4_to_openpi.yaml
```

### 3. Check Output

```bash
tree output/demo_processed
```

Expected structure:
```
output/demo_processed/
├── manifest.json
└── episodes/
    └── episode_000000/
        ├── metadata.json
        ├── observations.parquet
        └── videos/
            └── camera_0.mp4
```

### 4. Inspect Data

```python
import pandas as pd
import json

# Load manifest
with open("output/demo_processed/manifest.json") as f:
    manifest = json.load(f)
print(f"Episodes: {manifest['output']['num_episodes']}")

# Load observations
df = pd.read_parquet("output/demo_processed/episodes/episode_000000/observations.parquet")
print(df.head())
```

## Common Use Cases

### Video-Only Dataset

```yaml
# configs/video_only.yaml
input:
  type: mp4
  path: "./data/*.mp4"  # Glob pattern

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
  path: "./output/video_dataset"
```

Run:
```bash
uv run python -m loom.runners.preprocess --config configs/video_only.yaml
```

### Robot Teleoperation (MCAP)

```yaml
# configs/robot_teleop.yaml
input:
  type: mcap
  path: "./data/recording.mcap"
  topics:
    image: "/camera/image_raw"
    proprio: "/joint_states"
    action: "/action_commands"

transforms:
  - type: resample_fps
    target_fps: 10
  - type: resize
    height: 224
    width: 224

output:
  type: openpi
  path: "./output/robot_dataset"

metadata:
  robot_type: "franka_panda"
  task: "pick_and_place"
```

Run:
```bash
uv run python -m loom.runners.preprocess --config configs/robot_teleop.yaml
```

## What You Get

### Manifest (`manifest.json`)

- Input/output paths
- Configuration hash
- Processing statistics
- Dependency versions

Use for reproducibility and auditing.

### Episodes

Each episode is a self-contained directory:
- `metadata.json`: Episode-level info
- `observations.parquet`: Timestamps + proprioception
- `actions.parquet`: Commanded actions (if present)
- `videos/`: Video files (if images present)

### Loading Data in PyTorch

```python
from torch.utils.data import Dataset
import pandas as pd
import cv2

class OpenPIDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.episodes = sorted((self.root / "episodes").iterdir())

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode_dir = self.episodes[idx]

        # Load tables
        obs = pd.read_parquet(episode_dir / "observations.parquet")
        actions = pd.read_parquet(episode_dir / "actions.parquet")

        # Load first video frame (example)
        video_path = episode_dir / "videos" / "camera_0.mp4"
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        return {
            "observations": obs.to_numpy(),
            "actions": actions.to_numpy(),
            "image": frame,
        }

# Usage
dataset = OpenPIDataset("output/demo_processed")
sample = dataset[0]
```

## Next Steps

- **Read the design doc**: `docs/design_ingest_preprocess.md`
- **Explore transforms**: `src/loom/transforms/CATALOG.md`
- **Add custom transforms**: Subclass `loom.core.Transform`
- **Write tests**: See `tests/` for examples

## Troubleshooting

**"Output directory already exists"**
→ Use `overwrite: true` in config or delete output directory

**"No module named 'av'"**
→ Install video dependencies: `uv pip install av`

**"Failed to sync messages"**
→ Increase `time_tolerance_ms` in MCAP config

**"Out of memory"**
→ Reduce buffer size or process smaller batches

## Getting Help

- GitHub Issues: https://github.com/your-org/policy_loom/issues
- Documentation: `docs/`
- Examples: `configs/`
