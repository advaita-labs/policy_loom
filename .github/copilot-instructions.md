# Policy Loom AI Coding Instructions

## Project Overview
Open-source toolkit for preprocessing and training Vision-Language-Action (VLA) models for robotics and embodied AI applications. Uses hexagonal architecture with ports & adapters pattern for model-agnostic data pipelines.

## Development Philosophy

**Before writing code, always:**
1. Understand the requirements and constraints
2. Design the solution at a high level
3. Break down into smaller tasks
4. Identify potential challenges
5. **Keep things simple** - don't over-engineer
6. Keep models modular and composable
7. Follow best practices

## Critical Knowledge

### Import Convention: `loom` not `policy_loom`
- **Distribution name**: `policy-loom` (PyPI, with hyphen)
- **Import name**: `loom` (always use `from loom import ...`)
- The `src/policy_loom/` folder is metadata only; all code lives in `src/loom/`

### Multi-Environment Architecture
**Critical**: Pi0.5 and DiffusionPolicy cannot coexist in the same virtualenv due to conflicting transformers versions.

```bash
# DiffusionPolicy environment
uv sync --extra diffusion

# Pi0.5 environment (requires separate venv)
python -m venv venv-pi05 && source venv-pi05/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
```

When importing Pi0.5-specific code, always wrap in try/except with clear error messages pointing to installation docs.

### Core Data Flow
```
Reader → Sample → Transform → Sample → Preprocessor → Model Input → Trainer → Checkpoints
```

1. **Readers** (`loom.io.*`): Convert raw data (MP4/MCAP) to `Sample` objects with absolute timestamps
2. **Pipeline** (`loom.pipeline.merge_streams`): Temporal alignment using **nearest neighbor matching** (no interpolation)
3. **Preprocessors** (`loom.preprocessing.*`): Model-specific batching and normalization
4. **Adapters** (`loom.training.adapters.*`): Model-specific training logic implementing `ModelAdapter` protocol
5. **Trainer** (`loom.training.Trainer`): Generic training loop using adapters

### Temporal Synchronization
Video files have relative timestamps (0.0s), MCAP has absolute Unix timestamps. Always use `SynchronizedVideoMCAPReader` to extract MCAP camera timestamps and apply to video frames:

```python
from loom.io.synchronized import SynchronizedVideoMCAPReader
from loom.pipeline import merge_streams

reader = SynchronizedVideoMCAPReader(
    video_path="cam.mp4",
    mcap_path="data.mcap",
    camera_topic="arm/perception_interface/cam/state",
    camera_name="cam"
)
```

`merge_streams()` uses nearest neighbor (default tolerance: 33ms) - **never interpolates actions** (discrete robot states).

### Hexagonal Architecture Patterns

All pipeline components implement protocols in `loom.core.ports`:
- `Reader`: Input sources yielding `Sample` streams
- `Transform`: Stateless sample transformations
- `Preprocessor[TInput, TBatchInput]`: Model-specific preprocessing with typed inputs

Model training uses adapter pattern (`ModelAdapter` protocol) in `loom.training.adapter`. Register adapters with:
```python
from loom.training.adapter import register_adapter

@register_adapter("my_model")
class MyModelAdapter:
    def create_model(self) -> torch.nn.Module: ...
    def training_step(self, model, batch, device) -> tuple[Tensor, dict]: ...
    def eval_step(self, model, batch, device) -> dict: ...
    def create_dataloaders(self, train_ds, eval_ds, batch_size, num_workers): ...
```

## Code Standards

### Style & Tooling
- Follow PEP 8 style guide
- Python 3.11+, max line length 120 chars
- **All functions require type hints** (strict mypy mode)
- Google-style docstrings for all public APIs
- Format: `black src/ tests/` + `isort src/ tests/`
- Lint: `ruff check src/ tests/`
- Type check: `mypy src/`
- Use dataclasses for configs/types (see `loom.core.types.Sample`)
- Prefer `pathlib.Path` over `os.path`
- Use `logging` module, never `print()` for application code

### Type Hints
Always include return types and parameter types:
```python
def preprocess_sample(self, sample: Sample) -> DiffusionPolicyInput:
    """Convert Sample to model input format."""
    ...
```

### Docstrings
Google-style docstrings with Args/Returns/Raises. Include usage examples for public APIs:
```python
def merge_streams(*readers: Reader, time_tolerance: float = 0.033) -> Iterator[Sample]:
    """Merge multiple readers into temporally aligned samples.
    
    Args:
        readers: Data readers to merge
        time_tolerance: Max time difference in seconds for grouping
        
    Returns:
        Iterator of merged Sample objects
        
    Example:
        >>> for sample in merge_streams(left_cam, right_cam):
        ...     print(f"{len(sample.cameras)} cameras")
    """
```

### Error Handling
Always use proper error handling with logging:
```python
import logging

logger = logging.getLogger(__name__)

try:
    model = load_vla_model(config.model_path)
except FileNotFoundError:
    logger.error(f"Model not found at {config.model_path}")
    raise
except RuntimeError as e:
    logger.error(f"Failed to load model: {e}")
    raise ModelLoadError(f"Could not initialize model: {e}") from e
```

### What to Avoid
- Hardcoded paths or credentials
- Large files in git (use git-lfs or external storage)
- Undocumented magic numbers
- Global state or mutable defaults
- Catching exceptions without logging
- Using `print()` instead of proper logging
- Blocking the main thread with I/O operations
- Unnecessary dependencies
- Breaking changes without deprecation warnings

## Testing

### Test-Driven Development
Write tests first. Aim for >80% coverage (check with `uv run pytest --cov=loom --cov-report=html`).

```bash
# Run all tests
uv run pytest -v

# Run specific test
uv run pytest tests/test_pipeline.py::TestMergeStreams::test_merge_multiple_proprio_uses_nearest -v

# Test with coverage
uv run pytest --cov=loom --cov-report=term-missing
```

**Testing Best Practices:**
- Write unit tests for all utility functions
- Include integration tests for pipelines
- Test structure mirrors `src/loom/` structure
- Use small synthetic datasets (avoid fixtures >1MB)
- Mock expensive operations (model inference, I/O, GPU ops)
- Check for reproducibility (random seeds, deterministic ops)
- Verify memory efficiency and GPU utilization

## Configuration

YAML configs in `configs/` use dataclasses from `loom.training.config`. Key sections:
- `model`: Adapter type + model-specific params
- `training`: Hyperparameters (lr, batch_size, epochs)
- `data`: Dataset paths and preprocessing
- `checkpoints`: Saving strategy (steps/epochs, keep_top_k)
- `logging`: Wandb/local logging config

Example structure:
```yaml
model:
  type: diffusion  # or pi05
  # ... model-specific config

training:
  learning_rate: 1e-4
  batch_size: 32
  epochs: 100
```

Load with: `config = TrainingConfig.from_yaml("config.yaml")`

## CLI Usage

```bash
# Train with config
loom train configs/diffusion_minimal.yaml

# List available model adapters
loom list-adapters

# Preprocess data (planned feature)
loom preprocess <input> <output> --format openpi
```

## Common Patterns

### Adding a New Model Adapter
1. Create `src/loom/training/adapters/my_model.py`
2. Implement `ModelAdapter` protocol
3. Register with `@register_adapter("my_model")`
4. Add dependencies to `pyproject.toml` optional-dependencies
5. Create example config in `configs/my_model_example.yaml`
6. Add tests in `tests/training/adapters/test_my_model.py`

### Adding a New Reader
1. Create `src/loom/io/my_format/reader.py`
2. Inherit from `Reader` protocol
3. Implement `read() -> Iterator[Sample]`
4. Add tests in `tests/test_io.py`
5. Export from `src/loom/io/my_format/__init__.py`

## Design Principles

1. **Simplicity over abstraction**: Code should be self-explanatory
2. **Late batching**: Keep individual samples until model preprocessing
3. **Temporal correctness**: No interpolation for discrete robot states
4. **Dataclass-first**: Use dataclasses over dicts for type safety
5. **Composability**: Components are Lego blocks (see `docs/design_ingest_preprocess.md`)

## Performance Best Practices

- Use DataLoader with multiple workers (`num_workers` in config)
- Implement efficient data augmentation pipelines
- Profile code to identify bottlenecks before optimizing
- Use `torch.compile()` for PyTorch 2.0+ models
- Implement gradient checkpointing for large models
- Pin memory (`pin_memory=True`) for faster GPU transfers
- Use mixed precision training (`mixed_precision=True`)

## Data Processing Guidelines

- Support multiple data formats (MP4, MCAP, LeRobot/HuggingFace datasets)
- Implement efficient batching and caching strategies
- Handle multi-modal data (images, text, actions) consistently
- Provide clear data transformation pipelines
- Include data validation and sanity checks
- Validate shape/dtype of tensors at pipeline boundaries

## Key Files Reference
- Architecture overview: `README.md` (includes mermaid diagram)
- Temporal sync details: `docs/SYNCHRONIZATION.md`
- Testing multi-env setup: `docs/TESTING_GUIDE.md`
- Pi0.5 configuration: `docs/PI05_CONFIG_GUIDE.md`
- Core types: `src/loom/core/types.py` (Sample, CameraImage)
- Protocols: `src/loom/core/ports.py` (Reader, Transform, Preprocessor)
- Merging logic: `src/loom/pipeline/merge.py` (nearest neighbor algorithm)
