# Tests

## Overview

Test suite for `policy_loom`. Follows Test-Driven Development (TDD) principles.

## Structure

```
tests/
├── test_core.py           # Core types (Sample, CameraImage)
├── test_io.py             # Data readers (MP4, MCAP)
├── test_pipeline.py       # Stream merging and synchronization
├── test_synchronized.py   # Synchronized video+MCAP reader
├── preprocessing/         # Preprocessor tests (SmolVLA, DiffusionPolicy)
│   ├── test_smolvla.py
│   └── test_diffusion_policy.py
├── training/              # Training infrastructure tests
│   ├── test_adapter.py
│   ├── test_config.py
│   ├── test_trainer.py
│   └── adapters/
│       └── test_diffusion_policy.py
└── cli/                   # CLI tests
    └── test_cli.py
```

## Running Tests

### All Tests

```bash
# Using pytest
uv run pytest

# With coverage
uv run pytest --cov=loom --cov-report=html

# Verbose
uv run pytest -v
```

### Specific Tests

```bash
# Test a module
uv run pytest tests/training/

# Test a file
uv run pytest tests/test_io.py

# Test a function
uv run pytest tests/test_io.py::test_mp4_reader
```

## Writing Tests

### Test Template

```python
import pytest
from loom.core import Sample

def test_sample_creation():
    """Test Sample dataclass initialization."""
    sample = Sample(
        timestamp=0.0,
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),
    )
    assert sample.timestamp == 0.0
    assert sample.rgb.shape == (224, 224, 3)

def test_sample_validation():
    """Test Sample validation in __post_init__."""
    with pytest.raises(ValueError, match="Expected rgb to have 3 dimensions"):
        Sample(timestamp=0.0, rgb=np.zeros((224, 224)))
```

### Using Fixtures

```python
@pytest.fixture
def sample_video(tmp_path):
    """Create a small test video."""
    video_path = tmp_path / "test.mp4"
    # Generate test video
    return video_path

def test_mp4_reader(sample_video):
    """Test MP4 reader with fixture."""
    reader = MP4Reader(sample_video)
    samples = list(reader.read())
    assert len(samples) > 0
```

### Regression Tests

Test that model outputs remain consistent:

```python
def test_preprocessor_output(sample_data):
    """Test preprocessor produces consistent output."""
    preprocessor = DiffusionPolicyPreprocessor(config)
    output = preprocessor.preprocess_sample(sample_data)

    # Verify output shape and values
    assert output.state.shape == (obs_horizon, state_dim)
    assert output.action.shape == (action_horizon, action_dim)
```

## Test Categories

### Unit Tests

Test individual functions/classes in isolation.

**Example**: Test `MP4Reader` with known inputs/outputs.

### Integration Tests

Test multiple components working together.

**Example**: Test full training pipeline (data loading → preprocessing → training).

### Regression Tests

Use golden outputs to detect unintended changes.

**Example**: Compare preprocessor output with expected values.

## Mocking

Mock expensive operations:

```python
from unittest.mock import Mock, patch

@patch('loom.io.mp4.av.open')
def test_mp4_reader_mock(mock_av_open):
    """Test MP4 reader with mocked av.open."""
    mock_container = Mock()
    mock_av_open.return_value = mock_container

    reader = MP4Reader("fake.mp4")
    mock_av_open.assert_called_once_with("fake.mp4")
```

## Synthetic Data

Create small, synthetic datasets for testing:

```python
def create_synthetic_video(path, num_frames=10, fps=10):
    """Create a small test video."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))

    for i in range(num_frames):
        frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()
```

## Coverage Goals

- **Core modules**: 100%
- **IO/Transforms**: 80%+
- **Integration**: 70%+

Check coverage:
```bash
uv run pytest --cov=loom --cov-report=term-missing
```

## CI/CD

Tests run automatically on:
- Every pull request (via GitHub Actions)
- Before releases

## Test Utilities

Create common test utilities in `tests/conftest.py`:

```python
# tests/conftest.py
import pytest
import numpy as np
from loom.core import Sample

@pytest.fixture
def dummy_sample():
    """Create a dummy sample for testing."""
    return Sample(
        timestamp=0.0,
        rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        proprio=np.zeros(6, dtype=np.float32),
        action=np.zeros(7, dtype=np.float32),
        metadata={"episode_id": "test"},
    )
```

## Debugging Tests

```bash
# Run with debugger
uv run pytest --pdb

# Print statements (disable capture)
uv run pytest -s

# Last failed tests only
uv run pytest --lf
```
