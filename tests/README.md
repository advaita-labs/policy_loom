# Tests

## Overview

Test suite for `policy_loom`. Follows Test-Driven Development (TDD) principles.

## Structure

```
tests/
├── ingest/              # Tests for data readers (mp4, mcap)
│   ├── test_mp4.py
│   └── test_mcap.py
├── transforms/          # Tests for sample transforms
│   ├── test_vision.py
│   └── test_time.py
├── export_openpi/       # Tests for OpenPI writer
│   └── test_writer.py
├── goldens/             # Golden outputs for regression tests
│   ├── sample_episode/
│   └── expected_outputs/
└── test_core.py         # Tests for core types and ports
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
uv run pytest tests/ingest/

# Test a file
uv run pytest tests/ingest/test_mp4.py

# Test a function
uv run pytest tests/ingest/test_mp4.py::test_read_frames
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

### Golden Tests

For outputs that should remain consistent:

```python
def test_openpi_writer_golden(tmp_path):
    """Test OpenPI writer produces expected output."""
    # Write samples
    writer = OpenPIWriter(tmp_path / "output")
    for sample in generate_test_samples():
        writer.write(sample)
    writer.close()

    # Compare with golden
    expected = Path("tests/goldens/expected_episode")
    assert_directories_equal(tmp_path / "output", expected)
```

## Test Categories

### Unit Tests

Test individual functions/classes in isolation.

**Example**: Test `Resize` transform with known inputs/outputs.

### Integration Tests

Test multiple components working together.

**Example**: Test full pipeline (read → transform → write).

### Regression Tests

Use golden outputs to detect unintended changes.

**Example**: Compare writer output with stored golden files.

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
- Every commit (via pre-commit hooks)
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
