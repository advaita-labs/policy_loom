# Testing Guide

policy_loom currently provides a diffusion-policy training pipeline that depends on optional extras. Running the test
suite inside an isolated virtual environment keeps those dependencies separate from your base installation.

## Environment Setup

```bash
# Create and activate a dedicated environment
python3 -m venv venv-diffusion
source venv-diffusion/bin/activate

# Install diffusion extras
cd policy_loom
uv sync --extra diffusion
```

## Test Plan

### 1. Sanity-check Imports

```bash
python3 -c "
from loom.core.types import Sample, CameraImage
from loom.training.adapters import DiffusionPolicyAdapter
print('✓ Base imports working')
"
```

### 2. Verify Adapter Registry

```bash
python3 -c "
from loom.training.adapter import list_adapters
adapters = list_adapters()
print(f'Available adapters: {adapters}')
assert 'diffusion_policy' in adapters, 'DiffusionPolicy should be registered'
print('✓ Adapter registry ready')
"
```

### 3. Run Automated Tests

Run the targeted test suite with diffusion dependencies available:

```bash
uv run pytest -v --tb=short
```

All tests should pass before making changes or opening a pull request.
