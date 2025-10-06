#!/bin/bash
# Test DiffusionPolicy environment setup
set -e

echo "============================================"
echo "Testing DiffusionPolicy Environment"
echo "============================================"
echo ""

# Check we're in the right venv
if [[ "$VIRTUAL_ENV" != *"venv-diffusion"* ]]; then
    echo "⚠️  Warning: Not in venv-diffusion"
    echo "   Run: source venv-diffusion/bin/activate"
    exit 1
fi

echo "✓ In DiffusionPolicy virtual environment"
echo ""

echo "Test 1: Base imports..."
python3 -c "
from loom.core.types import Sample, CameraImage
from loom.training.adapters import DiffusionPolicyAdapter
print('✓ Base imports working')
"

echo ""
echo "Test 2: Pi05 NOT available (expected)..."
python3 -c "
try:
    from loom.training.adapters.pi05 import Pi05Adapter
    print('✗ FAIL: Pi05Adapter should not be importable')
    exit(1)
except ImportError:
    print('✓ Pi05 correctly unavailable')
"

echo ""
echo "Test 3: Adapter registry..."
python3 -c "
from loom.training.adapter import list_adapters
adapters = list_adapters()
print(f'Available adapters: {adapters}')
assert 'diffusion_policy' in adapters
assert 'pi05' not in adapters
print('✓ Adapter registry correct')
"

echo ""
echo "============================================"
echo "✓ DiffusionPolicy Environment: ALL TESTS PASSED"
echo "============================================"
