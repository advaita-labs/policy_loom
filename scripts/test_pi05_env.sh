#!/bin/bash
# Test Pi0.5 environment setup
set -e

echo "============================================"
echo "Testing Pi0.5 Environment"
echo "============================================"
echo ""

# Check we're in the right venv
if [[ "$VIRTUAL_ENV" != *"venv-pi05"* ]]; then
    echo "⚠️  Warning: Not in venv-pi05"
    echo "   Run: source venv-pi05/bin/activate"
    exit 1
fi

echo "✓ In Pi0.5 virtual environment"
echo ""

echo "Test 1: Pi0.5 dependencies..."
python3 -c "
import lerobot
import datasets
from transformers import __version__ as tf_version
print('✓ lerobot available')
print('✓ datasets available')
print(f'✓ transformers: {tf_version}')
"

echo ""
echo "Test 2: LeRobot loader imports..."
python3 -c "
from loom.io.lerobot import LeRobotDatasetLoader
print('✓ LeRobotDatasetLoader imports')
"

echo ""
echo "Test 3: Pi05Adapter imports..."
python3 -c "
from loom.training.adapters import Pi05Adapter
print('✓ Pi05Adapter imports')
"

echo ""
echo "Test 4: Adapter registry..."
python3 -c "
from loom.training.adapter import list_adapters
adapters = list_adapters()
print(f'Available adapters: {adapters}')
assert 'pi05' in adapters, 'Pi05 should be registered'
print('✓ Adapter registry includes pi05')
"

echo ""
echo "Test 5: Pi05Adapter configuration..."
python3 << 'EOF'
from loom.training.adapters import Pi05Adapter

config = {
    "type": "pi05",
    "pretrained_model_name_or_path": "lerobot/pi05_base",
    "freeze_backbone": False,
}

adapter = Pi05Adapter(config)
print(f'✓ Pi05Adapter instantiates')
print(f'  Model: {adapter.pretrained_model_path}')
print(f'  Freeze: {adapter.freeze_backbone}')
EOF

echo ""
echo "Test 6: Training script CLI..."
python3 scripts/train_pi05.py --help > /dev/null 2>&1
echo "✓ Training script runs"

echo ""
echo "============================================"
echo "✓ Pi0.5 Environment: ALL TESTS PASSED"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Test dataset loading (requires internet):"
echo "     python3 scripts/test_pi05_env.sh --with-download"
echo "  2. Start training:"
echo "     python3 scripts/train_pi05.py --dataset lerobot/koch_test --output checkpoints/"
