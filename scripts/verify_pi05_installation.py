#!/usr/bin/env python3
"""
Pi0.5 Integration Verification Script

This script verifies that all components of the Pi0.5 integration are working correctly.
Run this after installing Pi0.5 dependencies to confirm everything is set up properly.

Usage:
    uv run python scripts/verify_pi05_installation.py
"""

import sys
from pathlib import Path


def print_header(title: str):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def test_openpi_installation():
    """Test 1: Verify OpenPI is installed."""
    print("1. Checking OpenPI installation...")
    try:
        import openpi
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        
        print("   [OK] OpenPI package imported successfully")
        print(f"   [OK] Pi0Config available")
        print(f"   [OK] PI0Pytorch model available")
        return True
    except ImportError as e:
        print(f"   [FAIL] OpenPI not installed: {e}")
        print()
        print("   To install OpenPI:")
        print("   1. Create separate venv: python -m venv venv-pi05")
        print("   2. Activate: source venv-pi05/bin/activate")
        print("   3. Install: GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05")
        return False


def test_transformers_patches():
    """Test 2: Verify transformers patches are applied."""
    print()
    print("2. Checking transformers patches...")
    try:
        import transformers
        from transformers.models.siglip import check
        
        version_ok = transformers.__version__ == "4.53.2"
        patches_ok = check.check_whether_transformers_replace_is_installed_correctly()
        
        if version_ok and patches_ok:
            print(f"   [OK] Transformers version: {transformers.__version__}")
            print("   [OK] AdaRMS patches applied correctly")
            return True
        else:
            if not version_ok:
                print(f"   [FAIL] Wrong transformers version: {transformers.__version__} (need 4.53.2)")
            if not patches_ok:
                print("   [FAIL] Transformers patches not applied")
            print()
            print("   To apply patches:")
            print("   cp -r .venv/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/* \\")
            print("         .venv/lib/python3.11/site-packages/transformers/")
            return False
    except Exception as e:
        print(f"   [FAIL] Transformers check failed: {e}")
        return False


def test_adapter_registration():
    """Test 3: Verify Pi05Adapter is registered."""
    print()
    print("3. Checking Pi05Adapter registration...")
    try:
        from loom.training.adapter import list_adapters, get_adapter
        
        adapters = list_adapters()
        if 'pi05' in adapters:
            print(f"   [OK] Pi05Adapter registered")
            print(f"   [OK] Available adapters: {adapters}")
            return True
        else:
            print(f"   [FAIL] Pi05Adapter not found in: {adapters}")
            return False
    except Exception as e:
        print(f"   [FAIL] Adapter registration check failed: {e}")
        return False


def test_adapter_initialization():
    """Test 4: Verify adapter can be initialized."""
    print()
    print("4. Testing Pi05Adapter initialization...")
    try:
        from loom.training.adapters.pi05 import Pi05Adapter
        
        config = {
            'action_dim': 32,
            'action_horizon': 10,
            'max_token_len': 256,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
        }
        
        adapter = Pi05Adapter(config)
        
        print("   [OK] Adapter created successfully")
        print(f"     - Action dim: {adapter.action_dim}")
        print(f"     - Action horizon: {adapter.action_horizon}")
        print(f"     - Max token len: {adapter.max_token_len}")
        return True
    except Exception as e:
        print(f"   [FAIL] Adapter initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test 5: Verify model can be created."""
    print()
    print("5. Testing Pi0.5 model creation...")
    try:
        from loom.training.adapters.pi05 import Pi05Adapter
        import torch
        
        config = {
            'action_dim': 32,
            'action_horizon': 10,
            'max_token_len': 256,
        }
        
        adapter = Pi05Adapter(config)
        model = adapter.create_model()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("   [OK] Model created successfully")
        print(f"     - Type: {type(model).__name__}")
        print(f"     - Pi05 flag: {model.pi05}")
        print(f"     - Device: {device}")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_creation():
    """Test 6: Verify optimizer can be created."""
    print()
    print("6. Testing optimizer creation...")
    try:
        from loom.training.adapters.pi05 import Pi05Adapter
        
        config = {'action_dim': 32, 'action_horizon': 10, 'max_token_len': 256}
        adapter = Pi05Adapter(config)
        model = adapter.create_model()
        optimizer = adapter.create_optimizer(model, lr=1e-4, weight_decay=0.01)
        
        print("   [OK] Optimizer created successfully")
        print(f"     - Type: {type(optimizer).__name__}")
        print(f"     - Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"     - Weight decay: {optimizer.param_groups[0]['weight_decay']:.2e}")
        
        return True
    except Exception as e:
        print(f"   [FAIL] Optimizer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer():
    """Test 7: Verify tokenizer works."""
    print()
    print("7. Testing tokenizer...")
    try:
        from loom.training.adapters.pi05 import Pi05Adapter
        
        config = {'action_dim': 32, 'action_horizon': 10, 'max_token_len': 256}
        adapter = Pi05Adapter(config)
        tokenizer = adapter._get_tokenizer()
        
        if tokenizer:
            # Test tokenization
            tokens, mask = tokenizer.tokenize('Pick and place the cube on the table')
            
            print("   [OK] Tokenizer working")
            print(f"     - Type: {type(tokenizer).__name__}")
            print(f"     - Tokens shape: {tokens.shape}")
            print(f"     - Mask shape: {mask.shape}")
            print(f"     - Max length: {adapter.max_token_len}")
            return True
        else:
            print("   [FAIL] Tokenizer not available")
            return False
    except Exception as e:
        print(f"   [FAIL] Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_file():
    """Test 8: Verify configuration file exists."""
    print()
    print("8. Checking configuration file...")
    try:
        config_path = Path("configs/pi05_minimal.yaml")
        
        if config_path.exists():
            print(f"   [OK] Configuration file found: {config_path}")
            
            # Check file size
            size = config_path.stat().st_size
            print(f"     - Size: {size} bytes")
            
            # Try to read it
            content = config_path.read_text()
            if 'model:' in content and 'type: pi05' in content:
                print("     - Contains valid Pi05 configuration")
                return True
            else:
                print("     [FAIL] Configuration file missing required fields")
                return False
        else:
            print(f"   [FAIL] Configuration file not found: {config_path}")
            return False
    except Exception as e:
        print(f"   [FAIL] Configuration file check failed: {e}")
        return False


def test_documentation():
    """Test 9: Verify documentation exists."""
    print()
    print("9. Checking documentation...")
    try:
        guide_path = Path("docs/PI05_TRAINING_GUIDE.md")
        summary_path = Path("PI05_IMPLEMENTATION_SUMMARY.md")
        
        results = []
        
        if guide_path.exists():
            print(f"   [OK] Training guide found: {guide_path}")
            size = guide_path.stat().st_size
            print(f"     - Size: {size:,} bytes")
            results.append(True)
        else:
            print(f"   [FAIL] Training guide not found: {guide_path}")
            results.append(False)
        
        if summary_path.exists():
            print(f"   [OK] Implementation summary found: {summary_path}")
            size = summary_path.stat().st_size
            print(f"     - Size: {size:,} bytes")
            results.append(True)
        else:
            print(f"   [FAIL] Implementation summary not found: {summary_path}")
            results.append(False)
        
        return all(results)
    except Exception as e:
        print(f"   [FAIL] Documentation check failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print_header("Pi0.5 Integration Verification")
    
    tests = [
        ("OpenPI Installation", test_openpi_installation),
        ("Transformers Patches", test_transformers_patches),
        ("Adapter Registration", test_adapter_registration),
        ("Adapter Initialization", test_adapter_initialization),
        ("Model Creation", test_model_creation),
        ("Optimizer Creation", test_optimizer_creation),
        ("Tokenizer", test_tokenizer),
        ("Configuration File", test_configuration_file),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"   [FAIL] Unexpected error: {e}")
            results.append((name, False))
    
    # Print summary
    print()
    print_header("Verification Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"  {status}  {name}")
    
    print()
    print(f"  Total: {passed}/{total} tests passed")
    print()
    
    if passed == total:
        print_header("[SUCCESS] SUCCESS: Pi0.5 Integration Verified!")
        print("All components are working correctly.")
        print()
        print("Next steps:")
        print("  1. Convert data: python scripts/convert_mp4_mcap_to_lerobot.py")
        print("  2. Start training: loom train configs/pi05_minimal.yaml")
        print("  3. Read guide: docs/PI05_TRAINING_GUIDE.md")
        print()
        return 0
    else:
        print_header("[FAILURE] FAILURE: Some Tests Failed")
        print(f"{total - passed} test(s) failed. Please review the errors above.")
        print()
        print("Common fixes:")
        print("  • Install Pi05: GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05")
        print("  • Apply patches: cp -r .venv/.../openpi/models_pytorch/transformers_replace/* ...")
        print("  • Check docs: docs/PI05_TRAINING_GUIDE.md")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
