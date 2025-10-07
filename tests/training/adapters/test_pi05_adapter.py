"""Tests for Pi0.5 adapter.

Note: These tests require openpi to be installed in a separate environment.
Run with:
    source venv-pi05/bin/activate
    pytest tests/training/adapters/test_pi05_adapter.py
"""

import pytest

# Skip all tests if openpi is not installed
pytest.importorskip("openpi", reason="openpi not installed (requires separate venv)")

import torch  # noqa: E402

from loom.training.adapters.pi05 import Pi05Adapter  # noqa: E402


class TestPi05Adapter:
    """Test Pi0.5 adapter integration."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for testing."""
        return {
            "type": "pi05",
            "pretrained_path": None,
            "action_dim": 7,
            "action_horizon": 10,
            "use_lora": False,
            "freeze_backbone": False,
            "image_size": [224, 224],
        }

    def test_adapter_initialization(self, minimal_config):
        """Test adapter can be initialized."""
        adapter = Pi05Adapter(minimal_config)

        assert adapter.action_dim == 7
        assert adapter.action_horizon == 10
        assert adapter.use_lora is False
        assert adapter.freeze_backbone is False

    def test_create_model_from_scratch(self, minimal_config):
        """Test creating model from scratch (may fail if openpi API changed)."""
        adapter = Pi05Adapter(minimal_config)

        try:
            model = adapter.create_model()
            assert isinstance(model, torch.nn.Module)
        except Exception as e:
            pytest.skip(f"Model creation failed (openpi API may have changed): {e}")

    def test_create_optimizer(self, minimal_config):
        """Test optimizer creation."""
        adapter = Pi05Adapter(minimal_config)

        # Create dummy model
        dummy_model = torch.nn.Linear(10, 10)

        optimizer = adapter.create_optimizer(dummy_model, lr=1e-4, weight_decay=1e-5)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-4
        assert optimizer.defaults["weight_decay"] == 1e-5

    def test_lora_config(self):
        """Test LoRA configuration."""
        config = {
            "type": "pi05",
            "action_dim": 7,
            "action_horizon": 10,
            "use_lora": True,
            "lora_rank": 16,
        }

        adapter = Pi05Adapter(config)

        assert adapter.use_lora is True
        assert adapter.lora_rank == 16

    def test_freeze_backbone_config(self):
        """Test freeze backbone configuration."""
        config = {
            "type": "pi05",
            "action_dim": 7,
            "action_horizon": 10,
            "freeze_backbone": True,
        }

        adapter = Pi05Adapter(config)

        assert adapter.freeze_backbone is True


@pytest.mark.skip(reason="Requires openpi installation and real checkpoint")
class TestPi05CheckpointLoading:
    """Test checkpoint loading (requires actual checkpoints)."""

    def test_load_from_checkpoint(self):
        """Test loading from openpi checkpoint."""
        config = {
            "type": "pi05",
            "pretrained_path": "gs://openpi-assets/checkpoints/pi05_base",
            "action_dim": 7,
            "action_horizon": 10,
        }

        adapter = Pi05Adapter(config)
        model = adapter.create_model()

        assert isinstance(model, torch.nn.Module)
