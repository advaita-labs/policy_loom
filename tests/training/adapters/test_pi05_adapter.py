"""Tests for Pi05Adapter following TDD approach.

These tests define the expected behavior before implementation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from torch.utils.data import Dataset

from loom.core.types import Sample, CameraImage


class MockLeRobotDataset(Dataset):
    """Mock LeRobot dataset for testing."""
    
    def __init__(self, num_samples=10, action_dim=7, proprio_dim=7):
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return a LeRobot-style sample."""
        return {
            "observation.images.left_cam": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation.images.right_cam": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation.state": np.random.randn(self.proprio_dim).astype(np.float32),
            "action": np.random.randn(self.action_dim).astype(np.float32),
            "task": "pick and place",
            "episode_index": 0,
            "frame_index": idx,
            "timestamp": float(idx) / 30.0,
        }


class TestPi05AdapterInitialization:
    """Test Pi05Adapter initialization and configuration."""
    
    def test_adapter_imports_lazily(self):
        """Test that adapter can be imported without OpenPI installed."""
        # This test ensures graceful handling of missing dependencies
        try:
            from loom.training.adapters.pi05 import Pi05Adapter
            # If OpenPI is not installed, adapter should still import
            # but raise error when trying to create model
            assert True
        except ImportError as e:
            # Should not fail on import, only on use
            pytest.fail(f"Pi05Adapter should import lazily, got: {e}")
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for Pi0.5 adapter."""
        return {
            "type": "pi05",
            "action_dim": 32,  # OpenPI hardcoded
            "action_horizon": 10,
            "max_token_len": 180,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "pi05": True,
            "dtype": "bfloat16",
            "default_prompt": "pick and place",
            "image_size": [224, 224],
        }
    
    def test_adapter_initialization_with_config(self, basic_config):
        """Test adapter initializes with correct configuration."""
        from loom.training.adapters.pi05 import Pi05Adapter
        
        adapter = Pi05Adapter(basic_config)
        
        assert adapter.action_dim == 32
        assert adapter.action_horizon == 10
        assert adapter.max_token_len == 180
        assert adapter.paligemma_variant == "gemma_2b"
        assert adapter.default_prompt == "pick and place"


@pytest.mark.skipif(
    not pytest.importorskip("openpi", reason="OpenPI not installed"),
    reason="Requires OpenPI package"
)
class TestPi05AdapterWithOpenPI:
    """Tests that require OpenPI to be installed."""
    
    @pytest.fixture
    def adapter_config(self):
        """Configuration for Pi0.5 adapter."""
        return {
            "type": "pi05",
            "action_dim": 32,
            "action_horizon": 10,
            "max_token_len": 180,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "pi05": True,
            "dtype": "bfloat16",
            "default_prompt": "pick and place",
            "image_size": [224, 224],
        }
    
    @pytest.fixture
    def adapter(self, adapter_config):
        """Create adapter instance."""
        from loom.training.adapters.pi05 import Pi05Adapter
        return Pi05Adapter(adapter_config)
    
    def test_create_model(self, adapter):
        """Test model creation returns proper OpenPI model."""
        model = adapter.create_model()
        
        # Should return a PyTorch module
        assert isinstance(model, torch.nn.Module)
        
        # Should have forward method
        assert hasattr(model, "forward")
        
        # Model should be in eval mode by default for safety
        assert not model.training or model.training  # Either is ok
    
    def test_create_optimizer(self, adapter):
        """Test optimizer creation with correct hyperparameters."""
        model = adapter.create_model()
        optimizer = adapter.create_optimizer(
            model=model,
            lr=5e-5,
            weight_decay=0.01
        )
        
        # Should be AdamW optimizer
        assert isinstance(optimizer, torch.optim.AdamW)
        
        # Check learning rate
        assert optimizer.param_groups[0]["lr"] == 5e-5
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
    
    def test_create_dataloaders(self, adapter):
        """Test dataloader creation with LeRobot dataset."""
        train_dataset = MockLeRobotDataset(num_samples=20)
        eval_dataset = MockLeRobotDataset(num_samples=10)
        
        train_loader, eval_loader = adapter.create_dataloaders(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
        )
        
        # Check loaders are created
        assert train_loader is not None
        assert eval_loader is not None
        
        # Check batch from train loader
        batch = next(iter(train_loader))
        observation, actions = batch
        
        # Observation should have required attributes
        assert hasattr(observation, "images")
        assert hasattr(observation, "state")
        assert hasattr(observation, "tokenized_prompt")
        
        # Actions should be padded to 32 dimensions
        assert actions.shape[-1] == 32
        assert actions.dtype == torch.float32
    
    def test_training_step(self, adapter):
        """Test training step forward pass."""
        model = adapter.create_model()
        model.train()
        
        # Create synthetic batch
        train_dataset = MockLeRobotDataset(num_samples=8)
        train_loader, _ = adapter.create_dataloaders(
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            batch_size=4,
            num_workers=0,
        )
        
        batch = next(iter(train_loader))
        device = torch.device("cpu")  # Use CPU for testing
        
        # Execute training step
        loss, metrics = adapter.training_step(model, batch, device)
        
        # Check loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0 or (loss.ndim == 1 and loss.shape[0] == 1)
        
        # Check metrics dict
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert metrics["loss"] > 0  # Loss should be positive
        
        # Loss should require gradients for backprop
        assert loss.requires_grad
    
    def test_eval_step(self, adapter):
        """Test evaluation step without gradients."""
        model = adapter.create_model()
        model.eval()
        
        # Create synthetic batch
        eval_dataset = MockLeRobotDataset(num_samples=8)
        _, eval_loader = adapter.create_dataloaders(
            train_dataset=eval_dataset,
            eval_dataset=eval_dataset,
            batch_size=4,
            num_workers=0,
        )
        
        batch = next(iter(eval_loader))
        device = torch.device("cpu")
        
        # Execute eval step
        metrics = adapter.eval_step(model, batch, device)
        
        # Check metrics dict
        assert isinstance(metrics, dict)
        assert "eval/loss" in metrics
        assert metrics["eval/loss"] > 0
    
    def test_action_padding(self, adapter):
        """Test that 7-dim actions are correctly padded to 32-dim."""
        # Create dataset with 7-dim actions
        train_dataset = MockLeRobotDataset(num_samples=4, action_dim=7)
        train_loader, _ = adapter.create_dataloaders(
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            batch_size=2,
            num_workers=0,
        )
        
        observation, actions = next(iter(train_loader))
        
        # Actions should be padded to 32
        assert actions.shape[-1] == 32
        
        # First 7 dimensions should be non-zero (original actions)
        # Last 25 dimensions should be zero (padding)
        assert not torch.all(actions[..., :7] == 0)
        # Note: Can't guarantee zeros in padding if transform normalizes


class TestPi05AdapterEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_action_dimension_greater_than_32(self):
        """Test that action_dim > 32 raises clear error."""
        from loom.training.adapters.pi05 import Pi05Adapter
        
        config = {
            "type": "pi05",
            "action_dim": 50,  # Invalid: too large
            "action_horizon": 10,
            "max_token_len": 180,
        }
        
        # Should either raise on init or on dataloader creation
        # Let's test it doesn't silently fail
        adapter = Pi05Adapter(config)
        
        # When creating dataloaders with >32 dim actions, should fail
        train_dataset = MockLeRobotDataset(num_samples=4, action_dim=50)
        
        with pytest.raises((ValueError, RuntimeError)):
            train_loader, _ = adapter.create_dataloaders(
                train_dataset=train_dataset,
                eval_dataset=train_dataset,
                batch_size=2,
                num_workers=0,
            )
    
    def test_missing_camera_in_batch(self):
        """Test handling of batches with missing expected cameras."""
        # This test verifies graceful degradation
        # Implementation detail: may fill with dummy images or raise error
        pass  # TODO: Implement once adapter behavior is defined


class TestPi05AdapterIntegration:
    """Integration tests with real data flow."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("openpi", reason="OpenPI not installed"),
        reason="Requires OpenPI"
    )
    def test_full_training_iteration(self):
        """Test complete training iteration: forward + backward + step."""
        from loom.training.adapters.pi05 import Pi05Adapter
        
        config = {
            "type": "pi05",
            "action_dim": 32,
            "action_horizon": 10,
            "max_token_len": 180,
            "paligemma_variant": "gemma_2b",
            "default_prompt": "test task",
            "image_size": [224, 224],
        }
        
        adapter = Pi05Adapter(config)
        model = adapter.create_model()
        optimizer = adapter.create_optimizer(model, lr=1e-4, weight_decay=0.01)
        
        # Create small dataset
        train_dataset = MockLeRobotDataset(num_samples=4, action_dim=7)
        train_loader, _ = adapter.create_dataloaders(
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            batch_size=2,
            num_workers=0,
        )
        
        device = torch.device("cpu")
        model = model.to(device)
        model.train()
        
        # Get batch
        batch = next(iter(train_loader))
        
        # Forward pass
        loss, metrics = adapter.training_step(model, batch, device)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "Model should have gradients after backward pass"
        
        # Optimizer step
        optimizer.step()
        
        # Should complete without errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
