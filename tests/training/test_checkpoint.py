"""Tests for checkpoint management."""

import json

import pytest
import torch
import torch.nn as nn

from loom.training.checkpoint import CheckpointManager
from loom.training.config import CheckpointConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_initialization(self, tmp_path):
        """Test checkpoint manager initializes correctly."""
        config = CheckpointConfig(dir=tmp_path / "checkpoints")
        manager = CheckpointManager(config)

        assert manager.checkpoint_dir.exists()
        assert manager.checkpoint_dir == tmp_path / "checkpoints"

    def test_save_checkpoint(self, tmp_path):
        """Test saving a checkpoint."""
        config = CheckpointConfig(dir=tmp_path / "checkpoints")
        manager = CheckpointManager(config)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        metrics = {"loss": 0.5, "acc": 0.8}

        checkpoint_path = manager.save(model, optimizer, step=100, epoch=1, metrics=metrics)

        assert checkpoint_path.exists()
        assert checkpoint_path.name == "checkpoint_step_100.pt"

        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint["step"] == 100
        assert checkpoint["epoch"] == 1
        assert checkpoint["metrics"] == metrics
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint(self, tmp_path):
        """Test loading a checkpoint."""
        config = CheckpointConfig(dir=tmp_path / "checkpoints")
        manager = CheckpointManager(config)

        # Save checkpoint
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        initial_weight = model.linear.weight.clone()

        checkpoint_path = manager.save(model, optimizer, step=100, epoch=1, metrics={"loss": 0.5})

        # Modify model
        model.linear.weight.data.fill_(999.0)

        # Load checkpoint
        loaded_step = manager.load(checkpoint_path, model, optimizer)

        assert loaded_step == 100
        # Model should be restored
        assert torch.allclose(model.linear.weight, initial_weight)

    def test_load_latest(self, tmp_path):
        """Test loading the latest checkpoint."""
        config = CheckpointConfig(dir=tmp_path / "checkpoints")
        manager = CheckpointManager(config)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save multiple checkpoints
        manager.save(model, optimizer, step=100, epoch=1, metrics={"loss": 0.5})
        manager.save(model, optimizer, step=200, epoch=2, metrics={"loss": 0.3})
        manager.save(model, optimizer, step=300, epoch=3, metrics={"loss": 0.2})

        # Load latest
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        loaded_step = manager.load_latest(new_model, new_optimizer)

        assert loaded_step == 300

    def test_load_latest_no_checkpoints(self, tmp_path):
        """Test loading latest when no checkpoints exist."""
        config = CheckpointConfig(dir=tmp_path / "checkpoints")
        manager = CheckpointManager(config)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        loaded_step = manager.load_latest(model, optimizer)
        assert loaded_step == 0

    def test_checkpoint_pruning_top_k(self, tmp_path):
        """Test checkpoint pruning keeps top-K by metric."""
        config = CheckpointConfig(
            dir=tmp_path / "checkpoints",
            keep_top_k=2,
            keep_last_k=1,
            metric_for_best="loss",
            mode="min",
        )
        manager = CheckpointManager(config)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save 5 checkpoints with different losses
        manager.save(model, optimizer, step=100, epoch=1, metrics={"loss": 0.5})
        manager.save(model, optimizer, step=200, epoch=2, metrics={"loss": 0.3})
        manager.save(model, optimizer, step=300, epoch=3, metrics={"loss": 0.4})
        manager.save(model, optimizer, step=400, epoch=4, metrics={"loss": 0.2})  # Best
        manager.save(model, optimizer, step=500, epoch=5, metrics={"loss": 0.25})  # 2nd best

        # Should keep: step 200 (2nd best), step 400 (best), step 500 (last)
        existing_checkpoints = list(manager.checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(existing_checkpoints) == 3

        checkpoint_names = {p.name for p in existing_checkpoints}
        assert "checkpoint_step_200.pt" in checkpoint_names  # 2nd best loss
        assert "checkpoint_step_400.pt" in checkpoint_names  # Best loss
        assert "checkpoint_step_500.pt" in checkpoint_names  # Last

    def test_checkpoint_index_persistence(self, tmp_path):
        """Test checkpoint index is saved and loaded correctly."""
        config = CheckpointConfig(dir=tmp_path / "checkpoints")
        manager = CheckpointManager(config)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoints
        manager.save(model, optimizer, step=100, epoch=1, metrics={"loss": 0.5})
        manager.save(model, optimizer, step=200, epoch=2, metrics={"loss": 0.3})

        # Check index file exists
        index_path = manager.checkpoint_dir / "checkpoint_index.json"
        assert index_path.exists()

        # Load index and verify
        with open(index_path) as f:
            index_data = json.load(f)

        assert len(index_data) == 2
        assert index_data[0]["step"] == 100
        assert index_data[1]["step"] == 200

        # Create new manager (should load index)
        new_manager = CheckpointManager(config)
        assert len(new_manager._checkpoints) == 2

    def test_load_nonexistent_checkpoint(self, tmp_path):
        """Test loading non-existent checkpoint raises error."""
        config = CheckpointConfig(dir=tmp_path / "checkpoints")
        manager = CheckpointManager(config)

        model = SimpleModel()
        fake_path = tmp_path / "fake_checkpoint.pt"

        with pytest.raises(FileNotFoundError):
            manager.load(fake_path, model)

    def test_checkpoint_without_optimizer(self, tmp_path):
        """Test loading checkpoint without optimizer state."""
        config = CheckpointConfig(dir=tmp_path / "checkpoints")
        manager = CheckpointManager(config)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = manager.save(model, optimizer, step=100, epoch=1, metrics={"loss": 0.5})

        # Load without optimizer
        new_model = SimpleModel()
        loaded_step = manager.load(checkpoint_path, new_model, optimizer=None)

        assert loaded_step == 100
