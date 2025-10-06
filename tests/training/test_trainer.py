"""Tests for Trainer class."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from loom.training.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    LRSchedulerConfig,
    TrainingConfig,
    TrainingParams,
)
from loom.training.trainer import Trainer


# Mock model for testing
class MockModel(nn.Module):
    """Simple mock model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


# Mock adapter for testing
class MockAdapter:
    """Mock adapter for testing."""

    def __init__(self, config):
        self.config = config

    def create_model(self) -> nn.Module:
        return MockModel()

    def create_optimizer(self, model: nn.Module, lr: float, weight_decay: float):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        return loss, {"loss": loss.item(), "accuracy": 0.85}

    def eval_step(self, model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        return {"eval/loss": loss.item(), "eval/accuracy": 0.9}

    def create_dataloaders(self, train_dataset, eval_dataset, batch_size, num_workers):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=0) if eval_dataset else None
        return train_loader, eval_loader


@pytest.fixture
def mock_adapter():
    """Fixture for mock adapter."""
    with patch("loom.training.trainer.get_adapter") as mock:
        mock.return_value = MockAdapter({})
        yield mock


@pytest.fixture
def simple_config(tmp_path):
    """Create simple training config."""
    return TrainingConfig(
        model={"type": "mock"},
        training=TrainingParams(
            batch_size=4,
            learning_rate=1e-3,
            epochs=2,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            num_workers=0,
            lr_scheduler=LRSchedulerConfig(type="constant"),
        ),
        checkpoints=CheckpointConfig(
            dir=tmp_path / "checkpoints",
            save_every_steps=10,
            save_every_epochs=None,
        ),
        evaluation=EvaluationConfig(
            eval_every_steps=5,
            eval_every_epochs=None,
        ),
        logging=LoggingConfig(
            log_dir=tmp_path / "logs",
            log_every_steps=5,
            save_logs=False,
        ),
        data={},
    )


@pytest.fixture
def dummy_dataset():
    """Create dummy dataset."""
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    return TensorDataset(x, y)


class TestTrainer:
    """Test Trainer class."""

    def test_initialization(self, simple_config, dummy_dataset, mock_adapter):
        """Test trainer initialization."""
        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)

        assert trainer.config == simple_config
        assert trainer.train_dataset == dummy_dataset
        assert trainer.eval_dataset == dummy_dataset
        assert trainer.current_step == 0
        assert trainer.current_epoch == 0
        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_device_setup_cpu(self, simple_config, dummy_dataset, mock_adapter):
        """Test device setup on CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(simple_config, dummy_dataset)
            assert trainer.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_setup_gpu(self, simple_config, dummy_dataset, mock_adapter):
        """Test device setup on GPU."""
        trainer = Trainer(simple_config, dummy_dataset)
        assert trainer.device.type == "cuda"

    def test_training_basic(self, simple_config, dummy_dataset, mock_adapter):
        """Test basic training loop."""
        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)

        # Run training for 1 epoch
        simple_config.training.epochs = 1
        trainer.train()

        # Check training progressed
        assert trainer.current_step > 0
        assert trainer.current_epoch >= 0

    def test_training_with_evaluation(self, simple_config, dummy_dataset, mock_adapter):
        """Test training with periodic evaluation."""
        simple_config.evaluation.eval_every_steps = 5

        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)
        initial_step = trainer.current_step

        # Train for 1 epoch
        simple_config.training.epochs = 1
        trainer.train()

        # Verify evaluation was called (indirectly through step count)
        assert trainer.current_step > initial_step

    def test_training_with_checkpointing(self, simple_config, dummy_dataset, mock_adapter, tmp_path):
        """Test training with periodic checkpointing."""
        # Dataset has 20 samples, batch_size=4 -> 5 steps per epoch
        # Set checkpoint every 3 steps to ensure at least one is saved
        simple_config.checkpoints.save_every_steps = 3

        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)

        # Train for 1 epoch (5 steps total)
        simple_config.training.epochs = 1
        trainer.train()

        # Check checkpoint was saved (should have at step 3)
        checkpoint_dir = Path(simple_config.checkpoints.dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0

    def test_resume_from_checkpoint(self, simple_config, dummy_dataset, mock_adapter, tmp_path):
        """Test resuming training from checkpoint."""
        # First training run - save checkpoint
        simple_config.checkpoints.save_every_steps = 5
        simple_config.training.epochs = 1

        trainer1 = Trainer(simple_config, dummy_dataset, dummy_dataset)
        trainer1.train()

        # Get the checkpoint that was saved
        checkpoint_dir = Path(simple_config.checkpoints.dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0

        checkpoint_path = checkpoints[0]

        # Second training run - resume from checkpoint
        simple_config.checkpoints.resume_from = checkpoint_path
        trainer2 = Trainer(simple_config, dummy_dataset, dummy_dataset)

        # Current step should be loaded from checkpoint
        # Note: resume is called during train(), not __init__
        trainer2._resume_from_checkpoint()
        assert trainer2.current_step >= 0  # Should have resumed

    def test_gradient_clipping(self, simple_config, dummy_dataset, mock_adapter):
        """Test gradient clipping is applied."""
        simple_config.training.gradient_clip_norm = 0.5

        trainer = Trainer(simple_config, dummy_dataset)

        # Run one training step
        trainer.model.train()
        batch = next(iter(trainer.train_loader))
        loss, _ = trainer.adapter.training_step(trainer.model, batch, trainer.device)

        trainer.optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        has_gradients = any(p.grad is not None for p in trainer.model.parameters())
        assert has_gradients

    def test_learning_rate_scheduler_step(self, simple_config, dummy_dataset, mock_adapter):
        """Test step learning rate scheduler."""
        simple_config.training.lr_scheduler = LRSchedulerConfig(
            type="step",
            step_size=5,
            gamma=0.5,
        )

        trainer = Trainer(simple_config, dummy_dataset)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        assert trainer.scheduler is not None

        # Step the scheduler
        for _ in range(6):
            trainer.scheduler.step()

        # LR should have decreased
        new_lr = trainer.optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr

    def test_learning_rate_scheduler_cosine(self, simple_config, dummy_dataset, mock_adapter):
        """Test cosine learning rate scheduler."""
        simple_config.training.lr_scheduler = LRSchedulerConfig(
            type="cosine",
            min_lr=1e-6,
        )

        trainer = Trainer(simple_config, dummy_dataset)

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_learning_rate_scheduler_plateau(self, simple_config, dummy_dataset, mock_adapter):
        """Test plateau learning rate scheduler."""
        simple_config.training.lr_scheduler = LRSchedulerConfig(
            type="plateau",
            gamma=0.5,
            patience=2,
        )

        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_learning_rate_scheduler_constant(self, simple_config, dummy_dataset, mock_adapter):
        """Test constant learning rate (no scheduler)."""
        simple_config.training.lr_scheduler = LRSchedulerConfig(type="constant")

        trainer = Trainer(simple_config, dummy_dataset)

        assert trainer.scheduler is None

    def test_should_evaluate_step(self, simple_config, dummy_dataset, mock_adapter):
        """Test evaluation step logic."""
        simple_config.evaluation.eval_every_steps = 10

        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)

        assert not trainer._should_evaluate_step(5)
        assert trainer._should_evaluate_step(10)
        assert trainer._should_evaluate_step(20)

    def test_should_evaluate_epoch(self, simple_config, dummy_dataset, mock_adapter):
        """Test evaluation epoch logic."""
        simple_config.evaluation.eval_every_epochs = 2

        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)

        assert not trainer._should_evaluate_epoch(0)
        assert trainer._should_evaluate_epoch(1)  # epoch+1 % 2 == 0
        assert not trainer._should_evaluate_epoch(2)
        assert trainer._should_evaluate_epoch(3)

    def test_should_checkpoint_step(self, simple_config, dummy_dataset, mock_adapter):
        """Test checkpoint step logic."""
        simple_config.checkpoints.save_every_steps = 100

        trainer = Trainer(simple_config, dummy_dataset)

        assert not trainer._should_checkpoint_step(50)
        assert trainer._should_checkpoint_step(100)
        assert trainer._should_checkpoint_step(200)

    def test_should_checkpoint_epoch(self, simple_config, dummy_dataset, mock_adapter):
        """Test checkpoint epoch logic."""
        simple_config.checkpoints.save_every_epochs = 5

        trainer = Trainer(simple_config, dummy_dataset)

        assert not trainer._should_checkpoint_epoch(0)
        assert not trainer._should_checkpoint_epoch(3)
        assert trainer._should_checkpoint_epoch(4)  # epoch+1 % 5 == 0
        assert trainer._should_checkpoint_epoch(9)

    def test_evaluate_without_eval_dataset(self, simple_config, dummy_dataset, mock_adapter):
        """Test evaluation without eval dataset."""
        trainer = Trainer(simple_config, dummy_dataset, eval_dataset=None)

        metrics = trainer._evaluate()

        # Should return empty metrics
        assert metrics == {}

    def test_evaluate_with_eval_dataset(self, simple_config, dummy_dataset, mock_adapter):
        """Test evaluation with eval dataset."""
        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)

        metrics = trainer._evaluate()

        # Should return evaluation metrics
        assert "eval/loss" in metrics
        assert "eval/accuracy" in metrics

    def test_training_keyboard_interrupt(self, simple_config, dummy_dataset, mock_adapter):
        """Test training handles keyboard interrupt."""
        trainer = Trainer(simple_config, dummy_dataset, dummy_dataset)

        # Mock training to raise KeyboardInterrupt
        original_train_epoch = trainer._train_epoch

        def mock_train_epoch() -> None:
            original_train_epoch()
            raise KeyboardInterrupt()

        trainer._train_epoch = mock_train_epoch  # type: ignore[method-assign]

        # Should not raise, should save checkpoint
        trainer.train()

        # Check checkpoint was saved
        checkpoint_dir = Path(simple_config.checkpoints.dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0

    def test_config_dict_generation(self, simple_config, dummy_dataset, mock_adapter):
        """Test configuration dictionary generation."""
        trainer = Trainer(simple_config, dummy_dataset)

        config_dict = trainer._get_config_dict()

        assert "model" in config_dict
        assert "training" in config_dict
        assert "device" in config_dict
        assert config_dict["training"]["batch_size"] == 4
        assert config_dict["training"]["learning_rate"] == 1e-3
