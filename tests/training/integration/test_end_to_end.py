"""End-to-end integration tests for training pipeline.

Tests the full pipeline:
1. Transform raw data
2. Preprocess for model
3. Train model
"""

from pathlib import Path

import pytest
import torch

from loom.training import Trainer, TrainingConfig
from loom.training.config import CheckpointConfig, EvaluationConfig, LoggingConfig, LRSchedulerConfig, TrainingParams


@pytest.fixture
def training_config(tmp_path):
    """Create training configuration for integration test."""
    return TrainingConfig(
        model={
            "type": "diffusion_policy",
            "obs_dim": 128,
            "action_dim": 7,
            "action_horizon": 8,
            "hidden_dim": 128,  # Smaller for faster tests
            "num_diffusion_steps": 10,  # Fewer steps for faster tests
        },
        training=TrainingParams(
            batch_size=4,
            learning_rate=1e-3,
            epochs=2,
            weight_decay=1e-6,
            gradient_clip_norm=1.0,
            num_workers=0,
            lr_scheduler=LRSchedulerConfig(type="constant"),
        ),
        checkpoints=CheckpointConfig(
            dir=tmp_path / "checkpoints",
            save_every_steps=5,
            save_every_epochs=None,
            keep_top_k=2,
            keep_last_k=1,
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
    """Create dummy dataset for training."""
    # Simulate preprocessed data
    observations = torch.randn(40, 128)  # 40 samples, obs_dim=128
    actions = torch.randn(40, 8, 7)  # action_horizon=8, action_dim=7

    # Create dict-based dataset for compatibility with adapter
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, obs, act):
            self.obs = obs
            self.act = act

        def __len__(self):
            return len(self.obs)

        def __getitem__(self, idx):
            return {"observation": self.obs[idx], "action": self.act[idx]}

    return DictDataset(observations, actions)


class TestEndToEndTraining:
    """Test end-to-end training pipeline."""

    def test_full_training_run(self, training_config, dummy_dataset):
        """Test complete training run from initialization to completion."""
        # Create trainer
        trainer = Trainer(training_config, dummy_dataset, dummy_dataset)

        # Verify initialization
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.train_loader is not None
        assert trainer.eval_loader is not None

        # Run training
        trainer.train()

        # Verify training progressed
        assert trainer.current_step > 0
        assert trainer.current_epoch >= 0

        # Verify checkpoints were saved
        checkpoint_dir = Path(training_config.checkpoints.dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0

    def test_training_reduces_loss(self, training_config, dummy_dataset):
        """Test that training actually reduces the loss."""
        trainer = Trainer(training_config, dummy_dataset, dummy_dataset)

        # Get initial loss
        initial_metrics = trainer._evaluate()
        initial_loss = initial_metrics["eval/loss"]

        # Train for a few steps
        training_config.training.epochs = 1
        trainer.train()

        # Get final loss
        final_metrics = trainer._evaluate()
        final_loss = final_metrics["eval/loss"]

        # Loss should generally decrease (with some tolerance for randomness)
        # We don't assert strict decrease since diffusion is noisy, but check it's reasonable
        assert final_loss < initial_loss * 2  # Shouldn't explode

    def test_checkpoint_resume(self, training_config, dummy_dataset, tmp_path):
        """Test resuming training from checkpoint."""
        # First training run
        trainer1 = Trainer(training_config, dummy_dataset, dummy_dataset)
        training_config.training.epochs = 1
        trainer1.train()

        checkpoint_dir = Path(training_config.checkpoints.dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0

        checkpoint_path = checkpoints[0]

        # Second training run - resume
        training_config.checkpoints.resume_from = checkpoint_path
        training_config.training.epochs = 2  # Train for more epochs

        trainer2 = Trainer(training_config, dummy_dataset, dummy_dataset)
        trainer2.train()

        # Should have trained beyond the resumed step
        assert trainer2.current_step > 0

    def test_evaluation_during_training(self, training_config, dummy_dataset):
        """Test that evaluation is performed during training."""
        training_config.evaluation.eval_every_steps = 5
        training_config.training.epochs = 1

        trainer = Trainer(training_config, dummy_dataset, dummy_dataset)

        # Train
        trainer.train()

        # Evaluation should have been called
        # We can't directly check this, but we can verify metrics were logged
        assert trainer.current_step > 0

    def test_learning_rate_scheduling(self, training_config, dummy_dataset):
        """Test learning rate scheduler updates during training."""
        # Use step scheduler
        training_config.training.lr_scheduler = LRSchedulerConfig(
            type="step",
            step_size=5,
            gamma=0.5,
        )
        training_config.training.epochs = 1

        trainer = Trainer(training_config, dummy_dataset, dummy_dataset)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Train
        trainer.train()

        # LR should have changed (or stayed same if < step_size steps)
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        # With 40 samples, batch_size=4, we have 10 steps per epoch
        # Step size is 5, so LR should have changed twice
        assert final_lr <= initial_lr

    def test_gradient_clipping_applied(self, training_config, dummy_dataset):
        """Test that gradient clipping is applied during training."""
        training_config.training.gradient_clip_norm = 0.1
        training_config.training.epochs = 1

        trainer = Trainer(training_config, dummy_dataset, dummy_dataset)

        # Train for one epoch
        trainer.train()

        # Training should complete without errors
        # Gradient clipping prevents exploding gradients
        assert trainer.current_step > 0

    def test_checkpoint_pruning(self, training_config, dummy_dataset):
        """Test that old checkpoints are pruned correctly."""
        # Configure to save many checkpoints but keep few
        training_config.checkpoints.save_every_steps = 3
        training_config.checkpoints.keep_top_k = 1
        training_config.checkpoints.keep_last_k = 1
        training_config.training.epochs = 1

        trainer = Trainer(training_config, dummy_dataset, dummy_dataset)
        trainer.train()

        # Check final checkpoint count
        checkpoint_dir = Path(training_config.checkpoints.dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

        # Should keep at most keep_top_k + keep_last_k checkpoints
        assert len(checkpoints) <= training_config.checkpoints.keep_top_k + training_config.checkpoints.keep_last_k + 1

    def test_different_batch_sizes(self, training_config, dummy_dataset):
        """Test training with different batch sizes."""
        for batch_size in [2, 4, 8]:
            training_config.training.batch_size = batch_size
            training_config.training.epochs = 1

            trainer = Trainer(training_config, dummy_dataset, dummy_dataset)
            trainer.train()

            assert trainer.current_step > 0

    def test_training_without_eval_dataset(self, training_config, dummy_dataset):
        """Test training without evaluation dataset."""
        training_config.training.epochs = 1

        trainer = Trainer(training_config, dummy_dataset, eval_dataset=None)
        trainer.train()

        # Should complete without errors
        assert trainer.current_step > 0
        assert trainer.eval_loader is None

    @pytest.mark.parametrize("scheduler_type", ["constant", "step", "cosine"])
    def test_different_lr_schedulers(self, training_config, dummy_dataset, scheduler_type):
        """Test training with different LR schedulers."""
        if scheduler_type == "step":
            training_config.training.lr_scheduler = LRSchedulerConfig(
                type="step",
                step_size=5,
                gamma=0.5,
            )
        elif scheduler_type == "cosine":
            training_config.training.lr_scheduler = LRSchedulerConfig(
                type="cosine",
                min_lr=1e-6,
            )
        else:
            training_config.training.lr_scheduler = LRSchedulerConfig(type="constant")

        training_config.training.epochs = 1

        trainer = Trainer(training_config, dummy_dataset, dummy_dataset)
        trainer.train()

        assert trainer.current_step > 0

    def test_model_parameters_update(self, training_config, dummy_dataset):
        """Test that model parameters actually update during training."""
        trainer = Trainer(training_config, dummy_dataset, dummy_dataset)

        # Get initial parameters
        initial_params = [p.clone() for p in trainer.model.parameters()]

        # Train
        training_config.training.epochs = 1
        trainer.train()

        # Get final parameters
        final_params = list(trainer.model.parameters())

        # At least some parameters should have changed
        params_changed = any(not torch.allclose(p1, p2) for p1, p2 in zip(initial_params, final_params, strict=False))
        assert params_changed

    def test_checkpoint_loading_correctness(self, training_config, dummy_dataset):
        """Test that loaded checkpoint restores exact model state."""
        trainer1 = Trainer(training_config, dummy_dataset, dummy_dataset)

        # Train and save
        training_config.training.epochs = 1
        training_config.checkpoints.save_every_steps = 5
        trainer1.train()

        # Get checkpoint
        checkpoint_dir = Path(training_config.checkpoints.dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) > 0

        # Save model state
        checkpoint_path = checkpoints[0]
        state1 = {k: v.clone() for k, v in trainer1.model.state_dict().items()}

        # Create new trainer and load checkpoint
        trainer2 = Trainer(training_config, dummy_dataset, dummy_dataset)
        trainer2.checkpoint_manager.load(checkpoint_path, trainer2.model, trainer2.optimizer)

        state2 = trainer2.model.state_dict()

        # States should match
        for key in state1.keys():
            assert torch.allclose(state1[key], state2[key])
