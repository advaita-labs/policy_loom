"""Tests for training configuration."""

from pathlib import Path

import pytest

from loom.training.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    LRSchedulerConfig,
    TrainingConfig,
    TrainingParams,
    WandbConfig,
)


class TestLRSchedulerConfig:
    """Test LRSchedulerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LRSchedulerConfig()
        assert config.type == "cosine"
        assert config.warmup_steps == 1000
        assert config.min_lr == 1e-6

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LRSchedulerConfig(type="step", warmup_steps=500, step_size=1000, gamma=0.5)
        assert config.type == "step"
        assert config.warmup_steps == 500
        assert config.step_size == 1000
        assert config.gamma == 0.5


class TestTrainingParams:
    """Test TrainingParams dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingParams()
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.optimizer == "adamw"

    def test_nested_lr_scheduler(self):
        """Test nested LRSchedulerConfig."""
        lr_config = LRSchedulerConfig(type="constant")
        config = TrainingParams(lr_scheduler=lr_config)
        assert config.lr_scheduler.type == "constant"


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CheckpointConfig()
        assert config.dir == Path("./checkpoints")
        assert config.save_every_steps == 1000
        assert config.keep_top_k == 3
        assert config.mode == "min"

    def test_path_handling(self):
        """Test Path object handling."""
        config = CheckpointConfig(dir=Path("/tmp/ckpts"))
        assert isinstance(config.dir, Path)
        assert config.dir == Path("/tmp/ckpts")


class TestWandbConfig:
    """Test WandbConfig dataclass."""

    def test_default_disabled(self):
        """Test WandB is disabled by default."""
        config = WandbConfig()
        assert config.enabled is False
        assert config.project == "policy_loom"

    def test_enabled_with_settings(self):
        """Test WandB with custom settings."""
        config = WandbConfig(enabled=True, project="my_project", entity="my_team", tags=["exp1", "test"])
        assert config.enabled is True
        assert config.project == "my_project"
        assert config.entity == "my_team"
        assert config.tags == ["exp1", "test"]


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_from_yaml_valid_config(self, tmp_path):
        """Test loading valid YAML config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
model:
  type: diffusion_policy
  obs_horizon: 2

training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  lr_scheduler:
    type: cosine
    warmup_steps: 500

checkpoints:
  dir: ./test_checkpoints
  save_every_steps: 100

evaluation:
  eval_every_steps: 50

logging:
  log_every_steps: 5
  wandb:
    enabled: false

data:
  train_path: ./data/train
"""
        config_file.write_text(config_content)

        config = TrainingConfig.from_yaml(config_file)

        assert config.model["type"] == "diffusion_policy"
        assert config.model["obs_horizon"] == 2
        assert config.training.epochs == 50
        assert config.training.batch_size == 16
        assert config.training.lr_scheduler.warmup_steps == 500
        assert config.checkpoints.save_every_steps == 100
        assert config.evaluation.eval_every_steps == 50
        assert config.logging.wandb.enabled is False
        assert config.data["train_path"] == "./data/train"

    def test_from_yaml_missing_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_yaml(Path("/nonexistent/config.yaml"))

    def test_to_yaml_roundtrip(self, tmp_path):
        """Test saving and loading config maintains data."""
        config = TrainingConfig(
            model={"type": "test_model", "param": 42},
            training=TrainingParams(epochs=10, batch_size=8),
            checkpoints=CheckpointConfig(dir=Path("./ckpts")),
            evaluation=EvaluationConfig(eval_every_steps=100),
            logging=LoggingConfig(wandb=WandbConfig(enabled=False)),
            data={"train_path": "./data"},
        )

        config_file = tmp_path / "saved_config.yaml"
        config.to_yaml(config_file)

        loaded_config = TrainingConfig.from_yaml(config_file)

        assert loaded_config.model == config.model
        assert loaded_config.training.epochs == config.training.epochs
        assert loaded_config.training.batch_size == config.training.batch_size
        assert loaded_config.checkpoints.dir == config.checkpoints.dir
        assert loaded_config.evaluation.eval_every_steps == config.evaluation.eval_every_steps

    def test_from_yaml_minimal_config(self, tmp_path):
        """Test loading minimal config with defaults."""
        config_file = tmp_path / "minimal.yaml"
        config_content = """
model:
  type: simple_model

training: {}
checkpoints: {}
evaluation: {}
logging: {}
data: {}
"""
        config_file.write_text(config_content)

        config = TrainingConfig.from_yaml(config_file)

        # Should use defaults
        assert config.training.epochs == 100  # default
        assert config.checkpoints.keep_top_k == 3  # default
        assert config.logging.log_every_steps == 10  # default
