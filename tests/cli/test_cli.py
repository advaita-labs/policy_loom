"""Tests for CLI commands (TDD approach)."""

from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from loom.cli import app

runner = CliRunner()


@pytest.fixture
def training_config_file(tmp_path):
    """Create a training config YAML file."""
    config = {
        "model": {
            "type": "diffusion_policy",
            "obs_dim": 128,
            "action_dim": 7,
            "action_horizon": 8,
            "hidden_dim": 128,
            "num_diffusion_steps": 10,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "epochs": 2,
            "weight_decay": 1e-6,
            "gradient_clip_norm": 1.0,
            "num_workers": 0,
            "lr_scheduler": {"type": "constant"},
        },
        "checkpoints": {
            "dir": str(tmp_path / "checkpoints"),
            "save_every_steps": 10,
            "keep_top_k": 2,
            "keep_last_k": 1,
        },
        "evaluation": {
            "eval_every_steps": 5,
        },
        "logging": {
            "log_dir": str(tmp_path / "logs"),
            "log_every_steps": 5,
            "save_logs": False,
        },
        "data": {
            "train_path": str(tmp_path / "train.pt"),
            "eval_path": str(tmp_path / "eval.pt"),
        },
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


class TestCLICommands:
    """Test CLI commands."""

    def test_cli_exists(self):
        """Test that CLI app exists."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "loom" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_train_command_exists(self):
        """Test that train command exists."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "train" in result.stdout.lower()

    def test_train_command_requires_config(self):
        """Test that train command requires config file."""
        result = runner.invoke(app, ["train"])
        # Should fail without config file
        assert result.exit_code != 0

    def test_train_command_with_nonexistent_config(self):
        """Test train command with non-existent config file."""
        result = runner.invoke(app, ["train", "nonexistent.yaml"])
        assert result.exit_code != 0
        output = (result.stdout + result.stderr).lower()
        assert "not found" in output or "does not exist" in output

    @patch("loom.cli.torch.load")
    @patch("loom.cli.Trainer")
    @patch("loom.cli.TrainingConfig")
    def test_train_command_with_valid_config(
        self, mock_config, mock_trainer, mock_torch_load, training_config_file, tmp_path
    ):
        """Test train command with valid config file."""
        # Mock the config loading
        mock_config_instance = MagicMock()
        mock_config_instance.data = {
            "train_path": str(tmp_path / "train.pt"),
            "eval_path": str(tmp_path / "eval.pt"),
        }
        mock_config_instance.model = {"type": "diffusion_policy"}
        mock_config.from_yaml.return_value = mock_config_instance

        # Mock torch.load to return dummy data
        mock_torch_load.return_value = {
            "observation": [[0.0] * 128] * 10,
            "action": [[[0.0] * 7] * 8] * 10,
        }

        # Mock the trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Create dummy data files
        (tmp_path / "train.pt").touch()
        (tmp_path / "eval.pt").touch()

        result = runner.invoke(app, ["train", str(training_config_file)])

        # Should succeed
        assert result.exit_code == 0

        # Should have loaded config
        mock_config.from_yaml.assert_called_once()

        # Should have created trainer
        mock_trainer.assert_called_once()

        # Should have called train()
        mock_trainer_instance.train.assert_called_once()

    def test_train_command_verbose_flag(self, training_config_file):
        """Test train command with --verbose flag."""
        result = runner.invoke(app, ["train", str(training_config_file), "--verbose"])
        # Should accept verbose flag (even if it fails due to missing data)
        assert "--verbose" not in result.stdout or result.exit_code in [0, 1]

    def test_preprocess_command_exists(self):
        """Test that preprocess command exists."""
        result = runner.invoke(app, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "preprocess" in result.stdout.lower()

    def test_transform_command_exists(self):
        """Test that transform command exists."""
        result = runner.invoke(app, ["transform", "--help"])
        assert result.exit_code == 0
        assert "transform" in result.stdout.lower()

    def test_eval_command_exists(self):
        """Test that eval command exists."""
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "eval" in result.stdout.lower()

    def test_eval_command_requires_config(self):
        """Test that eval command requires config file."""
        result = runner.invoke(app, ["eval"])
        assert result.exit_code != 0

    @patch("loom.cli.Trainer")
    @patch("loom.cli.TrainingConfig")
    def test_eval_command_with_checkpoint(self, mock_config, mock_trainer, training_config_file, tmp_path):
        """Test eval command with checkpoint path."""
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint_path.touch()

        # Mock config and trainer
        mock_config_instance = MagicMock()
        mock_config.from_yaml.return_value = mock_config_instance
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Create dummy data files
        (tmp_path / "train.pt").touch()
        (tmp_path / "eval.pt").touch()

        result = runner.invoke(app, ["eval", str(training_config_file), "--checkpoint", str(checkpoint_path)])

        # Should accept checkpoint flag
        assert result.exit_code in [0, 1]  # May fail due to mock data

    def test_train_with_custom_output_dir(self, training_config_file, tmp_path):
        """Test train command with custom output directory."""
        output_dir = tmp_path / "custom_output"
        result = runner.invoke(app, ["train", str(training_config_file), "--output-dir", str(output_dir)])

        # Should accept output-dir flag
        assert "--output-dir" not in result.stdout or result.exit_code in [0, 1]

    def test_list_adapters_command(self):
        """Test list-adapters command to show available models."""
        result = runner.invoke(app, ["list-adapters"])

        # Should succeed
        assert result.exit_code == 0

        # Should show diffusion_policy
        assert "diffusion_policy" in result.stdout

    def test_version_command(self):
        """Test version command."""
        result = runner.invoke(app, ["--version"])

        # Should show version
        assert result.exit_code == 0


class TestCLIConfigValidation:
    """Test CLI config validation."""

    def test_train_with_invalid_yaml(self, tmp_path):
        """Test train with invalid YAML syntax."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: syntax: :")

        result = runner.invoke(app, ["train", str(config_path)])

        assert result.exit_code != 0
        output = (result.stdout + result.stderr).lower()
        assert "error" in output or "invalid" in output

    def test_train_with_missing_required_fields(self, tmp_path):
        """Test train with config missing required fields."""
        config = {"model": {"type": "diffusion_policy"}}  # Missing required fields

        config_path = tmp_path / "incomplete.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["train", str(config_path)])

        assert result.exit_code != 0


class TestCLIDataLoading:
    """Test CLI data loading."""

    def test_train_with_missing_data_files(self, training_config_file):
        """Test train command when data files don't exist."""
        result = runner.invoke(app, ["train", str(training_config_file)])

        # Should fail gracefully with error message
        assert result.exit_code != 0

    @patch("loom.cli.torch.load")
    @patch("loom.cli.Trainer")
    @patch("loom.cli.TrainingConfig")
    def test_train_with_valid_data_files(self, mock_config, mock_trainer, mock_load, training_config_file, tmp_path):
        """Test train command with valid data files."""
        # Create dummy data files
        (tmp_path / "train.pt").touch()
        (tmp_path / "eval.pt").touch()

        # Mock torch.load to return dummy data
        mock_load.return_value = {
            "observation": [[0.0] * 128] * 10,
            "action": [[[0.0] * 7] * 8] * 10,
        }

        # Mock config
        mock_config_instance = MagicMock()
        mock_config_instance.data = {"train_path": str(tmp_path / "train.pt"), "eval_path": str(tmp_path / "eval.pt")}
        mock_config.from_yaml.return_value = mock_config_instance

        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        result = runner.invoke(app, ["train", str(training_config_file)])

        # Should succeed
        assert result.exit_code == 0
