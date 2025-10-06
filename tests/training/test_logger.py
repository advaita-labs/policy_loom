"""Tests for training logger."""

from unittest.mock import MagicMock, patch

import pytest

from loom.training.config import LoggingConfig, WandbConfig
from loom.training.logger import WANDB_AVAILABLE, TrainingLogger


class TestTrainingLogger:
    """Test TrainingLogger class."""

    def test_initialization_without_wandb(self, tmp_path):
        """Test logger initializes without WandB."""
        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            save_logs=True,
            wandb=WandbConfig(enabled=False),
        )

        logger = TrainingLogger(config)
        assert logger._wandb_run is None

    def test_file_logging_enabled(self, tmp_path):
        """Test file logging creates log file."""
        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            save_logs=True,
            wandb=WandbConfig(enabled=False),
        )

        TrainingLogger(config)

        log_file = tmp_path / "logs" / "training.log"
        assert log_file.exists()

    def test_log_metrics(self, tmp_path):
        """Test logging metrics to console."""
        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            save_logs=False,
            wandb=WandbConfig(enabled=False),
        )

        logger = TrainingLogger(config)
        # Should not raise error
        logger.log({"loss": 0.5, "accuracy": 0.85}, step=100)
        logger.log({"loss": 0.3}, step=200)

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="wandb not installed")
    @patch("loom.training.logger.wandb")
    def test_wandb_initialization(self, mock_wandb, tmp_path):
        """Test WandB initialization when enabled."""
        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            wandb=WandbConfig(
                enabled=True,
                project="test_project",
                entity="test_entity",
                name="test_run",
                tags=["test"],
            ),
        )

        mock_wandb_run = MagicMock()
        mock_wandb_run.url = "https://wandb.ai/test"
        mock_wandb.init.return_value = mock_wandb_run

        TrainingLogger(config)

        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "test_project"
        assert call_kwargs["entity"] == "test_entity"
        assert call_kwargs["name"] == "test_run"
        assert call_kwargs["tags"] == ["test"]

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="wandb not installed")
    @patch("loom.training.logger.wandb")
    def test_log_to_wandb(self, mock_wandb, tmp_path):
        """Test logging metrics to WandB."""
        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            wandb=WandbConfig(enabled=True, project="test"),
        )

        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run

        logger = TrainingLogger(config)
        logger.log({"loss": 0.5}, step=100)

        mock_wandb_run.log.assert_called_once_with({"loss": 0.5}, step=100)

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="wandb not installed")
    @patch("loom.training.logger.wandb")
    def test_log_config(self, mock_wandb, tmp_path):
        """Test logging configuration."""
        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            wandb=WandbConfig(enabled=True, project="test"),
        )

        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run

        logger = TrainingLogger(config)
        test_config = {"epochs": 100, "batch_size": 32}
        logger.log_config(test_config)

        mock_wandb_run.config.update.assert_called_once_with(test_config)

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="wandb not installed")
    @patch("loom.training.logger.wandb")
    def test_finish_closes_wandb(self, mock_wandb, tmp_path):
        """Test finish() closes WandB run."""
        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            wandb=WandbConfig(enabled=True, project="test"),
        )

        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run

        logger = TrainingLogger(config)
        logger.finish()

        mock_wandb_run.finish.assert_called_once()
        assert logger._wandb_run is None

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="wandb not installed")
    @patch("loom.training.logger.wandb")
    def test_context_manager(self, mock_wandb, tmp_path):
        """Test logger works as context manager."""
        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            wandb=WandbConfig(enabled=True, project="test"),
        )

        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run

        with TrainingLogger(config) as logger:
            assert logger._wandb_run is not None

        mock_wandb_run.finish.assert_called_once()

    def test_wandb_not_installed_raises_error(self, tmp_path):
        """Test enabling WandB without installation raises error."""
        if WANDB_AVAILABLE:
            pytest.skip("wandb is installed")

        config = LoggingConfig(
            log_dir=tmp_path / "logs",
            wandb=WandbConfig(enabled=True, project="test"),
        )

        with pytest.raises(ImportError, match="WandB is enabled but not installed"):
            TrainingLogger(config)
