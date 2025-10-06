"""Logging for training with console and WandB support."""

import logging
from pathlib import Path
from typing import Any

from loom.training.config import LoggingConfig

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Logger for training with console and optional WandB support.

    Handles:
    - Console logging
    - File logging
    - WandB logging (if enabled and available)

    Example:
        >>> logger = TrainingLogger(config)
        >>> logger.log({"loss": 0.5, "lr": 1e-4}, step=100)
        >>> logger.finish()
    """

    def __init__(self, config: LoggingConfig, project_name: str | None = None):
        """Initialize logger.

        Args:
            config: Logging configuration
            project_name: Optional project name override

        Raises:
            ImportError: If wandb is enabled but not installed
        """
        self.config = config
        self._wandb_run = None

        # Setup console logging
        self._setup_console_logging()

        # Setup file logging
        if config.save_logs:
            self._setup_file_logging()

        # Setup wandb
        if config.wandb.enabled:
            self._setup_wandb(project_name)

    def _setup_console_logging(self) -> None:
        """Setup console logging."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True
        )

    def _setup_file_logging(self) -> None:
        """Setup file logging."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    def _setup_wandb(self, project_name: str | None = None) -> None:
        """Setup Weights & Biases logging.

        Args:
            project_name: Optional project name override

        Raises:
            ImportError: If wandb is not installed
        """
        if not WANDB_AVAILABLE:
            raise ImportError("WandB is enabled but not installed. " "Install with: pip install wandb")

        wandb_config = self.config.wandb
        project = project_name or wandb_config.project

        self._wandb_run = wandb.init(
            project=project,
            entity=wandb_config.entity,
            name=wandb_config.name,
            tags=wandb_config.tags,
            notes=wandb_config.notes,
            group=wandb_config.group,
            reinit=True,
        )

        if self._wandb_run is not None:
            logger.info(f"Initialized WandB: {self._wandb_run.url}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        # Log to console
        metrics_str = " ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"Step {step}: {metrics_str}")

        # Log to wandb
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration.

        Args:
            config: Configuration dictionary
        """
        logger.info(f"Config: {config}")

        if self._wandb_run is not None:
            self._wandb_run.config.update(config)

    def finish(self) -> None:
        """Finish logging and cleanup."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
            logger.info("Finished WandB logging")

    def __enter__(self) -> "TrainingLogger":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.finish()
