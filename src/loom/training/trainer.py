"""Generic trainer for VLA models."""

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from loom.training.adapter import ModelAdapter, get_adapter
from loom.training.checkpoint import CheckpointManager
from loom.training.config import TrainingConfig
from loom.training.logger import TrainingLogger
from loom.training.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class Trainer:
    """Generic trainer for VLA models.

    Orchestrates training loop using model-specific adapters.
    Handles checkpointing, logging, evaluation, and learning rate scheduling.

    Example:
        >>> config = TrainingConfig.from_yaml("config.yaml")
        >>> trainer = Trainer(config, train_dataset, eval_dataset)
        >>> trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Setup device
        self.device = self._setup_device()

        # Get model adapter
        model_config = config.model if isinstance(config.model, dict) else config.model.__dict__
        self.adapter: ModelAdapter = get_adapter(
            model_config["type"],
            model_config,
        )

        # Create model and optimizer
        self.model = self.adapter.create_model().to(self.device)
        self.optimizer = self.adapter.create_optimizer(
            self.model,
            config.training.learning_rate,
            config.training.weight_decay,
        )

        # Create dataloaders
        self.train_loader, self.eval_loader = self.adapter.create_dataloaders(
            train_dataset,
            eval_dataset,
            config.training.batch_size,
            config.training.num_workers,
        )

        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Setup infrastructure
        self.checkpoint_manager = CheckpointManager(config.checkpoints)
        self.logger = TrainingLogger(config.logging)
        self.train_metrics = MetricsTracker()
        self.eval_metrics = MetricsTracker()

        # Training state
        self.current_step = 0
        self.current_epoch = 0

        # Log configuration
        self.logger.log_config(self._get_config_dict())

    def train(self) -> None:
        """Run training loop.

        Handles:
        - Training for specified epochs/steps
        - Periodic evaluation
        - Periodic checkpointing
        - Learning rate scheduling
        - Gradient clipping
        - Resuming from checkpoints
        """
        # Resume from checkpoint if specified
        if self.config.checkpoints.resume_from:
            self._resume_from_checkpoint()

        logger.info(f"Starting training from step {self.current_step}")
        logger.info(f"Training for {self.config.training.epochs} epochs")
        logger.info(f"Device: {self.device}")

        try:
            for epoch in range(self.current_epoch, self.config.training.epochs):
                self.current_epoch = epoch
                self._train_epoch()

                # Epoch-based evaluation
                if self._should_evaluate_epoch(epoch):
                    self._evaluate()

                # Epoch-based checkpointing
                if self._should_checkpoint_epoch(epoch):
                    self._save_checkpoint()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.logger.finish()

        logger.info("Training completed!")

    def _train_epoch(self) -> None:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        for batch in self.train_loader:
            # Training step
            loss, metrics = self.adapter.training_step(self.model, batch, self.device)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )

            self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None and self.config.training.lr_scheduler.type in ["step", "cosine"]:
                self.scheduler.step()

            # Track metrics
            self.train_metrics.update(metrics)

            self.current_step += 1

            # Step-based logging
            if self.current_step % self.config.logging.log_every_steps == 0:
                avg_metrics = self.train_metrics.compute()
                avg_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                avg_metrics["epoch"] = self.current_epoch
                self.logger.log(avg_metrics, step=self.current_step)
                self.train_metrics.reset()

            # Step-based evaluation
            if self._should_evaluate_step(self.current_step):
                self._evaluate()

            # Step-based checkpointing
            if self._should_checkpoint_step(self.current_step):
                self._save_checkpoint()

        # Update scheduler at epoch end if needed
        if self.scheduler is not None and self.config.training.lr_scheduler.type == "plateau":
            # For plateau scheduler, need eval loss
            if self.eval_loader is not None:
                eval_metrics = self._evaluate()
                metric_value = eval_metrics.get("eval/loss", 0.0)
                self.scheduler.step(metric_value)

    def _evaluate(self) -> dict[str, float]:
        """Run evaluation.

        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_loader is None:
            logger.warning("No evaluation dataset provided, skipping evaluation")
            return {}

        logger.info(f"Evaluating at step {self.current_step}")
        self.model.eval()
        self.eval_metrics.reset()

        with torch.no_grad():
            for batch in self.eval_loader:
                metrics = self.adapter.eval_step(self.model, batch, self.device)
                self.eval_metrics.update(metrics)

        # Compute and log metrics
        avg_metrics = self.eval_metrics.compute()
        self.logger.log(avg_metrics, step=self.current_step)

        self.model.train()
        return avg_metrics

    def _save_checkpoint(self) -> None:
        """Save checkpoint."""
        # Get current metrics
        eval_metrics = self._evaluate() if self.eval_loader is not None else {}

        checkpoint_path = self.checkpoint_manager.save(
            self.model,
            self.optimizer,
            step=self.current_step,
            epoch=self.current_epoch,
            metrics=eval_metrics,
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint."""
        if self.config.checkpoints.resume_from is None:
            return

        resume_path = Path(self.config.checkpoints.resume_from)

        if not resume_path.exists():
            logger.warning(f"Checkpoint not found: {resume_path}, starting from scratch")
            return

        self.current_step = self.checkpoint_manager.load(
            resume_path,
            self.model,
            self.optimizer,
        )

        # Estimate epoch from step
        steps_per_epoch = len(self.train_loader)
        self.current_epoch = self.current_step // steps_per_epoch

        logger.info(f"Resumed from checkpoint: step={self.current_step}, epoch={self.current_epoch}")

    def _should_evaluate_step(self, step: int) -> bool:
        """Check if should evaluate at this step."""
        if self.config.evaluation.eval_every_steps is None:
            return False
        return step % self.config.evaluation.eval_every_steps == 0

    def _should_evaluate_epoch(self, epoch: int) -> bool:
        """Check if should evaluate at this epoch."""
        if self.config.evaluation.eval_every_epochs is None:
            return False
        return (epoch + 1) % self.config.evaluation.eval_every_epochs == 0

    def _should_checkpoint_step(self, step: int) -> bool:
        """Check if should save checkpoint at this step."""
        if self.config.checkpoints.save_every_steps is None:
            return False
        return step % self.config.checkpoints.save_every_steps == 0

    def _should_checkpoint_epoch(self, epoch: int) -> bool:
        """Check if should save checkpoint at this epoch."""
        if self.config.checkpoints.save_every_epochs is None:
            return False
        return (epoch + 1) % self.config.checkpoints.save_every_epochs == 0

    def _setup_device(self) -> torch.device:
        """Setup training device.

        Returns:
            Device for training (cuda or cpu)
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")

        return device

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Create learning rate scheduler.

        Returns:
            LR scheduler or None if not configured
        """
        lr_config = self.config.training.lr_scheduler

        if lr_config.type == "constant":
            return None

        if lr_config.type == "step":
            if lr_config.step_size is None:
                raise ValueError("step_size must be specified for step scheduler")
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=lr_config.step_size,
                gamma=lr_config.gamma,
            )

        if lr_config.type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs * len(self.train_loader),
                eta_min=lr_config.min_lr,
            )

        if lr_config.type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=lr_config.gamma,
                patience=lr_config.patience,
            )

        raise ValueError(f"Unknown scheduler type: {lr_config.type}")

    def _get_config_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary for logging.

        Returns:
            Configuration dictionary
        """
        return {
            "model": self.config.model if isinstance(self.config.model, dict) else self.config.model.__dict__,
            "training": {
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "epochs": self.config.training.epochs,
                "weight_decay": self.config.training.weight_decay,
                "gradient_clip_norm": self.config.training.gradient_clip_norm,
            },
            "device": str(self.device),
        }
