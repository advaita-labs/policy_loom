"""Pi0.5 VLA training adapter using LeRobot implementation.

⚠️  IMPORTANT: Pi0.5 requires a custom transformers branch that conflicts with SmolVLA.
Install in a separate virtual environment:
    python -m venv venv-pi05
    source venv-pi05/bin/activate
    uv sync --extra pi05

Once the upstream transformers bug is fixed, this restriction will be removed.
"""

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from loom.training.adapter import register_adapter

logger = logging.getLogger(__name__)


def _check_lerobot_available() -> None:
    """Check if LeRobot is installed and import successful."""
    try:
        import lerobot  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Pi0.5 training requires LeRobot. Install with: uv sync --extra pi05\n"
            "Note: This must be done in a separate virtualenv due to transformers conflicts."
        ) from e


def _check_transformers_version() -> None:
    """Check transformers version compatibility for pi0.5."""
    try:
        from importlib.metadata import version

        from transformers import __version__ as tf_version

        logger.info(f"Using transformers version: {tf_version}")

        # Check if it's the custom branch
        if "git" not in version("transformers").lower():
            logger.warning(
                f"Pi0.5 requires custom transformers branch but found version {tf_version}. "
                "This may cause training issues. "
                "Install with: uv sync --extra pi05"
            )
    except Exception as e:
        logger.warning(f"Could not verify transformers version: {e}")


@register_adapter("pi05")
class Pi05Adapter:
    """Adapter for Physical Intelligence pi0.5 model.

    Wraps LeRobot's official pi0.5 implementation rather than reimplementing.
    Uses LeRobot's training infrastructure with policy_loom's data pipeline.

    Example:
        >>> config = {
        ...     "type": "pi05",
        ...     "pretrained_model_name_or_path": "lerobot/pi05_base",
        ...     "freeze_backbone": False,
        ... }
        >>> adapter = Pi05Adapter(config)
        >>> model = adapter.create_model()

    Args:
        config: Model configuration dict with keys:
            - type: "pi05"
            - pretrained_model_name_or_path: HuggingFace model path
            - freeze_backbone: Whether to freeze VLM backbone
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize pi0.5 adapter with configuration.

        Args:
            config: Model configuration dictionary

        Raises:
            ImportError: If lerobot is not installed
            ValueError: If config is invalid
        """
        _check_lerobot_available()
        _check_transformers_version()

        self.config = config
        self.pretrained_model_path = config.get("pretrained_model_name_or_path", "lerobot/pi05_base")
        self.freeze_backbone = config.get("freeze_backbone", False)

        logger.info(f"Initializing Pi05Adapter with model: {self.pretrained_model_path}")
        logger.info(f"Freeze backbone: {self.freeze_backbone}")

    def create_model(self) -> torch.nn.Module:
        """Create and initialize pi0.5 model from LeRobot.

        Returns:
            Pi0.5 PyTorch model ready for training

        Raises:
            ImportError: If required dependencies not installed
            ValueError: If model loading fails
        """
        try:
            from lerobot.common.policies.pi0.modeling_pi0 import Pi0Policy
        except ImportError as e:
            raise ImportError(
                "Could not import Pi0Policy from lerobot. "
                "Ensure lerobot is installed: uv sync --extra pi05"
            ) from e

        logger.info(f"Loading pi0.5 model from {self.pretrained_model_path}")

        try:
            # Load pretrained model
            model = Pi0Policy.from_pretrained(self.pretrained_model_path)

            # Freeze backbone if requested
            if self.freeze_backbone:
                logger.info("Freezing VLM backbone parameters")
                for name, param in model.named_parameters():
                    if "vlm" in name or "vision" in name:
                        param.requires_grad = False

            logger.info(f"Successfully loaded pi0.5 model: {self.pretrained_model_path}")
            return model

        except Exception as e:
            raise ValueError(f"Failed to load pi0.5 model from {self.pretrained_model_path}: {e}") from e

    def create_optimizer(
        self,
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Create optimizer for pi0.5 model.

        Args:
            model: The pi0.5 model to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient

        Returns:
            AdamW optimizer configured for pi0.5
        """
        # Only optimize parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        logger.info(f"Optimizing {len(trainable_params)} parameter groups")
        logger.info(f"Learning rate: {lr}, Weight decay: {weight_decay}")

        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def training_step(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute one training step for pi0.5.

        Args:
            model: Pi0.5 model in training mode
            batch: Batch of data (LeRobot format)
            device: Device to move data to

        Returns:
            Tuple of (loss tensor for backprop, metrics dict for logging)
        """
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass - pi0.5 returns dict with 'loss' key
        output = model(batch)

        # Extract loss
        if isinstance(output, dict) and "loss" in output:
            loss = output["loss"]
        else:
            # Fallback: assume output is the loss tensor
            loss = output

        # Collect metrics
        metrics = {
            "loss": loss.item(),
        }

        # Add additional metrics if available
        if isinstance(output, dict):
            for key, value in output.items():
                if key != "loss" and isinstance(value, torch.Tensor) and value.numel() == 1:
                    metrics[key] = value.item()

        return loss, metrics

    def eval_step(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        device: torch.device,
    ) -> dict[str, float]:
        """Execute one evaluation step for pi0.5.

        Args:
            model: Pi0.5 model in eval mode
            batch: Batch of data (LeRobot format)
            device: Device to move data to

        Returns:
            Metrics dict for logging with eval/ prefix
        """
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        with torch.no_grad():
            output = model(batch)

        # Extract loss
        if isinstance(output, dict) and "loss" in output:
            loss = output["loss"]
        else:
            loss = output

        # Collect metrics with eval/ prefix
        metrics = {
            "eval/loss": loss.item(),
        }

        # Add additional metrics if available
        if isinstance(output, dict):
            for key, value in output.items():
                if key != "loss" and isinstance(value, torch.Tensor) and value.numel() == 1:
                    metrics[f"eval/{key}"] = value.item()

        return metrics

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        batch_size: int,
        num_workers: int,
    ) -> tuple[DataLoader, DataLoader | None]:
        """Create train and eval dataloaders for pi0.5.

        Args:
            train_dataset: Training dataset (LeRobot format)
            eval_dataset: Optional evaluation dataset
            batch_size: Batch size
            num_workers: Number of dataloader workers

        Returns:
            Tuple of (train_loader, eval_loader)
            eval_loader is None if eval_dataset is None
        """
        # Create train dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # Pi0.5 may require consistent batch sizes
        )

        # Create eval dataloader if provided
        eval_loader = None
        if eval_dataset is not None:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )

        logger.info(f"Created dataloaders: train_batches={len(train_loader)}")
        if eval_loader:
            logger.info(f"Eval batches: {len(eval_loader)}")

        return train_loader, eval_loader
