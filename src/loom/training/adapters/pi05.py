"""Physical Intelligence Pi0.5 adapter using openpi package.

This adapter uses Physical Intelligence's official openpi implementation.
It supports:
- Training from scratch
- Loading from openpi checkpoints
- LoRA fine-tuning (if available)

⚠️  IMPORTANT: Install with pi05 extra:
    GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from loom.training.adapter import register_adapter

if TYPE_CHECKING:
    from loom.training.transforms.openpi_transform import OpenPITransform

logger = logging.getLogger(__name__)


def _check_openpi_available() -> None:
    """Check if openpi is installed."""
    try:
        import openpi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Pi0.5 training requires openpi. Install with:\n" "  GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05"
        ) from e


@register_adapter("pi05")
class Pi05Adapter:
    """Adapter for Physical Intelligence Pi0.5 using openpi implementation.

    This adapter integrates openpi's Pi0.5 model with policy_loom's training infrastructure.
    It accepts LeRobot format data as input and transforms it to openpi's expected format.

    Example config:
        ```yaml
        model:
          type: pi05
          pretrained_path: null  # Or path/URL to openpi checkpoint
          action_dim: 7
          action_horizon: 10
          use_lora: false
          lora_rank: 8
          freeze_backbone: false
        ```

    Args:
        config: Model configuration dict with keys:
            - type: "pi05"
            - pretrained_path: Path to openpi checkpoint (optional)
            - action_dim: Action space dimension
            - action_horizon: Number of future actions to predict
            - use_lora: Whether to use LoRA fine-tuning
            - lora_rank: LoRA rank (if use_lora=True)
            - freeze_backbone: Whether to freeze vision/language backbone
            - image_size: Image input size (default: [224, 224])
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Pi0.5 adapter.

        Args:
            config: Model configuration dictionary

        Raises:
            ImportError: If openpi is not installed
            ValueError: If config is invalid
        """
        _check_openpi_available()

        self.config = config
        self.pretrained_path = config.get("pretrained_path")
        self.action_dim = config.get("action_dim", 7)
        self.action_horizon = config.get("action_horizon", 10)
        self.use_lora = config.get("use_lora", False)
        self.lora_rank = config.get("lora_rank", 8)
        self.freeze_backbone = config.get("freeze_backbone", False)
        self.image_size = tuple(config.get("image_size", [224, 224]))

        logger.info("Initializing Pi05Adapter:")
        logger.info(f"  Pretrained path: {self.pretrained_path}")
        logger.info(f"  Action dim: {self.action_dim}, horizon: {self.action_horizon}")
        logger.info(f"  Use LoRA: {self.use_lora}, rank: {self.lora_rank if self.use_lora else 'N/A'}")
        logger.info(f"  Freeze backbone: {self.freeze_backbone}")

        # Lazy-load tokenizer and transform (will be initialized when needed)
        self._tokenizer: Any = None
        self._transform: OpenPITransform | None = None

    def create_model(self) -> nn.Module:
        """Create and initialize Pi0.5 model.

        Returns:
            Pi0.5 model ready for training

        Raises:
            ImportError: If required openpi modules not available
            ValueError: If model creation fails
        """
        try:
            # Import openpi modules
            from openpi.models.pi0 import Pi0Config, create_pi0_model
            from openpi.policies.policy_config import create_trained_policy
            from openpi.shared.download import maybe_download
        except ImportError as e:
            raise ImportError(
                f"Failed to import openpi modules: {e}\n" "Ensure openpi is installed: uv sync --extra native_pi05"
            ) from e

        # Load from checkpoint if provided
        if self.pretrained_path:
            logger.info(f"Loading Pi0.5 from checkpoint: {self.pretrained_path}")

            try:
                # Download checkpoint if it's a GCS URL
                if self.pretrained_path.startswith("gs://"):
                    checkpoint_dir = maybe_download(self.pretrained_path)
                else:
                    checkpoint_dir = Path(self.pretrained_path)

                # Load trained policy
                model = create_trained_policy(
                    checkpoint_dir=checkpoint_dir,
                    pytorch_device="cuda" if torch.cuda.is_available() else "cpu",
                )

                logger.info(f"Successfully loaded checkpoint from {checkpoint_dir}")

            except Exception as e:
                raise ValueError(f"Failed to load checkpoint from {self.pretrained_path}: {e}") from e

        else:
            # Create model from scratch
            logger.info("Creating Pi0.5 model from scratch")

            try:
                # Create model config
                model_config = Pi0Config(
                    action_dim=self.action_dim,
                    action_horizon=self.action_horizon,
                )

                # Create model
                model = create_pi0_model(model_config)

                logger.info("Successfully created Pi0.5 model from scratch")

            except Exception as e:
                raise ValueError(f"Failed to create Pi0.5 model: {e}") from e

        # Apply LoRA if requested
        if self.use_lora:
            logger.info(f"Applying LoRA with rank {self.lora_rank}")
            model = self._apply_lora(model)

        # Freeze backbone if requested
        if self.freeze_backbone:
            logger.info("Freezing vision/language backbone")
            self._freeze_backbone(model)

        return model

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA to model.

        Args:
            model: Pi0.5 model

        Returns:
            Model with LoRA applied
        """
        try:
            from openpi.models.lora import apply_lora

            model = apply_lora(model, rank=self.lora_rank)
            logger.info(f"Applied LoRA with rank {self.lora_rank}")

        except ImportError:
            logger.warning("LoRA not available in openpi. Skipping LoRA application.")
        except Exception as e:
            logger.warning(f"Failed to apply LoRA: {e}. Continuing without LoRA.")

        return model

    def _freeze_backbone(self, model: nn.Module) -> None:
        """Freeze vision/language backbone parameters.

        Args:
            model: Pi0.5 model
        """
        frozen_count = 0
        total_count = 0

        for name, param in model.named_parameters():
            total_count += 1
            # Freeze vision encoder and language model parameters
            if any(keyword in name.lower() for keyword in ["vision", "vlm", "siglip", "gemma", "transformer"]):
                param.requires_grad = False
                frozen_count += 1

        logger.info(f"Frozen {frozen_count}/{total_count} parameters")

    def create_optimizer(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Create optimizer for Pi0.5 model.

        Args:
            model: Pi0.5 model
            lr: Learning rate
            weight_decay: Weight decay

        Returns:
            AdamW optimizer
        """
        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)

        logger.info(f"Trainable parameters: {trainable_params_count:,} / {total_params:,}")
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
        model: nn.Module,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute one training step.

        Args:
            model: Pi0.5 model in training mode
            batch: Batch of data (LeRobot format)
            device: Device to run on

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        # Transform batch to OpenPI format
        obs_dict, actions = self._transform_batch(batch)

        # Move to device
        obs_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs_dict.items()}

        # Handle nested dicts (images, image_masks)
        for key in ["images", "image_masks"]:
            if key in obs_dict and isinstance(obs_dict[key], dict):
                obs_dict[key] = {
                    cam: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
                    for cam, tensor in obs_dict[key].items()
                }

        actions = actions.to(device)

        # Forward pass - Pi0.5 compute_loss expects (observations, actions)
        try:
            output = model.compute_loss(obs_dict, actions)

            # Extract loss
            if isinstance(output, dict) and "loss" in output:
                loss = output["loss"]
            else:
                loss = output

        except AttributeError:
            # Fallback: If compute_loss doesn't exist, try forward pass
            output = model(obs_dict)
            loss = nn.functional.mse_loss(output, actions)

        # Collect metrics
        metrics = {"loss": loss.item()}

        # Add additional metrics if available
        if isinstance(output, dict):
            for key, value in output.items():
                if key != "loss" and isinstance(value, torch.Tensor) and value.numel() == 1:
                    metrics[key] = value.item()

        return loss, metrics

    def eval_step(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        device: torch.device,
    ) -> dict[str, float]:
        """Execute one evaluation step.

        Args:
            model: Pi0.5 model in eval mode
            batch: Batch of data (LeRobot format)
            device: Device to run on

        Returns:
            Metrics dict with eval/ prefix
        """
        # Transform batch
        obs_dict, actions = self._transform_batch(batch)

        # Move to device
        obs_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs_dict.items()}

        for key in ["images", "image_masks"]:
            if key in obs_dict and isinstance(obs_dict[key], dict):
                obs_dict[key] = {
                    cam: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
                    for cam, tensor in obs_dict[key].items()
                }

        actions = actions.to(device)

        # Forward pass without gradients
        with torch.no_grad():
            try:
                output = model.compute_loss(obs_dict, actions)
                if isinstance(output, dict) and "loss" in output:
                    loss = output["loss"]
                else:
                    loss = output
            except AttributeError:
                output = model(obs_dict)
                loss = nn.functional.mse_loss(output, actions)

        # Collect metrics with eval/ prefix
        metrics = {"eval/loss": loss.item()}

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
        """Create train and eval dataloaders.

        Args:
            train_dataset: Training dataset (LeRobot format)
            eval_dataset: Optional evaluation dataset
            batch_size: Batch size
            num_workers: Number of dataloader workers

        Returns:
            Tuple of (train_loader, eval_loader)
        """
        # Use standard LeRobot collate function
        # Transformation happens in training_step/eval_step
        from loom.io.lerobot import collate_lerobot_batch

        # Create train dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_lerobot_batch,
            drop_last=True,
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
                collate_fn=collate_lerobot_batch,
                drop_last=False,
            )

        logger.info(f"Created dataloaders: train_batches={len(train_loader)}")
        if eval_loader:
            logger.info(f"Eval batches: {len(eval_loader)}")

        return train_loader, eval_loader

    def _get_tokenizer(self) -> Any:
        """Get or initialize tokenizer.

        Returns:
            Tokenizer instance or None
        """
        if self._tokenizer is None:
            try:
                from openpi.models.tokenizer import get_tokenizer

                self._tokenizer = get_tokenizer()
                logger.info("Initialized tokenizer for prompts")
            except ImportError:
                logger.info("Tokenizer not available, prompts will not be used")
                self._tokenizer = None
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer: {e}")
                self._tokenizer = None

        return self._tokenizer

    def _get_transform(self) -> "OpenPITransform":  # noqa: F821
        """Get or initialize transform.

        Returns:
            OpenPITransform instance
        """
        if self._transform is None:
            from loom.training.transforms.openpi_transform import OpenPITransform

            self._transform = OpenPITransform(
                tokenizer=self._get_tokenizer(),
                image_size=self.image_size,
                default_prompt=self.config.get("default_prompt"),
            )
            logger.info("Initialized OpenPI transform")

        return self._transform

    def _transform_batch(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Transform batch to OpenPI format with validation.

        Args:
            batch: Collated batch from standard collate_fn with keys:
                - observation: (B, state_dim) or None
                - images: list of dict[camera_name: ndarray] or None
                - action: (B, action_dim)

        Returns:
            Tuple of (obs_dict, actions)

        Raises:
            ValueError: If action dimensions don't match config
        """
        transform = self._get_transform()
        obs_dict, actions = transform(batch)

        # Validate action dimensions
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: config specifies action_dim={self.action_dim}, "
                f"but batch has shape {actions.shape} (action_dim={actions.shape[-1]}). "
                f"Check your model config and dataset."
            )

        # Check action horizon (LeRobot provides single-step actions)
        if self.action_horizon > 1:
            logger.warning(
                f"Model configured with action_horizon={self.action_horizon}, "
                f"but LeRobot datasets provide single-step actions. "
                f"Action sequences would require temporal windowing (not yet implemented)."
            )

        return obs_dict, actions
