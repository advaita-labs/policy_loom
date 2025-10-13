"""Pi0.5 model adapter for training.

This adapter integrates Physical Intelligence's Pi0.5 model with policy_loom's
training infrastructure. It follows the ModelAdapter protocol and leverages
existing OpenPI utilities while providing a clean interface for MP4/MCAP data.

Design Principles:
- Thin wrapper around OpenPI's PyTorch implementation
- Reuse existing policy_loom infrastructure (LeRobot conversion, OpenPITransform)
- Lazy imports for OpenPI dependencies (separate venv requirement)
- Action padding to 32 dimensions (OpenPI hardcoded requirement)

Data Flow:
    MP4/MCAP → LeRobot → collate_lerobot_batch → OpenPITransform → Pi0.5 model
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from loom.training.adapter import register_adapter

logger = logging.getLogger(__name__)


@register_adapter("pi05")
class Pi05Adapter:
    """Model adapter for Physical Intelligence's Pi0.5 VLA model.

    This adapter enables training Pi0.5 models using policy_loom's pipeline:
    1. MP4/MCAP data → LeRobot format (via convert_mp4_mcap_to_lerobot.py)
    2. LeRobot dataset → OpenPI Observation format (via OpenPITransform)
    3. Training with OpenPI's PI0Pytorch model

    Important Notes:
        - Requires separate venv due to transformers version conflict
        - Action dimension is hardcoded to 32 (OpenPI requirement)
        - Actions from robot are automatically padded to 32 dimensions
        - Uses OpenPI's preprocessing and tokenization utilities

    Example:
        >>> config = TrainingConfig.from_yaml("configs/pi05_minimal.yaml")
        >>> adapter = Pi05Adapter(config)
        >>> model = adapter.create_model()
        >>> optimizer = adapter.create_optimizer(model, lr=5e-5, weight_decay=0.01)
        >>> train_loader, eval_loader = adapter.create_dataloaders(
        ...     train_dataset, eval_dataset, batch_size=32, num_workers=4
        ... )
        >>> loss, metrics = adapter.training_step(model, next(iter(train_loader)), device)

    Args:
        config: Training configuration dict or object with model, training, data sections
    """

    def __init__(self, config: dict | Any):
        """Initialize Pi0.5 adapter with configuration.

        Args:
            config: Configuration dict or object. Expected structure:
                - model.action_dim: 32 (OpenPI hardcoded)
                - model.action_horizon: Action sequence length
                - model.max_token_len: Maximum token sequence length
                - model.paligemma_variant: PaliGemma model size
                - model.action_expert_variant: Action expert model size
                - model.dtype: Training precision (bfloat16 or float32)
                - data.default_prompt: Default task instruction
                - data.image_size: Target image size tuple
                - data.camera_name_mapping: Optional camera name remapping
                - checkpoints.pretrained_path: Optional pretrained weights path
        """
        # Handle both dict and object configs
        if isinstance(config, dict):
            self.config = config
            self.action_dim = config.get("action_dim", 32)
            self.action_horizon = config.get("action_horizon", 10)
            self.max_token_len = config.get("max_token_len", 180)
            self.paligemma_variant = config.get("paligemma_variant", "gemma_2b")
            self.action_expert_variant = config.get("action_expert_variant", "gemma_300m")
            self.pi05 = config.get("pi05", True)
            self.dtype_str = config.get("dtype", "bfloat16")
            self.default_prompt = config.get("default_prompt", "complete the task")
            self.image_size = tuple(config.get("image_size", (224, 224)))
            self.camera_name_mapping = config.get("camera_name_mapping", {})
            self.pretrained_path = config.get("pretrained_path")
            self.pytorch_weight_path = config.get("pytorch_weight_path")
        else:
            self.config = config
            self.action_dim = getattr(config.model, "action_dim", 32)
            self.action_horizon = getattr(config.model, "action_horizon", 10)
            self.max_token_len = getattr(config.model, "max_token_len", 180)
            self.paligemma_variant = getattr(config.model, "paligemma_variant", "gemma_2b")
            self.action_expert_variant = getattr(config.model, "action_expert_variant", "gemma_300m")
            self.pi05 = getattr(config.model, "pi05", True)
            self.dtype_str = getattr(config.model, "dtype", "bfloat16")
            self.default_prompt = getattr(config.data, "default_prompt", "complete the task")
            self.image_size = tuple(getattr(config.data, "image_size", (224, 224)))
            self.camera_name_mapping = getattr(config.data, "camera_name_mapping", {})
            self.pretrained_path = getattr(config.checkpoints, "pretrained_path", None)
            self.pytorch_weight_path = getattr(config.checkpoints, "pytorch_weight_path", None)

        # Validate action dimension
        if self.action_dim != 32:
            logger.warning(
                f"OpenPI requires action_dim=32, but config has {self.action_dim}. "
                f"Actions will be padded/truncated to 32 dimensions."
            )
            self.action_dim = 32

        logger.info(f"Initialized Pi0.5 adapter:")
        logger.info(f"  Action dim: {self.action_dim} (OpenPI hardcoded)")
        logger.info(f"  Action horizon: {self.action_horizon}")
        logger.info(f"  Max token len: {self.max_token_len}")
        logger.info(f"  PaliGemma variant: {self.paligemma_variant}")
        logger.info(f"  Dtype: {self.dtype_str}")

    def create_model(self) -> nn.Module:
        """Create and initialize Pi0.5 model.

        This method:
        1. Imports OpenPI dependencies (lazy import)
        2. Creates Pi0Config from adapter configuration
        3. Initializes PI0Pytorch model
        4. Optionally loads pretrained weights

        Returns:
            Initialized PI0Pytorch model

        Raises:
            ImportError: If OpenPI package is not installed
            FileNotFoundError: If pretrained weights path doesn't exist
        """
        try:
            from openpi.models.pi0_config import Pi0Config
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        except ImportError as e:
            raise ImportError(
                "OpenPI package required for Pi0.5 training. "
                "Install in separate venv: "
                "python -m venv venv-pi05 && source venv-pi05/bin/activate && "
                "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra pi05"
            ) from e

        logger.info("Creating Pi0.5 model...")

        # Create Pi0Config
        model_config = Pi0Config(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
            paligemma_variant=self.paligemma_variant,
            action_expert_variant=self.action_expert_variant,
            pi05=self.pi05,
            dtype=self.dtype_str,
        )

        # Create model
        model = PI0Pytorch(model_config)

        # Load pretrained weights if specified
        if self.pytorch_weight_path:
            logger.info(f"Loading pretrained weights from: {self.pytorch_weight_path}")
            try:
                import safetensors.torch
                safetensors.torch.load_model(model, self.pytorch_weight_path)
                logger.info("Successfully loaded pretrained weights")
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                raise

        logger.info("Pi0.5 model created successfully")
        return model

    def create_optimizer(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer for Pi0.5 model.

        Uses OpenPI's default hyperparameters:
        - betas: (0.9, 0.999)
        - eps: 1e-8

        Args:
            model: Pi0.5 model
            lr: Learning rate
            weight_decay: Weight decay coefficient

        Returns:
            Configured AdamW optimizer
        """
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )

        logger.info(f"Created AdamW optimizer: lr={lr}, weight_decay={weight_decay}")
        return optimizer

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        batch_size: int,
        num_workers: int,
    ) -> tuple[DataLoader, DataLoader]:
        """Create train and eval dataloaders with OpenPI preprocessing.

        This method:
        1. Uses existing LeRobotDataset (already converted from MP4/MCAP)
        2. Applies collate_lerobot_batch for batching
        3. Uses OpenPITransform for observation format conversion
        4. Handles action padding to 32 dimensions

        Args:
            train_dataset: Training dataset (LeRobot format)
            eval_dataset: Evaluation dataset (LeRobot format)
            batch_size: Batch size
            num_workers: Number of dataloader workers

        Returns:
            Tuple of (train_loader, eval_loader)
        """
        from loom.io.lerobot import collate_lerobot_batch
        from loom.io.lerobot.pi05 import convert_lerobot_batch_to_pi05
        from loom.training.transforms.openpi_transform import OpenPITransform

        logger.info("Creating dataloaders with OpenPI transforms...")

        # Create OpenPI transform
        transform = OpenPITransform(
            tokenizer=self._get_tokenizer(),
            image_size=self.image_size,
            default_prompt=self.default_prompt,
            camera_name_mapping=self.camera_name_mapping,
        )

        def collate_fn(batch: list[dict]) -> tuple:
            """Collate batch and convert to OpenPI format.

            Args:
                batch: List of LeRobot-style sample dicts

            Returns:
                Tuple of (Observation, actions_tensor)
            """
            # Collate into batched dict
            lerobot_batch = collate_lerobot_batch(batch)

            # Convert to OpenPI format
            observation, actions = convert_lerobot_batch_to_pi05(
                lerobot_batch,
                transform=transform,
                default_prompt=self.default_prompt,
            )

            return observation, actions

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,  # OpenPI expects consistent batch sizes
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

        logger.info(f"Created dataloaders: batch_size={batch_size}, num_workers={num_workers}")
        return train_loader, eval_loader

    def _get_tokenizer(self):
        """Get tokenizer for text prompts.

        Returns:
            PaliGemma tokenizer instance or None
        """
        try:
            from openpi.models.tokenizer import PaligemmaTokenizer

            return PaligemmaTokenizer(max_len=self.max_token_len)
        except ImportError:
            logger.warning("OpenPI tokenizer not available, using None")
            return None

    def training_step(
        self,
        model: nn.Module,
        batch: tuple,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute one training step.

        This method:
        1. Preprocesses observation using OpenPI's preprocessing
        2. Calls model.forward(observation, actions) - returns loss
        3. Returns loss tensor and metrics dict

        Args:
            model: Pi0.5 model in training mode
            batch: Tuple of (Observation, actions) from dataloader
            device: Device to move data to

        Returns:
            Tuple of (loss_tensor, metrics_dict)
                loss_tensor: Scalar loss for backpropagation
                metrics_dict: Dictionary with 'loss' key
        """
        try:
            from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
        except ImportError as e:
            raise ImportError("OpenPI preprocessing required for training step") from e

        model.train()

        # Unpack batch
        observation, actions = batch

        # Move to device
        observation = self._move_observation_to_device(observation, device)
        actions = actions.to(device, dtype=torch.float32)

        # Preprocess observation (image augmentation, etc.)
        observation = preprocess_observation_pytorch(
            observation,
            train=True,
            image_keys=list(observation.images.keys()) if hasattr(observation, 'images') else [],
        )

        # Forward pass - OpenPI's model.forward returns loss directly
        loss = model(observation, actions)

        # Handle different loss formats
        if isinstance(loss, (list, tuple)):
            loss = torch.stack(loss).mean()
        elif loss.ndim > 0:
            loss = loss.mean()

        # Create metrics dict
        metrics = {
            "loss": loss.item(),
        }

        return loss, metrics

    def eval_step(
        self,
        model: nn.Module,
        batch: tuple,
        device: torch.device,
    ) -> dict[str, float]:
        """Execute one evaluation step.

        Similar to training_step but:
        - Uses torch.no_grad()
        - Sets model to eval mode
        - Returns metrics with "eval/" prefix

        Args:
            model: Pi0.5 model
            batch: Tuple of (Observation, actions)
            device: Device to move data to

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
        except ImportError as e:
            raise ImportError("OpenPI preprocessing required for eval step") from e

        model.eval()

        with torch.no_grad():
            # Unpack batch
            observation, actions = batch

            # Move to device
            observation = self._move_observation_to_device(observation, device)
            actions = actions.to(device, dtype=torch.float32)

            # Preprocess observation (no augmentation in eval)
            observation = preprocess_observation_pytorch(
                observation,
                train=False,
                image_keys=list(observation.images.keys()) if hasattr(observation, 'images') else [],
            )

            # Forward pass
            loss = model(observation, actions)

            # Handle different loss formats
            if isinstance(loss, (list, tuple)):
                loss = torch.stack(loss).mean()
            elif loss.ndim > 0:
                loss = loss.mean()

            # Create metrics dict with eval prefix
            metrics = {
                "eval/loss": loss.item(),
            }

        return metrics

    def _move_observation_to_device(self, observation, device: torch.device):
        """Move observation components to device.

        Args:
            observation: OpenPI Observation object
            device: Target device

        Returns:
            Observation with all tensors moved to device
        """
        # OpenPI Observation is a frozen dataclass - create a new one
        import torch
        from dataclasses import replace

        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            elif isinstance(x, (list, tuple)):
                return type(x)(_to_device(v) for v in x)
            else:
                return x

        # Create new observation with moved tensors (frozen dataclass)
        moved_fields = {}
        for key in observation.__dataclass_fields__:
            value = getattr(observation, key)
            moved_fields[key] = _to_device(value)
        
        return type(observation)(**moved_fields)
