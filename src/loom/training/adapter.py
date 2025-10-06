"""Model adapter protocol and registry for training different VLA models."""

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader, Dataset


class ModelAdapter(Protocol):
    """Protocol for model-specific training logic.

    Each VLA model (Diffusion Policy, SmolVLA, etc.) implements this protocol
    to plug into the generic Trainer.

    Example:
        >>> @register_adapter("my_model")
        >>> class MyModelAdapter:
        ...     def __init__(self, config: dict[str, Any]):
        ...         self.config = config
        ...
        ...     def create_model(self) -> torch.nn.Module:
        ...         return MyModel(**self.config)
        ...
        ...     # ... implement other methods
    """

    def create_model(self) -> torch.nn.Module:
        """Create and initialize the model.

        Returns:
            Initialized PyTorch model ready for training

        Raises:
            ImportError: If required dependencies are not installed
        """
        ...

    def create_optimizer(self, model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        """Create optimizer for the model.

        Args:
            model: The model to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient

        Returns:
            Configured optimizer
        """
        ...

    def training_step(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute one training step.

        Args:
            model: Model in training mode
            batch: Batch of data from DataLoader
            device: Device to move data to

        Returns:
            Tuple of (loss tensor for backprop, metrics dict for logging)

        Example:
            >>> loss, metrics = adapter.training_step(model, batch, device)
            >>> metrics
            {'loss': 0.123, 'mse': 0.045}
        """
        ...

    def eval_step(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        device: torch.device,
    ) -> dict[str, float]:
        """Execute one evaluation step.

        Args:
            model: Model in eval mode
            batch: Batch of data from DataLoader
            device: Device to move data to

        Returns:
            Metrics dict for logging

        Example:
            >>> metrics = adapter.eval_step(model, batch, device)
            >>> metrics
            {'eval/loss': 0.098, 'eval/mse': 0.032}
        """
        ...

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        batch_size: int,
        num_workers: int,
    ) -> tuple[DataLoader, DataLoader | None]:
        """Create train and eval dataloaders with model-specific collation.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            batch_size: Batch size
            num_workers: Number of dataloader workers

        Returns:
            Tuple of (train_loader, eval_loader)
            eval_loader is None if eval_dataset is None

        Note:
            This method should handle model-specific collation functions.
        """
        ...


# Registry for model adapters
_ADAPTERS: dict[str, type] = {}


def register_adapter(name: str) -> Callable[[type], type]:
    """Decorator to register a model adapter.

    Args:
        name: Unique name for the adapter (used in config model.type)

    Returns:
        Decorator function

    Example:
        >>> @register_adapter("diffusion_policy")
        >>> class DiffusionPolicyAdapter:
        ...     pass
    """

    def decorator(cls: type) -> type:
        if name in _ADAPTERS:
            raise ValueError(f"Adapter '{name}' is already registered")
        _ADAPTERS[name] = cls
        return cls

    return decorator


def get_adapter(name: str, config: dict[str, Any]) -> Any:
    """Get adapter instance by name.

    Args:
        name: Registered adapter name
        config: Model-specific configuration dict

    Returns:
        Instantiated adapter

    Raises:
        ValueError: If adapter name is not registered

    Example:
        >>> config = {"type": "diffusion_policy", "obs_horizon": 2}
        >>> adapter = get_adapter("diffusion_policy", config)
    """
    if name not in _ADAPTERS:
        available = ", ".join(_ADAPTERS.keys()) if _ADAPTERS else "none"
        raise ValueError(f"Unknown adapter: '{name}'. Available adapters: {available}")

    return _ADAPTERS[name](config)


def list_adapters() -> list[str]:
    """List all registered adapter names.

    Returns:
        List of registered adapter names

    Example:
        >>> list_adapters()
        ['diffusion_policy', 'smolvla']
    """
    return list(_ADAPTERS.keys())
