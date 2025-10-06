"""Model training infrastructure for policy_loom."""

from loom.training.adapter import ModelAdapter, get_adapter, list_adapters, register_adapter

# Import adapters to register them
from loom.training.adapters.diffusion_policy import DiffusionPolicyAdapter  # noqa: F401
from loom.training.checkpoint import CheckpointManager
from loom.training.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    LRSchedulerConfig,
    TrainingConfig,
    TrainingParams,
    WandbConfig,
)
from loom.training.logger import TrainingLogger
from loom.training.metrics import MetricsTracker
from loom.training.trainer import Trainer

__all__ = [
    # Trainer
    "Trainer",
    # Adapter protocol and registry
    "ModelAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    # Configuration
    "TrainingConfig",
    "TrainingParams",
    "LRSchedulerConfig",
    "CheckpointConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "WandbConfig",
    # Utilities
    "CheckpointManager",
    "TrainingLogger",
    "MetricsTracker",
]
