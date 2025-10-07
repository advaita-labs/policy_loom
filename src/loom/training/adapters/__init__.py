"""Model adapters for different VLA architectures."""

from loom.training.adapters.diffusion_policy import DiffusionPolicyAdapter

# Pi0.5 adapter with lazy import to avoid dependency issues
try:
    from loom.training.adapters.pi05 import Pi05Adapter

    __all__ = ["DiffusionPolicyAdapter", "Pi05Adapter"]
except ImportError:
    # Pi0.5 dependencies not installed (requires separate venv)
    __all__ = ["DiffusionPolicyAdapter"]
