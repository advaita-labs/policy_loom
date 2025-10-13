"""Model adapters for different VLA architectures."""

from loom.training.adapters.diffusion_policy import DiffusionPolicyAdapter

# Lazy import for Pi0.5 (requires separate environment)
try:
    from loom.training.adapters.pi05 import Pi05Adapter
    __all__ = ["DiffusionPolicyAdapter", "Pi05Adapter"]
except ImportError:
    # Pi0.5 dependencies not available
    __all__ = ["DiffusionPolicyAdapter"]
