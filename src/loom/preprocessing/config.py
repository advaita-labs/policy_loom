"""Configuration dataclasses for model preprocessing."""

from dataclasses import dataclass, field
from enum import Enum


class NormalizationMode(str, Enum):
    """Normalization strategies for inputs/outputs."""

    IDENTITY = "identity"  # No normalization
    MEAN_STD = "mean_std"  # Z-score normalization
    MIN_MAX = "min_max"  # Scale to [0, 1] or [-1, 1]


@dataclass
class ImagePreprocessingConfig:
    """Configuration for image preprocessing.

    Attributes:
        target_size: Target image size (height, width)
        resize_with_padding: If True, pad to maintain aspect ratio. If False, distort to target_size.
        normalize: Whether to normalize with ImageNet stats
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)
        to_tensor: Convert numpy array to torch tensor
    """

    target_size: tuple[int, int] = (512, 512)
    resize_with_padding: bool = True
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    to_tensor: bool = True


@dataclass
class SmolVLAPreprocessingConfig:
    """Configuration for SmolVLA model preprocessing.

    Attributes:
        camera_names: List of camera names to extract from Sample. Order matters!
        image_config: Image preprocessing settings
        vlm_model_name: VLM model name for tokenizer
        max_language_tokens: Maximum sequence length for language tokens
        n_obs_steps: Number of observation steps for state history
        max_state_dim: Maximum state dimension (for padding)
        state_normalization: Normalization mode for proprioceptive state
        chunk_size: Maximum action chunk size
        max_action_dim: Maximum action dimension (for padding)
        action_normalization: Normalization mode for actions
        state_mean: State normalization mean (computed from dataset)
        state_std: State normalization std (computed from dataset)
        action_mean: Action normalization mean (computed from dataset)
        action_std: Action normalization std (computed from dataset)
        device: Device to move tensors to (cpu, cuda, mps)
    """

    camera_names: list[str] = field(default_factory=lambda: ["observation.image"])
    image_config: ImagePreprocessingConfig = field(default_factory=ImagePreprocessingConfig)
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    max_language_tokens: int = 256
    n_obs_steps: int = 1
    max_state_dim: int = 32
    state_normalization: NormalizationMode = NormalizationMode.MEAN_STD
    chunk_size: int = 50
    max_action_dim: int = 32
    action_normalization: NormalizationMode = NormalizationMode.MEAN_STD
    state_mean: list[float] | None = None
    state_std: list[float] | None = None
    action_mean: list[float] | None = None
    action_std: list[float] | None = None
    device: str = "cpu"


@dataclass
class DiffusionPolicyPreprocessingConfig:
    """Configuration for Diffusion Policy model preprocessing.

    Attributes:
        camera_names: List of camera names to extract from Sample. Order matters!
        image_config: Image preprocessing settings
        obs_horizon: Number of past observations to stack (observation history)
        action_horizon: Number of future actions to predict (action chunking)
        state_mean: State normalization mean (computed from dataset)
        state_std: State normalization std (computed from dataset)
        action_mean: Action normalization mean (computed from dataset)
        action_std: Action normalization std (computed from dataset)
        device: Device to move tensors to (cpu, cuda, mps)

    Note:
        Unlike SmolVLA, Diffusion Policy does not pad state/action to fixed dimensions.
        State and action tensors retain their natural dimensions.
    """

    camera_names: list[str] = field(default_factory=lambda: ["observation.image"])
    image_config: ImagePreprocessingConfig = field(
        default_factory=lambda: ImagePreprocessingConfig(
            target_size=(96, 96),
            resize_with_padding=False,
            normalize=False,  # Diffusion Policy typically normalizes to [0,1] not ImageNet
            to_tensor=True,
        )
    )
    obs_horizon: int = 2  # Number of past observations to stack
    action_horizon: int = 8  # Number of future actions to predict
    state_mean: list[float] | None = None
    state_std: list[float] | None = None
    action_mean: list[float] | None = None
    action_std: list[float] | None = None
    device: str = "cpu"
