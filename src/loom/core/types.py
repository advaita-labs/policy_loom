"""Core data types for policy_loom."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


@dataclass
class CameraImage:
    """Container for a single camera's RGB image.

    Attributes:
        name: Camera identifier (e.g., "left_cam", "right_cam", "middle_cam")
        image: RGB image data as numpy array, shape (H, W, C), dtype uint8 or float32
    """

    name: str
    image: npt.NDArray[np.uint8] | npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate camera image after initialization."""
        if self.image.ndim != 3:
            raise ValueError(f"Expected image to have 3 dimensions (H, W, C), got shape {self.image.shape}")


@dataclass
class Sample:
    """Canonical data sample representing a single timestep.

    This is the unified representation that flows through the entire pipeline:
    ingest → transforms → export.

    Attributes:
        timestamp: Time in seconds (float) or nanoseconds (int) since epoch
        cameras: List of camera images (empty for samples without vision data)
        proprio: Proprioceptive data (joint positions, velocities, etc.)
        action: Action taken at this timestep
        metadata: Additional fields (episode_id, frame_idx, source, etc.)
    """

    timestamp: float | int
    cameras: list[CameraImage] = field(default_factory=list)
    proprio: npt.NDArray[np.float32] | None = None
    action: npt.NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate sample after initialization."""
        if self.proprio is not None and self.proprio.ndim != 1:
            raise ValueError(f"Expected proprio to be 1D, got shape {self.proprio.shape}")
        if self.action is not None and self.action.ndim != 1:
            raise ValueError(f"Expected action to be 1D, got shape {self.action.shape}")

    def get_camera(self, name: str) -> CameraImage | None:
        """Get camera image by name.

        Args:
            name: Camera name to look up

        Returns:
            CameraImage if found, None otherwise
        """
        for cam in self.cameras:
            if cam.name == name:
                return cam
        return None

    def add_camera(self, name: str, image: npt.NDArray[np.uint8] | npt.NDArray[np.float32]) -> None:
        """Add a camera image to this sample.

        Args:
            name: Camera identifier
            image: RGB image array

        Raises:
            ValueError: If camera with this name already exists
        """
        if self.get_camera(name) is not None:
            raise ValueError(f"Camera '{name}' already exists in sample")
        self.cameras.append(CameraImage(name=name, image=image))


# Model Input Types


@dataclass
class SmolVLAInput:
    """Preprocessed input for SmolVLA model (single sample, unbatched).

    This represents a single preprocessed sample ready for the model.
    Use SmolVLAPreprocessor.collate_fn() to batch multiple inputs.

    Attributes:
        images: List of preprocessed image tensors, shape (C, H, W) each
        language_instruction: Task instruction string
        state: Proprioceptive state tensor, shape (state_dim,) or (n_obs_steps, state_dim)
        action: Action tensor, shape (action_dim,) or (chunk_size, action_dim)
    """

    images: list[torch.Tensor]
    language_instruction: str
    state: torch.Tensor
    action: torch.Tensor


@dataclass
class SmolVLABatchInput:
    """Batched input for SmolVLA model training.

    This is the format expected by SmolVLA model's forward pass.
    All tensors have batch dimension as first dimension.

    Attributes:
        observation_images: Dictionary mapping camera names to image tensors.
            Keys: "observation.image", "observation.image_1", etc.
            Values: Tensors of shape (B, C, H, W)
        language_tokens: Tokenized language input, shape (B, seq_len)
        language_attention_mask: Attention mask for language tokens, shape (B, seq_len)
        state: Proprioceptive state, shape (B, n_obs_steps, state_dim)
        action: Action trajectory, shape (B, chunk_size, action_dim)
    """

    observation_images: dict[str, torch.Tensor]
    language_tokens: torch.Tensor
    language_attention_mask: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
