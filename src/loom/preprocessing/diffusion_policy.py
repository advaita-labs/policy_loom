"""Diffusion Policy model preprocessor."""

import logging

import numpy as np
import torch

from loom.core.ports import Preprocessor
from loom.core.types import DiffusionPolicyBatchInput, DiffusionPolicyInput, Sample
from loom.preprocessing.base import BasePreprocessor
from loom.preprocessing.config import DiffusionPolicyPreprocessingConfig

logger = logging.getLogger(__name__)


class DiffusionPolicyPreprocessor(Preprocessor[DiffusionPolicyInput, DiffusionPolicyBatchInput], BasePreprocessor):
    """Preprocessor for Diffusion Policy model.

    Converts sequences of policy_loom Sample objects to Diffusion Policy input format.
    Unlike SmolVLA which processes single samples, Diffusion Policy requires:
    - Observation history (obs_horizon past observations stacked)
    - Action chunking (action_horizon future actions)

    Args:
        config: Diffusion Policy preprocessing configuration

    Example:
        >>> config = DiffusionPolicyPreprocessingConfig(
        ...     camera_names=["cam1"],
        ...     obs_horizon=2,
        ...     action_horizon=8,
        ...     state_mean=[0.0] * 7,
        ...     state_std=[1.0] * 7,
        ... )
        >>> preprocessor = DiffusionPolicyPreprocessor(config)
        >>>
        >>> # Preprocess sequence of samples
        >>> samples = [...]  # List of obs_horizon + action_horizon samples
        >>> model_input = preprocessor.preprocess_sample_sequence(samples)
        >>>
        >>> # Use with DataLoader
        >>> dataset = SequenceDataset(sequences, preprocessor)
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_fn=preprocessor.collate_fn
        ... )
    """

    def __init__(self, config: DiffusionPolicyPreprocessingConfig) -> None:
        """Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration

        Raises:
            ValueError: If config is invalid
        """
        self.config = config

        # Validate configuration
        self._validate_config()

        # Convert normalization stats to numpy arrays
        if config.state_mean is not None and config.state_std is not None:
            self.state_mean = np.array(config.state_mean, dtype=np.float32)
            self.state_std = np.array(config.state_std, dtype=np.float32)
        else:
            self.state_mean = None
            self.state_std = None

        if config.action_mean is not None and config.action_std is not None:
            self.action_mean = np.array(config.action_mean, dtype=np.float32)
            self.action_std = np.array(config.action_std, dtype=np.float32)
        else:
            self.action_mean = None
            self.action_std = None

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config is invalid
        """
        # Check camera list is not empty
        if not self.config.camera_names:
            raise ValueError("camera_names cannot be empty")

        # Check horizons are positive
        if self.config.obs_horizon <= 0:
            raise ValueError(f"obs_horizon must be positive, got {self.config.obs_horizon}")

        if self.config.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.config.action_horizon}")

        # Check mean/std dimensions match
        if self.config.state_mean is not None and self.config.state_std is not None:
            if len(self.config.state_mean) != len(self.config.state_std):
                raise ValueError(
                    f"state_mean length ({len(self.config.state_mean)}) "
                    f"does not match state_std length ({len(self.config.state_std)})"
                )

        if self.config.action_mean is not None and self.config.action_std is not None:
            if len(self.config.action_mean) != len(self.config.action_std):
                raise ValueError(
                    f"action_mean length ({len(self.config.action_mean)}) "
                    f"does not match action_std length ({len(self.config.action_std)})"
                )

    def preprocess_sample_sequence(self, samples: list[Sample]) -> DiffusionPolicyInput:
        """Convert sequence of Samples to Diffusion Policy input format.

        Args:
            samples: List of samples in temporal order.
                     Must contain at least obs_horizon + action_horizon samples.
                     - First obs_horizon samples: stacked for observation history
                     - Samples [obs_horizon-1 : obs_horizon-1+action_horizon]: action chunk
                       (action chunk includes action at current observation timestep)

        Example:
            For obs_horizon=2, action_horizon=4, given samples [t0, t1, t2, t3, t4]:
            - Observation history: [t0, t1]
            - Action chunk: [t1, t2, t3, t4] (starts at current timestep t1)

        Returns:
            DiffusionPolicyInput with stacked observations and action chunk

        Raises:
            ValueError: If insufficient samples
            ValueError: If required cameras are missing
            ValueError: If proprio or action is None
            ValueError: If data contains NaN or inf
        """
        required_samples = self.config.obs_horizon + self.config.action_horizon
        if len(samples) < required_samples:
            raise ValueError(
                f"Need at least {required_samples} samples "
                f"(obs_horizon={self.config.obs_horizon} + action_horizon={self.config.action_horizon}), "
                f"got {len(samples)}"
            )

        # Extract observation history (first obs_horizon samples)
        obs_samples = samples[: self.config.obs_horizon]

        # Extract action sequence (starting from current observation timestep)
        # Action chunk starts at obs_horizon-1 (current time) and includes next action_horizon actions
        action_samples = samples[self.config.obs_horizon - 1 : self.config.obs_horizon - 1 + self.config.action_horizon]

        # Process images for each camera
        observation_images = {}
        for camera_name in self.config.camera_names:
            camera_images = []
            for sample in obs_samples:
                camera = sample.get_camera(camera_name)
                if camera is None:
                    raise ValueError(
                        f"Camera '{camera_name}' not found in sample at t={sample.timestamp}. "
                        f"Available cameras: {[c.name for c in sample.cameras]}"
                    )
                preprocessed_img = self._preprocess_image(camera.image)
                camera_images.append(preprocessed_img)

            # Stack images along time dimension: (obs_horizon, C, H, W)
            stacked_images = torch.stack(camera_images, dim=0)
            observation_images[camera_name] = stacked_images

        # Process states (stack observation history)
        states = []
        for sample in obs_samples:
            if sample.proprio is None:
                raise ValueError(f"Sample at t={sample.timestamp} must have proprio (state) data")
            state = self._preprocess_state(sample.proprio)
            states.append(state)

        # Stack states: (obs_horizon, state_dim)
        stacked_states = torch.stack(states, dim=0)

        # Process actions (extract action chunk)
        actions = []
        for sample in action_samples:
            if sample.action is None:
                raise ValueError(f"Sample at t={sample.timestamp} must have action data")
            action = self._preprocess_action(sample.action)
            actions.append(action)

        # Stack actions: (action_horizon, action_dim)
        stacked_actions = torch.stack(actions, dim=0)

        return DiffusionPolicyInput(
            observation_images=observation_images,
            state=stacked_states,
            action=stacked_actions,
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image.

        Pipeline: uint8 [0,255] → float32 → /255 → [0,1] → channels-first

        Args:
            image: Input image (H, W, C) in RGB format, uint8 or float32

        Returns:
            Preprocessed image tensor (C, H, W) in channels-first format

        Raises:
            ValueError: If image shape or dtype is invalid
        """
        if image.ndim != 3:
            raise ValueError(f"Image must have 3 dimensions (H, W, C), got shape {image.shape}")

        if image.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels (RGB), got {image.shape[2]}")

        # Handle different input types
        if image.dtype == np.uint8:
            # Convert uint8 [0, 255] to float32 [0, 255]
            image_float = image.astype(np.float32)
        elif image.dtype == np.float32:
            # Assume already in [0, 255] or [0, 1] range
            if image.max() <= 1.0:
                # Already in [0, 1], scale to [0, 255]
                image_float = image * 255.0
            else:
                image_float = image
        else:
            raise ValueError(f"Image dtype must be uint8 or float32, got {image.dtype}")

        # Convert back to uint8 for resizing
        image_uint8 = image_float.astype(np.uint8)

        # Resize image
        if self.config.image_config.resize_with_padding:
            resized = self.resize_image_with_padding(
                image_uint8,
                self.config.image_config.target_size,
                pad_value=0,
            )
        else:
            resized = self.resize_image_distort(
                image_uint8,
                self.config.image_config.target_size,
            )

        # Convert to float32 and scale to [0, 1]
        image_normalized = resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization if enabled
        if self.config.image_config.normalize:
            mean = np.array(self.config.image_config.mean, dtype=np.float32)
            std = np.array(self.config.image_config.std, dtype=np.float32)
            # Reshape for broadcasting: (1, 1, 3)
            mean = mean.reshape(1, 1, 3)
            std = std.reshape(1, 1, 3)
            image_normalized = (image_normalized - mean) / std

        # Validate no NaN or inf
        self.validate_no_nan(image_normalized, "image")
        self.validate_no_inf(image_normalized, "image")

        # Convert to tensor and change to channels-first (C, H, W)
        tensor = torch.from_numpy(image_normalized).float()
        tensor = tensor.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

        return tensor

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess proprioceptive state.

        Args:
            state: Input state vector

        Returns:
            Normalized state tensor (no padding for diffusion policy)

        Raises:
            ValueError: If state contains NaN or inf
        """
        # Validate
        self.validate_no_nan(state, "proprio/state")
        self.validate_no_inf(state, "proprio/state")

        # Normalize if stats are provided
        if self.state_mean is not None and self.state_std is not None:
            # Ensure dimensions match (only normalize available dims)
            state_dim = min(len(state), len(self.state_mean))

            # Warn if dimensions don't match
            if len(state) != len(self.state_mean):
                extra_msg = (
                    "Extra dimensions will remain unnormalized."
                    if len(state) > len(self.state_mean)
                    else "Some normalization stats will be unused."
                )
                logger.warning(
                    f"State dimension ({len(state)}) does not match normalization "
                    f"stats dimension ({len(self.state_mean)}). {extra_msg}"
                )

            normalized = self.normalize_mean_std(
                state[:state_dim],
                self.state_mean[:state_dim],
                self.state_std[:state_dim],
            )
            # If state is longer than mean/std, keep extra dims unnormalized
            if len(state) > state_dim:
                normalized = np.concatenate([normalized, state[state_dim:]])
        else:
            normalized = state.copy()

        return torch.from_numpy(normalized).float()

    def _preprocess_action(self, action: np.ndarray) -> torch.Tensor:
        """Preprocess action.

        Args:
            action: Input action vector

        Returns:
            Normalized action tensor (no padding for diffusion policy)

        Raises:
            ValueError: If action contains NaN or inf
        """
        # Validate
        self.validate_no_nan(action, "action")
        self.validate_no_inf(action, "action")

        # Normalize if stats are provided
        if self.action_mean is not None and self.action_std is not None:
            # Ensure dimensions match (only normalize available dims)
            action_dim = min(len(action), len(self.action_mean))

            # Warn if dimensions don't match
            if len(action) != len(self.action_mean):
                extra_msg = (
                    "Extra dimensions will remain unnormalized."
                    if len(action) > len(self.action_mean)
                    else "Some normalization stats will be unused."
                )
                logger.warning(
                    f"Action dimension ({len(action)}) does not match normalization "
                    f"stats dimension ({len(self.action_mean)}). {extra_msg}"
                )

            normalized = self.normalize_mean_std(
                action[:action_dim],
                self.action_mean[:action_dim],
                self.action_std[:action_dim],
            )
            # If action is longer than mean/std, keep extra dims unnormalized
            if len(action) > action_dim:
                normalized = np.concatenate([normalized, action[action_dim:]])
        else:
            normalized = action.copy()

        return torch.from_numpy(normalized).float()

    def collate_fn(self, batch: list[DiffusionPolicyInput]) -> DiffusionPolicyBatchInput:
        """Collate batch for Diffusion Policy model.

        Args:
            batch: List of DiffusionPolicyInput from preprocess_sample_sequence()

        Returns:
            DiffusionPolicyBatchInput with batched tensors

        Raises:
            ValueError: If batch is empty
            ValueError: If batch contains inconsistent camera counts
        """
        if not batch:
            raise ValueError("Cannot collate empty batch")

        # Validate camera counts are consistent
        camera_keys = set(batch[0].observation_images.keys())
        for item in batch:
            if set(item.observation_images.keys()) != camera_keys:
                raise ValueError(
                    f"Inconsistent camera counts in batch: expected {len(camera_keys)}, "
                    f"got {len(item.observation_images)}"
                )

        # Stack images by camera
        observation_images = {}
        for camera_name in camera_keys:
            camera_tensors = [item.observation_images[camera_name] for item in batch]
            stacked = torch.stack(camera_tensors, dim=0)  # (B, obs_horizon, C, H, W)
            observation_images[camera_name] = stacked

        # Stack states
        state_tensors = [item.state for item in batch]
        state_stacked = torch.stack(state_tensors, dim=0)  # (B, obs_horizon, state_dim)

        # Stack actions
        action_tensors = [item.action for item in batch]
        action_stacked = torch.stack(action_tensors, dim=0)  # (B, action_horizon, action_dim)

        # Move to device if specified
        device = torch.device(self.config.device)
        observation_images = {k: v.to(device) for k, v in observation_images.items()}
        state_stacked = state_stacked.to(device)
        action_stacked = action_stacked.to(device)

        return DiffusionPolicyBatchInput(
            observation_images=observation_images,
            state=state_stacked,
            action=action_stacked,
        )

    def preprocess_sample(self, sample: Sample) -> DiffusionPolicyInput:
        """Single sample preprocessing not supported for Diffusion Policy.

        Diffusion Policy requires sequences of samples for observation history
        and action chunking. Use preprocess_sample_sequence() instead.

        Raises:
            NotImplementedError: Always raises this error
        """
        raise NotImplementedError(
            "Diffusion Policy requires sample sequences. "
            "Use preprocess_sample_sequence() with a list of samples instead."
        )
