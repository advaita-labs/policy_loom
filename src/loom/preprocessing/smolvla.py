"""SmolVLA model preprocessor."""

import logging

import numpy as np
import torch
from transformers import AutoTokenizer

from loom.core.ports import Preprocessor
from loom.core.types import Sample, SmolVLABatchInput, SmolVLAInput
from loom.preprocessing.base import BasePreprocessor
from loom.preprocessing.config import SmolVLAPreprocessingConfig

logger = logging.getLogger(__name__)


class SmolVLAPreprocessor(Preprocessor[SmolVLAInput, SmolVLABatchInput], BasePreprocessor):
    """Preprocessor for SmolVLA model.

    Converts policy_loom Sample objects to SmolVLA input format.

    Args:
        config: SmolVLA preprocessing configuration

    Example:
        >>> config = SmolVLAPreprocessingConfig(
        ...     camera_names=["left_cam", "right_cam"],
        ...     state_mean=[0.0] * 7,
        ...     state_std=[1.0] * 7,
        ... )
        >>> preprocessor = SmolVLAPreprocessor(config)
        >>>
        >>> # Single sample preprocessing
        >>> sample = Sample(...)
        >>> model_input = preprocessor.preprocess_sample(sample)
        >>>
        >>> # Use with DataLoader
        >>> dataset = SampleDataset(samples, preprocessor)
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     collate_fn=preprocessor.collate_fn
        ... )
    """

    def __init__(self, config: SmolVLAPreprocessingConfig) -> None:
        """Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration

        Raises:
            ValueError: If config is invalid
        """
        self.config = config

        # Validate configuration
        self._validate_config()

        # Load tokenizer
        logger.info(f"Loading tokenizer: {config.vlm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)

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

        # Check std is not zero
        if self.config.state_std is not None:
            if any(s == 0.0 for s in self.config.state_std):
                raise ValueError("state_std contains zero values")

        if self.config.action_std is not None:
            if any(s == 0.0 for s in self.config.action_std):
                raise ValueError("action_std contains zero values")

    def preprocess_sample(self, sample: Sample) -> SmolVLAInput:
        """Convert Sample to SmolVLA input format (unbatched).

        Args:
            sample: Input sample with cameras, proprio, action, metadata

        Returns:
            SmolVLAInput dataclass with preprocessed data (no batch dimension)

        Raises:
            ValueError: If required cameras are missing
            ValueError: If task instruction not in metadata
            ValueError: If proprio or action is None
            ValueError: If data contains NaN or inf
        """
        # Extract task instruction
        if "task" not in sample.metadata:
            raise ValueError("Sample metadata must contain 'task' field with instruction")
        language_instruction = sample.metadata["task"]

        # Extract and preprocess images (in config order)
        images = []
        for camera_name in self.config.camera_names:
            camera = sample.get_camera(camera_name)
            if camera is None:
                raise ValueError(
                    f"Camera '{camera_name}' not found in sample. "
                    f"Available cameras: {[c.name for c in sample.cameras]}"
                )
            preprocessed_img = self._preprocess_image(camera.image)
            images.append(preprocessed_img)

        # Preprocess state
        if sample.proprio is None:
            raise ValueError("Sample must have proprio (state) data")
        state = self._preprocess_state(sample.proprio)

        # Preprocess action
        if sample.action is None:
            raise ValueError("Sample must have action data")
        action = self._preprocess_action(sample.action)

        return SmolVLAInput(
            images=images,
            language_instruction=language_instruction,
            state=state,
            action=action,
        )

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image.

        Pipeline: uint8 [0,255] → float32 → /255 → [0,1] → (x-mean)/std → normalized

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
            Normalized and padded state tensor

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

        # Pad to max_state_dim
        padded = self.pad_vector(normalized, self.config.max_state_dim, pad_value=0.0)

        return torch.from_numpy(padded).float()

    def _preprocess_action(self, action: np.ndarray) -> torch.Tensor:
        """Preprocess action.

        Args:
            action: Input action vector

        Returns:
            Normalized and padded action tensor

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

        # Pad to max_action_dim
        padded = self.pad_vector(normalized, self.config.max_action_dim, pad_value=0.0)

        return torch.from_numpy(padded).float()

    def collate_fn(self, batch: list[SmolVLAInput]) -> SmolVLABatchInput:
        """Collate batch for SmolVLA model.

        Args:
            batch: List of SmolVLAInput from preprocess_sample()

        Returns:
            SmolVLABatchInput with batched tensors

        Raises:
            ValueError: If batch is empty
            ValueError: If batch contains inconsistent camera counts
        """
        if not batch:
            raise ValueError("Cannot collate empty batch")

        # Validate camera counts are consistent
        num_cameras = len(batch[0].images)
        for item in batch:
            if len(item.images) != num_cameras:
                raise ValueError(
                    f"Inconsistent camera counts in batch: expected {num_cameras}, " f"got {len(item.images)}"
                )

        # Stack images by camera
        observation_images = {}
        for cam_idx, camera_name in enumerate(self.config.camera_names):
            camera_tensors = [item.images[cam_idx] for item in batch]
            stacked = torch.stack(camera_tensors, dim=0)  # (B, C, H, W)
            observation_images[camera_name] = stacked

        # Tokenize language instructions
        instructions = [item.language_instruction for item in batch]
        tokenized = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.config.max_language_tokens,
            return_tensors="pt",
        )
        language_tokens = tokenized["input_ids"]
        language_attention_mask = tokenized["attention_mask"]

        # Stack states and reshape for n_obs_steps
        state_tensors = [item.state for item in batch]
        state_stacked = torch.stack(state_tensors, dim=0)  # (B, state_dim)
        # Reshape to (B, n_obs_steps, state_dim)
        state_reshaped = state_stacked.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1)

        # Stack actions and reshape
        action_tensors = [item.action for item in batch]
        action_stacked = torch.stack(action_tensors, dim=0)  # (B, action_dim)
        # Reshape to (B, 1, action_dim) for single actions
        action_reshaped = action_stacked.unsqueeze(1)

        # Move to device if specified
        device = torch.device(self.config.device)
        observation_images = {k: v.to(device) for k, v in observation_images.items()}
        language_tokens = language_tokens.to(device)
        language_attention_mask = language_attention_mask.to(device)
        state_reshaped = state_reshaped.to(device)
        action_reshaped = action_reshaped.to(device)

        return SmolVLABatchInput(
            observation_images=observation_images,
            language_tokens=language_tokens,
            language_attention_mask=language_attention_mask,
            state=state_reshaped,
            action=action_reshaped,
        )
