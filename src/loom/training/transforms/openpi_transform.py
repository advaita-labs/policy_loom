"""Data transformation from LeRobot format to OpenPI format.

This module provides transformations to convert LeRobot dataset batches
into the format expected by Physical Intelligence's OpenPI models.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class OpenPITransform:
    """Transform LeRobot batches to OpenPI Observation format.

    OpenPI expects:
        - images: dict[str, float32] normalized to [-1, 1], shape (B, H, W, 3)
        - image_masks: dict[str, bool], shape (B,)
        - state: float32, shape (B, state_dim)
        - tokenized_prompt: int32, shape (B, seq_len) [optional]
        - tokenized_prompt_mask: bool, shape (B, seq_len) [optional]

    LeRobot provides:
        - observation: float32, shape (B, state_dim)
        - images: list[dict[str, np.ndarray]], each uint8[0,255]
        - action: float32, shape (B, action_dim)
        - metadata: list[dict]

    Args:
        tokenizer: Optional tokenizer for text prompts
        image_size: Target image size (height, width)
        default_prompt: Optional default prompt to use if none provided
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        image_size: tuple[int, int] = (224, 224),
        default_prompt: str | None = None,
    ):
        """Initialize OpenPI transform.

        Args:
            tokenizer: Optional tokenizer for prompts
            image_size: Target size for images (H, W)
            default_prompt: Default prompt if none provided in batch
        """
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.default_prompt = default_prompt

    def __call__(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Transform LeRobot batch to OpenPI format.

        Args:
            batch: LeRobot batch dict with keys:
                - observation: (B, state_dim) or None
                - images: list of dict[str, np.ndarray] or None
                - action: (B, action_dim)
                - metadata: list of dicts [optional]
                - prompt: list of strings [optional]

        Returns:
            Tuple of (observations_dict, actions_tensor)
                - observations_dict: OpenPI format observation dict
                - actions_tensor: Action tensor (B, action_dim)

        Raises:
            ValueError: If batch is missing required fields
        """
        # Initialize observation dict
        obs_dict: dict[str, torch.Tensor] = {}

        # 1. Process images
        if "images" in batch and batch["images"] is not None:
            obs_dict.update(self._process_images(batch["images"]))

        # 2. Process proprioceptive state
        if "observation" in batch and batch["observation"] is not None:
            obs_dict["state"] = self._process_state(batch["observation"])

        # 3. Process prompts (if provided)
        if "prompt" in batch:
            prompt_tokens, prompt_mask = self._process_prompts(batch["prompt"])
            if prompt_tokens is not None:
                obs_dict["tokenized_prompt"] = prompt_tokens
                obs_dict["tokenized_prompt_mask"] = prompt_mask
        elif self.default_prompt and self.tokenizer:
            # Use default prompt
            batch_size = len(batch["action"])
            prompts = [self.default_prompt] * batch_size
            prompt_tokens, prompt_mask = self._process_prompts(prompts)
            obs_dict["tokenized_prompt"] = prompt_tokens
            obs_dict["tokenized_prompt_mask"] = prompt_mask

        # 4. Extract actions
        if "action" not in batch:
            raise ValueError("Batch must contain 'action' field")

        actions = batch["action"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(np.array(actions, dtype=np.float32))

        return obs_dict, actions

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image from uint8[0,255] to float32[-1,1].

        OpenPI expects images normalized to [-1, 1] where:
            - 0 (black) → -1.0
            - 127.5 (mid-gray) → 0.0
            - 255 (white) → 1.0

        Args:
            img: uint8 array in range [0, 255], shape (H, W, 3)

        Returns:
            float32 array in range [-1, 1], shape (H, W, 3)

        Raises:
            ValueError: If image dtype is not uint8 or shape is invalid
        """
        if img.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 image for normalization, got {img.dtype}. "
                f"Images must be in range [0, 255] with dtype uint8."
            )

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {img.shape}. " f"Only RGB images are supported.")

        # Normalize to [-1, 1]
        return (img.astype(np.float32) / 127.5) - 1.0

    def _process_images(self, images_list: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        """Process images from LeRobot format to OpenPI format.

        Args:
            images_list: List of dicts mapping camera names to uint8 arrays (H, W, 3)

        Returns:
            Dict with 'images' and 'image_masks' keys

        Raises:
            ValueError: If image format is invalid
        """
        # Get all camera names from first sample
        if not images_list or not images_list[0]:
            return {}

        camera_names = list(images_list[0].keys())
        batch_size = len(images_list)

        images_dict = {}
        masks_dict = {}

        for cam_name in camera_names:
            # Stack images for this camera
            cam_images = []
            for sample_imgs in images_list:
                if cam_name in sample_imgs:
                    img = sample_imgs[cam_name]
                    # Validate and normalize uint8[0,255] -> float32[-1,1]
                    img_float = self._normalize_image(img)
                    cam_images.append(img_float)
                else:
                    # Missing image - create black image (normalized to -1.0)
                    img_float = np.full((*self.image_size, 3), -1.0, dtype=np.float32)
                    cam_images.append(img_float)

            # Stack to (B, H, W, 3)
            images_tensor = torch.from_numpy(np.stack(cam_images, axis=0))
            images_dict[cam_name] = images_tensor

            # Create masks (all True for real images)
            masks_dict[cam_name] = torch.ones(batch_size, dtype=torch.bool)

        return {"images": images_dict, "image_masks": masks_dict}

    def _process_state(self, observation: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Process proprioceptive state.

        Args:
            observation: State tensor or array (B, state_dim)

        Returns:
            State tensor as float32
        """
        if not isinstance(observation, torch.Tensor):
            observation = torch.from_numpy(np.array(observation, dtype=np.float32))

        return observation.float()

    def _process_prompts(self, prompts: list[str]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Process text prompts using tokenizer.

        Args:
            prompts: List of prompt strings

        Returns:
            Tuple of (tokenized_prompts, prompt_masks) or (None, None) if no tokenizer
        """
        if self.tokenizer is None:
            return None, None

        # Tokenize prompts
        try:
            tokenized = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            tokens = tokenized["input_ids"]
            mask = tokenized["attention_mask"].bool()

            return tokens, mask

        except Exception as e:
            logger.warning(f"Failed to tokenize prompts: {e}")
            return None, None


def create_openpi_collate_fn(
    tokenizer: Any | None = None,
    image_size: tuple[int, int] = (224, 224),
    default_prompt: str | None = None,
) -> Callable[[list[dict]], tuple[dict, torch.Tensor]]:
    """Create a collate function that transforms LeRobot batches to OpenPI format.

    This is a convenience function for creating a DataLoader collate_fn.

    Args:
        tokenizer: Optional tokenizer for prompts
        image_size: Target image size (H, W)
        default_prompt: Default prompt if none provided

    Returns:
        Collate function suitable for DataLoader

    Example:
        >>> from torch.utils.data import DataLoader
        >>> collate_fn = create_openpi_collate_fn()
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    from loom.io.lerobot import collate_lerobot_batch

    transform = OpenPITransform(tokenizer=tokenizer, image_size=image_size, default_prompt=default_prompt)

    def collate_and_transform(batch_list: list[dict]) -> tuple[dict, torch.Tensor]:
        """Collate batch and transform to OpenPI format."""
        # First collate using standard LeRobot collate
        batch = collate_lerobot_batch(batch_list)

        # Then transform to OpenPI format
        obs_dict, actions = transform(batch)

        return obs_dict, actions

    return collate_and_transform
