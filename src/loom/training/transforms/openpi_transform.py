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
        camera_name_mapping: dict[str, str] | None = None,
    ):
        """Initialize OpenPI transform.

        Args:
            tokenizer: Optional tokenizer for prompts
            image_size: Target size for images (H, W)
            default_prompt: Default prompt if none provided in batch
            camera_name_mapping: Optional mapping from dataset camera names to OpenPI expected names
        """
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.default_prompt = default_prompt
        self.camera_name_mapping = camera_name_mapping or {}

    def __call__(self, batch: dict[str, Any]) -> tuple[Any, torch.Tensor]:
        """Transform LeRobot batch to OpenPI format.

        Args:
            batch: LeRobot batch dict with keys:
                - observation: (B, state_dim) or None
                - state: (B, state_dim) or None (alternative key)
                - images: list of dict[str, np.ndarray] or None
                - action: (B, action_dim)
                - metadata: list of dicts [optional]
                - prompt: list of strings [optional]

        Returns:
            Tuple of (observation, actions_tensor)
                - observation: OpenPI Observation object
                - actions_tensor: Action tensor (B, action_dim)

        Raises:
            ValueError: If batch is missing required fields
        """
        # Validate and prepare actions early
        if "action" not in batch:
            raise ValueError("Batch must contain 'action' field")

        actions = batch["action"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(np.array(actions, dtype=np.float32))

        # Initialize observation dict
        obs_data: dict[str, torch.Tensor] = {}

        batch_size = len(actions)

        # 1. Process images
        if "images" in batch and batch["images"] is not None:
            obs_data.update(self._process_images(batch["images"]))
        else:
            # Observation.from_dict requires 'image' and 'image_mask' keys
            # Provide empty dummies if no images
            obs_data["image"] = {}
            obs_data["image_mask"] = {}

        # 2. Process proprioceptive state
        # Try both 'observation' and 'state' keys
        state_data = batch.get("observation") if "observation" in batch else batch.get("state")
        if state_data is not None:
            obs_data["state"] = self._process_state(state_data)
        else:
            # Observation.from_dict requires 'state' key
            # Provide dummy state if none present
            obs_data["state"] = torch.zeros((batch_size, 1), dtype=torch.float32)

        # 3. Process prompts (if provided)
        if "prompt" in batch:
            prompt_tokens, prompt_mask = self._process_prompts(batch["prompt"])
            if prompt_tokens is not None:
                obs_data["tokenized_prompt"] = prompt_tokens
                obs_data["tokenized_prompt_mask"] = prompt_mask
        elif self.default_prompt and self.tokenizer:
            # Use default prompt
            prompts = [self.default_prompt] * batch_size
            prompt_tokens, prompt_mask = self._process_prompts(prompts)
            obs_data["tokenized_prompt"] = prompt_tokens
            obs_data["tokenized_prompt_mask"] = prompt_mask
        else:
            # No prompts - provide empty/minimal tokens
            # OpenPI requires tokenized_prompt and tokenized_prompt_mask together
            # Create minimal token sequence (e.g., just BOS/EOS tokens - token_id=1 is common for BOS)
            obs_data["tokenized_prompt"] = torch.ones((batch_size, 1), dtype=torch.int32)
            obs_data["tokenized_prompt_mask"] = torch.ones((batch_size, 1), dtype=torch.bool)

        # 4. Actions already validated above

        try:
            from openpi.models.model import Observation
        except ImportError:  # pragma: no cover
            logger.warning("OpenPI not installed; returning raw observation dictionary.")
            return obs_data, actions

        if actions.shape[-1] < 32:
            padding = torch.zeros((*actions.shape[:-1], 32 - actions.shape[-1]), dtype=actions.dtype)
            actions = torch.cat([actions, padding], dim=-1)
        elif actions.shape[-1] > 32:
            raise ValueError(
                f"Action dimension {actions.shape[-1]} exceeds OpenPI's hardcoded action_dim=32. "
                f"Cannot truncate actions. Please check your dataset."
            )

        observation = Observation.from_dict(obs_data)
        return observation, actions

    def _to_numpy_uint8(self, img: np.ndarray | Any) -> np.ndarray:
        """Convert image to numpy uint8[0,255] format.

        Args:
            img: Image as numpy array or PIL Image, shape (H, W, 3)

        Returns:
            uint8 numpy array in range [0, 255], shape (H, W, 3)

        Raises:
            ValueError: If image shape is invalid
        """
        # Convert PIL Image to numpy if needed
        if not isinstance(img, np.ndarray):
            from PIL import Image
            if isinstance(img, Image.Image):
                img = np.array(img)
            else:
                img = np.array(img)

        # Ensure uint8
        if img.dtype != np.uint8:
            # If float in [0, 1], scale to [0, 255]
            if img.dtype in [np.float32, np.float64] and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {img.shape}. " f"Only RGB images are supported.")

        return img

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
            # Apply camera name mapping if provided
            output_cam_name = self.camera_name_mapping.get(cam_name, cam_name)
            # Stack images for this camera
            cam_images = []
            for sample_imgs in images_list:
                if cam_name in sample_imgs:
                    img = sample_imgs[cam_name]
                    # Convert PIL/other to numpy uint8 if needed
                    img_uint8 = self._to_numpy_uint8(img)
                    cam_images.append(img_uint8)
                else:
                    # Missing image - create black image as uint8
                    img_uint8 = np.zeros((*self.image_size, 3), dtype=np.uint8)
                    cam_images.append(img_uint8)

            # Stack to (B, H, W, 3)
            images_tensor = torch.from_numpy(np.stack(cam_images, axis=0)).float()
            images_tensor = (images_tensor / 255.0) * 2.0 - 1.0
            images_dict[output_cam_name] = images_tensor

            # Create masks (all True for real images)
            masks_dict[output_cam_name] = torch.ones(batch_size, dtype=torch.bool)

        return {
            "image": images_dict,
            "image_mask": masks_dict,
            "images": images_dict,
            "image_masks": masks_dict,
        }

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
            # PaligemmaTokenizer uses .tokenize() method, not __call__
            # Process each prompt and stack results
            all_tokens = []
            all_masks = []
            
            for prompt in prompts:
                tokens, mask = self.tokenizer.tokenize(prompt)
                all_tokens.append(torch.from_numpy(tokens))
                all_masks.append(torch.from_numpy(mask))
            
            # Stack into batch tensors
            tokens = torch.stack(all_tokens)
            mask = torch.stack(all_masks).bool()

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
