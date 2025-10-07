"""LeRobot dataset loader with lazy imports for dependency isolation."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from loom.core.types import CameraImage, Sample

logger = logging.getLogger(__name__)


def collate_lerobot_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for LeRobot batches that handles None values.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch dictionary
    """
    # Extract keys from first sample
    keys = batch[0].keys()
    collated: dict[str, Any] = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        # Handle None values - keep as list if all None, skip if mixed
        if all(v is None for v in values):
            collated[key] = None
        elif any(v is None for v in values):
            # Mixed None/non-None - convert None to zero tensors
            non_none_shape = next(v.shape for v in values if v is not None)
            values = [v if v is not None else np.zeros(non_none_shape, dtype=np.float32) for v in values]
            collated[key] = torch.from_numpy(np.stack(values))
        elif key == "cameras":
            # List of camera lists - keep as nested list
            collated[key] = values
        elif key == "metadata":
            # List of metadata dicts - keep as list
            collated[key] = values
        elif key == "images":
            # Dict of camera images - keep as nested dict
            # Each value is a list of dicts: [{cam: img_array}, ...]
            collated[key] = values
        elif isinstance(values[0], np.ndarray):
            # Stack numpy arrays into tensor
            collated[key] = torch.from_numpy(np.stack(values))
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors
            collated[key] = torch.stack(values)
        elif isinstance(values[0], int | float):
            # Convert scalar lists to tensor
            collated[key] = torch.tensor(values)
        else:
            # Keep as list for other types
            collated[key] = values

    return collated


class LeRobotDatasetLoader:
    """Load LeRobot datasets from HuggingFace Hub.

    Uses lazy imports to avoid requiring lerobot dependencies unless actually used.
    This allows policy_loom to work without LeRobot installed.

    Example:
        >>> # Load from HuggingFace Hub
        >>> loader = LeRobotDatasetLoader("lerobot/koch_test", split="train")
        >>> dataset = loader.to_torch_dataset()
        >>>
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "lerobot/koch_test")
        split: Dataset split ("train" or "test")
        local_dir: Optional local directory to cache dataset
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        local_dir: Path | None = None,
    ):
        """Initialize loader and download dataset if needed.

        Args:
            repo_id: HuggingFace Hub dataset repository ID
            split: Dataset split to load
            local_dir: Optional directory for caching dataset locally

        Raises:
            ImportError: If lerobot or datasets library is not installed
            ValueError: If repo_id is invalid or dataset not found
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "LeRobot dataset loading requires 'lerobot' and 'datasets' packages. "
                "Install with: uv sync --extra pi05"
            ) from e

        self.repo_id = repo_id
        self.split = split
        self.local_dir = local_dir

        logger.info(f"Loading LeRobot dataset: {repo_id} (split={split})")

        # Load dataset from HuggingFace Hub
        try:
            self.dataset = load_dataset(
                repo_id,
                split=split,
                cache_dir=str(local_dir) if local_dir else None,
            )
            logger.info(f"Loaded {len(self.dataset)} samples from {repo_id}")
        except Exception as e:
            raise ValueError(f"Failed to load dataset {repo_id}: {e}") from e

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.dataset)

    def to_samples(self) -> Iterator[Sample]:
        """Convert LeRobot dataset to policy_loom Sample format.

        Yields:
            Sample objects with cameras, proprio, action, and metadata

        Note:
            LeRobot format:
                - observation.images.{camera_name}: RGB images (H, W, C)
                - observation.state: Proprioceptive state
                - action: Robot action
                - episode_index, frame_index, timestamp
        """
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]

            # Extract cameras
            cameras = []
            observation = item.get("observation", {})
            if "images" in observation:
                for cam_name, img_data in observation["images"].items():
                    # Convert to numpy if needed
                    if isinstance(img_data, torch.Tensor):
                        img_np = img_data.cpu().numpy()
                    else:
                        img_np = np.array(img_data)

                    # Ensure correct shape (H, W, C)
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC

                    cameras.append(CameraImage(name=cam_name, image=img_np.astype(np.uint8)))

            # Extract proprioceptive state
            proprio = None
            if "state" in observation:
                state_data = observation["state"]
                if isinstance(state_data, torch.Tensor):
                    proprio = state_data.cpu().numpy().astype(np.float32)
                else:
                    proprio = np.array(state_data, dtype=np.float32)

            # Extract action
            action = None
            if "action" in item:
                action_data = item["action"]
                if isinstance(action_data, torch.Tensor):
                    action = action_data.cpu().numpy().astype(np.float32)
                else:
                    action = np.array(action_data, dtype=np.float32)

            # Extract metadata
            metadata = {
                "episode_index": item.get("episode_index", -1),
                "frame_index": item.get("frame_index", idx),
                "timestamp": item.get("timestamp", idx),
                "repo_id": self.repo_id,
                "split": self.split,
            }

            # Create Sample
            yield Sample(
                timestamp=float(item.get("timestamp", idx)),
                cameras=cameras,
                proprio=proprio,
                action=action,
                metadata=metadata,
            )

    def to_torch_dataset(self) -> "LeRobotTorchDataset":
        """Convert to PyTorch Dataset.

        Returns:
            PyTorch Dataset that can be used with DataLoader

        Example:
            >>> dataset = loader.to_torch_dataset()
            >>> dataloader = DataLoader(dataset, batch_size=32)
        """
        return LeRobotTorchDataset(self.dataset, self.repo_id, self.split)


class LeRobotTorchDataset(Dataset):
    """PyTorch Dataset wrapper for LeRobot data.

    Args:
        dataset: HuggingFace dataset object
        repo_id: Repository ID for metadata
        split: Dataset split for metadata
    """

    def __init__(self, dataset: Any, repo_id: str, split: str):
        """Initialize PyTorch dataset wrapper."""
        self.dataset = dataset
        self.repo_id = repo_id
        self.split = split

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get single sample.

        Args:
            idx: Sample index

        Returns:
            Dict with keys: observation, action, images, metadata
        """
        item = self.dataset[idx]

        # Handle two dataset formats:
        # 1. Nested: item["observation"] = {"state": [...], "images": {...}}
        # 2. Flat: item["observation.state"] = [...]

        # Extract proprioceptive state (observation)
        observation = None
        if "observation.state" in item:
            # Flat key format
            state_data = item["observation.state"]
            if isinstance(state_data, torch.Tensor):
                observation = state_data.cpu().numpy().astype(np.float32)
            else:
                observation = np.array(state_data, dtype=np.float32)
        elif "observation" in item and isinstance(item["observation"], dict):
            # Nested dict format
            obs_dict = item["observation"]
            if "state" in obs_dict:
                state_data = obs_dict["state"]
                if isinstance(state_data, torch.Tensor):
                    observation = state_data.cpu().numpy().astype(np.float32)
                else:
                    observation = np.array(state_data, dtype=np.float32)

        # Extract images
        images = self._extract_images(item)

        # Extract action
        action = None
        if "action" in item:
            action_data = item["action"]
            if isinstance(action_data, torch.Tensor):
                action = action_data.cpu().numpy().astype(np.float32)
            else:
                action = np.array(action_data, dtype=np.float32)

        # Extract metadata
        metadata = {
            "episode_index": item.get("episode_index", -1),
            "frame_index": item.get("frame_index", idx),
            "timestamp": item.get("timestamp", idx),
            "repo_id": self.repo_id,
            "split": self.split,
        }

        # Return dict format for DataLoader compatibility
        return {
            "observation": observation,
            "action": action,
            "images": images,
            "metadata": metadata,
        }

    def _extract_images(self, item: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract images from LeRobot format.

        Args:
            item: Single dataset item

        Returns:
            Dict mapping camera names to image arrays (H, W, C) uint8
        """
        images = {}

        # Handle flat format: observation.images.cam_name
        for key in item.keys():
            if key.startswith("observation.images."):
                cam_name = key.split(".")[-1]
                img_data = item[key]
                images[cam_name] = self._to_numpy_image(img_data)

        # Handle nested format: observation -> images -> cam_name
        if "observation" in item and isinstance(item["observation"], dict):
            obs_dict = item["observation"]
            if "images" in obs_dict and isinstance(obs_dict["images"], dict):
                for cam_name, img_data in obs_dict["images"].items():
                    images[cam_name] = self._to_numpy_image(img_data)

        return images

    def _to_numpy_image(self, img_data: Any) -> np.ndarray:
        """Convert image data to numpy array (H, W, C) uint8.

        Args:
            img_data: Image data (tensor, PIL Image, or numpy array)

        Returns:
            Numpy array of shape (H, W, C) with dtype uint8
        """
        if isinstance(img_data, torch.Tensor):
            img_np = img_data.cpu().numpy()
        else:
            img_np = np.array(img_data)

        # Ensure uint8 dtype
        if img_np.dtype != np.uint8:
            # Assume float [0, 1] -> scale to [0, 255]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        # Ensure (H, W, C) format
        if img_np.ndim == 3 and img_np.shape[0] == 3:
            # CHW -> HWC
            img_np = np.transpose(img_np, (1, 2, 0))

        return img_np
