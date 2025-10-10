"""LeRobot dataset writer for converting policy_loom Samples to LeRobot format.

This module provides a bridge between policy_loom's Sample objects and LeRobot's
dataset format, enabling:
- Conversion of MP4/MCAP data to standardized LeRobot format
- Dataset versioning and sharing via HuggingFace Hub
- Reusability across different VLA models

Reference:
- LeRobot dataset format: https://huggingface.co/docs/lerobot
- OpenPI conversion examples: openpi/examples/*/convert_*_to_lerobot.py
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from loom.core.types import Sample

logger = logging.getLogger(__name__)


class LeRobotDatasetWriter:
    """Convert policy_loom Sample streams to LeRobot dataset format.

    This writer bridges policy_loom's data pipeline with LeRobot's standardized
    dataset format, following the pattern used by OpenPI.

    Example:
        >>> from loom.io.synchronized import SynchronizedVideoMCAPReader
        >>> from loom.pipeline import merge_streams
        >>> from loom.io.lerobot import LeRobotDatasetWriter
        >>>
        >>> # Create readers
        >>> left_cam = SynchronizedVideoMCAPReader(...)
        >>> right_cam = SynchronizedVideoMCAPReader(...)
        >>>
        >>> # Merge streams
        >>> samples = list(merge_streams(left_cam, right_cam))
        >>>
        >>> # Write to LeRobot format
        >>> writer = LeRobotDatasetWriter(
        ...     repo_id="username/my_robot",
        ...     robot_type="franka",
        ...     fps=30,
        ...     camera_names=["left_cam", "right_cam"],
        ...     action_dim=7,
        ...     proprio_dim=7,
        ... )
        >>> writer.add_episode(samples, task="pick_cube")
        >>> writer.consolidate(push_to_hub=True)
    """

    def __init__(
        self,
        repo_id: str,
        robot_type: str,
        fps: int,
        camera_names: list[str],
        action_dim: int,
        proprio_dim: int,
        root: str | Path | None = None,
        use_videos: bool = True,
        image_writer_threads: int = 4,
    ):
        """Initialize LeRobot dataset writer.

        Args:
            repo_id: HuggingFace dataset repo ID (e.g., "username/my_robot_data")
            robot_type: Robot identifier (e.g., "franka", "ur5", "aloha")
            fps: Frames per second for the dataset
            camera_names: List of camera identifiers matching Sample.cameras[].name
            action_dim: Action space dimensionality
            proprio_dim: Proprioception dimensionality (0 if no proprio)
            root: Local directory to save dataset (default: ~/.cache/lerobot)
            use_videos: Store as video files (True) or individual images (False)
            image_writer_threads: Number of threads for image writing

        Raises:
            ImportError: If lerobot package is not installed
        """
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as e:
            raise ImportError(
                "LeRobot package required for dataset writing. "
                "Install with: pip install lerobot"
            ) from e

        self.repo_id = repo_id
        self.robot_type = robot_type
        self.fps = fps
        self.camera_names = camera_names
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.use_videos = use_videos

        # Create LeRobot features specification
        features = self._create_features()

        logger.info(f"Creating LeRobot dataset: {repo_id}")
        logger.info(f"  Robot: {robot_type}, FPS: {fps}")
        logger.info(f"  Cameras: {camera_names}")
        logger.info(f"  Action dim: {action_dim}, Proprio dim: {proprio_dim}")
        logger.info(f"  Use videos: {use_videos}")

        # Create LeRobot dataset
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
        )

        self._num_episodes = 0

    def _create_features(self) -> dict[str, dict]:
        """Create LeRobot feature specification from configuration.

        Returns:
            Dictionary mapping feature names to their specifications

        Example output:
            {
                "observation.images.left_cam": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["joint_0", "joint_1", ...],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action_0", "action_1", ...],
                },
            }
        """
        features = {}

        # Camera features
        for cam_name in self.camera_names:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": (480, 640, 3),  # Default, will be determined from first frame
                "names": ["height", "width", "channels"],
            }

        # Proprioception feature
        if self.proprio_dim > 0:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (self.proprio_dim,),
                "names": [f"joint_{i}" for i in range(self.proprio_dim)],
            }

        # Action feature
        features["action"] = {
            "dtype": "float32",
            "shape": (self.action_dim,),
            "names": [f"action_{i}" for i in range(self.action_dim)],
        }

        return features

    def add_episode(
        self,
        samples: list[Sample],
        task: str = "default_task",
    ) -> None:
        """Add an episode from a list of Sample objects.

        Args:
            samples: List of temporally aligned Sample objects from merge_streams()
            task: Task description/name (used as language instruction for VLA models)

        Raises:
            ValueError: If sample validation fails (wrong camera names, action dims, etc.)
        """
        if not samples:
            raise ValueError("Cannot add empty episode")

        logger.info(f"Adding episode: {len(samples)} frames, task='{task}'")

        for idx, sample in enumerate(samples):
            # Validate and convert sample to LeRobot frame format
            frame = self._sample_to_frame(sample, idx)
            frame["task"] = task

            # Add frame to dataset
            self.dataset.add_frame(frame)

        # Save episode
        self.dataset.save_episode()
        self._num_episodes += 1

        logger.info(f"Episode added successfully (total episodes: {self._num_episodes})")

    def _sample_to_frame(self, sample: Sample, frame_idx: int) -> dict[str, Any]:
        """Convert Sample object to LeRobot frame format.

        Args:
            sample: Input Sample object
            frame_idx: Frame index within episode

        Returns:
            Dictionary in LeRobot frame format

        Raises:
            ValueError: If validation fails
        """
        frame = {}

        # Add camera images
        for camera in sample.cameras:
            if camera.name not in self.camera_names:
                raise ValueError(
                    f"Camera '{camera.name}' not in expected cameras: {self.camera_names}"
                )

            key = f"observation.images.{camera.name}"

            # LeRobot expects (H, W, C) format
            image = camera.image
            if image.ndim != 3:
                raise ValueError(f"Expected 3D image (H, W, C), got shape {image.shape}")

            frame[key] = image

        # Add proprioception
        if self.proprio_dim > 0:
            if sample.proprio is None:
                raise ValueError(f"Expected proprioception with dim {self.proprio_dim}, got None")

            if sample.proprio.shape[0] != self.proprio_dim:
                raise ValueError(
                    f"Proprioception dimension mismatch: expected {self.proprio_dim}, "
                    f"got {sample.proprio.shape[0]}"
                )

            frame["observation.state"] = sample.proprio

        # Add action
        if sample.action is None:
            raise ValueError("Sample missing action")

        if sample.action.shape[0] != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: expected {self.action_dim}, " f"got {sample.action.shape[0]}"
            )

        frame["action"] = sample.action

        return frame

    def consolidate(self, push_to_hub: bool = False) -> None:
        """Consolidate dataset and optionally push to HuggingFace Hub.

        This finalizes the dataset by:
        - Computing dataset statistics (mean/std for normalization)
        - Creating metadata files
        - Optionally uploading to HuggingFace Hub

        Args:
            push_to_hub: If True, push dataset to HuggingFace Hub

        Raises:
            RuntimeError: If consolidation fails
        """
        logger.info("Consolidating dataset...")

        try:
            self.dataset.consolidate()
            logger.info("Dataset consolidated successfully")

            if push_to_hub:
                logger.info(f"Pushing to HuggingFace Hub: {self.repo_id}")
                self.dataset.push_to_hub()
                logger.info("Dataset pushed successfully")
        except Exception as e:
            logger.error(f"Failed to consolidate dataset: {e}")
            raise RuntimeError(f"Dataset consolidation failed: {e}") from e

    @property
    def num_episodes(self) -> int:
        """Return number of episodes added to dataset."""
        return self._num_episodes

    def __len__(self) -> int:
        """Return total number of frames in dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Access dataset sample by index.

        Args:
            idx: Frame index

        Returns:
            Dictionary with LeRobot format:
                - "observation.images.{camera_name}": image array
                - "observation.state": proprioception array
                - "action": action array
                - "task": task string
        """
        return self.dataset[idx]
