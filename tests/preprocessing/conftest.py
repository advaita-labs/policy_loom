"""Shared fixtures for preprocessing tests."""

import numpy as np
import pytest

from loom.core.types import CameraImage, Sample
from loom.preprocessing.config import ImagePreprocessingConfig, SmolVLAPreprocessingConfig


@pytest.fixture
def simple_sample() -> Sample:
    """Create a simple sample for testing with synthetic data.

    Returns:
        Sample with 2 cameras, 7-dim proprio, 7-dim action, and task instruction
    """
    return Sample(
        timestamp=1000.0,
        cameras=[
            CameraImage(
                name="left_cam",
                image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            ),
            CameraImage(
                name="right_cam",
                image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            ),
        ],
        proprio=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        action=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        metadata={"task": "Pick up the red cube"},
    )


@pytest.fixture
def single_camera_sample() -> Sample:
    """Create a sample with single camera.

    Returns:
        Sample with 1 camera, 7-dim proprio, 7-dim action, and task instruction
    """
    return Sample(
        timestamp=1000.0,
        cameras=[
            CameraImage(
                name="observation.image",
                image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            ),
        ],
        proprio=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        action=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        metadata={"task": "Pick up the red cube"},
    )


@pytest.fixture
def sample_without_task() -> Sample:
    """Create a sample missing task instruction in metadata.

    Returns:
        Sample without task in metadata
    """
    return Sample(
        timestamp=1000.0,
        cameras=[
            CameraImage(
                name="left_cam",
                image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            ),
        ],
        proprio=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        action=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        metadata={},
    )


@pytest.fixture
def sample_without_camera() -> Sample:
    """Create a sample with no cameras.

    Returns:
        Sample with empty camera list
    """
    return Sample(
        timestamp=1000.0,
        cameras=[],
        proprio=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        action=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        metadata={"task": "Pick up the red cube"},
    )


@pytest.fixture
def basic_smolvla_config() -> SmolVLAPreprocessingConfig:
    """Create basic SmolVLA config for testing.

    Returns:
        Config with minimal settings for unit tests
    """
    return SmolVLAPreprocessingConfig(
        camera_names=["left_cam", "right_cam"],
        image_config=ImagePreprocessingConfig(
            target_size=(512, 512),
            resize_with_padding=True,
            normalize=True,
        ),
        n_obs_steps=1,
        max_state_dim=32,
        max_action_dim=32,
        chunk_size=50,
        state_mean=[0.0] * 7,
        state_std=[1.0] * 7,
        action_mean=[0.0] * 7,
        action_std=[1.0] * 7,
        device="cpu",
    )


@pytest.fixture
def single_camera_config() -> SmolVLAPreprocessingConfig:
    """Create SmolVLA config for single camera.

    Returns:
        Config with single camera
    """
    return SmolVLAPreprocessingConfig(
        camera_names=["observation.image"],
        image_config=ImagePreprocessingConfig(
            target_size=(512, 512),
            resize_with_padding=True,
            normalize=True,
        ),
        state_mean=[0.0] * 7,
        state_std=[1.0] * 7,
        action_mean=[0.0] * 7,
        action_std=[1.0] * 7,
        device="cpu",
    )
