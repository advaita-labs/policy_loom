"""Tests for core types and ports."""

import numpy as np
import pytest

from loom.core import CameraImage, Sample


class TestCameraImage:
    """Tests for CameraImage dataclass."""

    def test_camera_image_creation(self) -> None:
        """Test creating a CameraImage."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        camera = CameraImage(name="left_cam", image=image)

        assert camera.name == "left_cam"
        assert camera.image.shape == (224, 224, 3)

    def test_camera_image_invalid_shape(self) -> None:
        """Test that invalid image shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected image to have 3 dimensions"):
            CameraImage(name="test", image=np.zeros((224, 224)))

    def test_camera_image_float32(self) -> None:
        """Test CameraImage with float32 dtype."""
        image = np.zeros((224, 224, 3), dtype=np.float32)
        camera = CameraImage(name="test_cam", image=image)

        assert camera.image.dtype == np.float32


class TestSample:
    """Tests for Sample dataclass."""

    def test_sample_creation_minimal(self) -> None:
        """Test creating a Sample with only timestamp."""
        sample = Sample(timestamp=1.0)
        assert sample.timestamp == 1.0
        assert sample.cameras == []
        assert sample.proprio is None
        assert sample.action is None
        assert sample.metadata == {}

    def test_sample_creation_with_single_camera(self) -> None:
        """Test creating a Sample with a single camera."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        camera = CameraImage(name="left_cam", image=image)

        sample = Sample(timestamp=1.0, cameras=[camera])

        assert sample.timestamp == 1.0
        assert len(sample.cameras) == 1
        assert sample.cameras[0].name == "left_cam"
        assert sample.cameras[0].image.shape == (224, 224, 3)

    def test_sample_creation_with_multiple_cameras(self) -> None:
        """Test creating a Sample with multiple cameras."""
        left_img = np.zeros((224, 224, 3), dtype=np.uint8)
        right_img = np.zeros((224, 224, 3), dtype=np.uint8)

        sample = Sample(
            timestamp=1.0,
            cameras=[
                CameraImage(name="left_cam", image=left_img),
                CameraImage(name="right_cam", image=right_img),
            ],
        )

        assert len(sample.cameras) == 2
        assert sample.cameras[0].name == "left_cam"
        assert sample.cameras[1].name == "right_cam"

    def test_sample_creation_full(self) -> None:
        """Test creating a Sample with all fields."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        camera = CameraImage(name="test_cam", image=image)
        proprio = np.zeros(6, dtype=np.float32)
        action = np.zeros(7, dtype=np.float32)
        metadata = {"episode_id": "test"}

        sample = Sample(
            timestamp=1.0,
            cameras=[camera],
            proprio=proprio,
            action=action,
            metadata=metadata,
        )

        assert sample.timestamp == 1.0
        assert len(sample.cameras) == 1
        assert sample.cameras[0].image.shape == (224, 224, 3)
        assert sample.proprio.shape == (6,)
        assert sample.action.shape == (7,)
        assert sample.metadata["episode_id"] == "test"

    def test_sample_get_camera(self) -> None:
        """Test getting a camera by name."""
        left_img = np.zeros((224, 224, 3), dtype=np.uint8)
        right_img = np.zeros((224, 224, 3), dtype=np.uint8)

        sample = Sample(
            timestamp=1.0,
            cameras=[
                CameraImage(name="left_cam", image=left_img),
                CameraImage(name="right_cam", image=right_img),
            ],
        )

        left_cam = sample.get_camera("left_cam")
        assert left_cam is not None
        assert left_cam.name == "left_cam"

        missing_cam = sample.get_camera("nonexistent")
        assert missing_cam is None

    def test_sample_add_camera(self) -> None:
        """Test adding a camera to a sample."""
        sample = Sample(timestamp=1.0)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        sample.add_camera("test_cam", image)

        assert len(sample.cameras) == 1
        assert sample.cameras[0].name == "test_cam"

    def test_sample_add_camera_duplicate_name(self) -> None:
        """Test that adding a camera with duplicate name raises ValueError."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        sample = Sample(timestamp=1.0, cameras=[CameraImage(name="test_cam", image=image)])

        with pytest.raises(ValueError, match="Camera 'test_cam' already exists"):
            sample.add_camera("test_cam", image)

    def test_sample_invalid_proprio_shape(self) -> None:
        """Test that invalid proprio shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected proprio to be 1D"):
            Sample(timestamp=1.0, proprio=np.zeros((6, 1)))

    def test_sample_invalid_action_shape(self) -> None:
        """Test that invalid action shape raises ValueError."""
        with pytest.raises(ValueError, match="Expected action to be 1D"):
            Sample(timestamp=1.0, action=np.zeros((7, 1)))

    def test_sample_timestamp_types(self) -> None:
        """Test that both float and int timestamps work."""
        sample_float = Sample(timestamp=1.5)
        assert isinstance(sample_float.timestamp, float)

        sample_int = Sample(timestamp=1234567890)
        assert isinstance(sample_int.timestamp, int)

    def test_sample_empty_cameras_list(self) -> None:
        """Test that empty cameras list is valid."""
        sample = Sample(timestamp=1.0, cameras=[])
        assert sample.cameras == []
