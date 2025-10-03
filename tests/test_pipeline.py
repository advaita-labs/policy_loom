"""Tests for pipeline merging."""

import numpy as np
import pytest

from loom.core import CameraImage, Reader, Sample
from loom.pipeline import merge_streams


class MockReader(Reader):
    """Mock reader for testing."""

    def __init__(self, samples: list[Sample]) -> None:
        """Initialize with list of samples to yield."""
        self.samples = samples

    def read(self) -> list[Sample]:
        """Yield samples."""
        yield from self.samples


class TestMergeStreams:
    """Tests for merge_streams function."""

    def test_merge_streams_single_reader(self) -> None:
        """Test merging with a single reader."""
        samples = [
            Sample(timestamp=1.0, proprio=np.array([1.0, 2.0], dtype=np.float32)),
            Sample(timestamp=2.0, proprio=np.array([3.0, 4.0], dtype=np.float32)),
        ]

        reader = MockReader(samples)
        merged = list(merge_streams(reader))

        assert len(merged) == 2
        assert merged[0].timestamp == 1.0
        assert merged[1].timestamp == 2.0

    def test_merge_streams_empty(self) -> None:
        """Test merging with no readers."""
        merged = list(merge_streams())
        assert merged == []

    def test_merge_streams_merge_cameras(self) -> None:
        """Test merging multiple camera streams."""
        left_img = np.zeros((480, 640, 3), dtype=np.uint8)
        right_img = np.ones((480, 640, 3), dtype=np.uint8)

        reader1 = MockReader(
            [
                Sample(timestamp=1.0, cameras=[CameraImage(name="left_cam", image=left_img)]),
                Sample(timestamp=2.0, cameras=[CameraImage(name="left_cam", image=left_img)]),
            ]
        )

        reader2 = MockReader(
            [
                Sample(timestamp=1.01, cameras=[CameraImage(name="right_cam", image=right_img)]),
                Sample(timestamp=2.01, cameras=[CameraImage(name="right_cam", image=right_img)]),
            ]
        )

        merged = list(merge_streams(reader1, reader2, time_tolerance=0.05))

        assert len(merged) == 2

        # First merged sample should have both cameras
        assert len(merged[0].cameras) == 2
        camera_names = {cam.name for cam in merged[0].cameras}
        assert camera_names == {"left_cam", "right_cam"}

        # Second merged sample should also have both cameras
        assert len(merged[1].cameras) == 2

    def test_merge_streams_merge_proprio_and_vision(self) -> None:
        """Test merging vision and proprioceptive data."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        proprio = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        vision_reader = MockReader([Sample(timestamp=1.0, cameras=[CameraImage(name="cam", image=img)])])

        proprio_reader = MockReader([Sample(timestamp=1.01, proprio=proprio)])

        merged = list(merge_streams(vision_reader, proprio_reader, time_tolerance=0.05))

        assert len(merged) == 1
        assert len(merged[0].cameras) == 1
        assert merged[0].cameras[0].name == "cam"
        assert merged[0].proprio is not None
        assert np.array_equal(merged[0].proprio, proprio)

    def test_merge_streams_no_overlap(self) -> None:
        """Test merging streams with no temporal overlap."""
        reader1 = MockReader([Sample(timestamp=1.0, proprio=np.array([1.0], dtype=np.float32))])

        reader2 = MockReader([Sample(timestamp=10.0, proprio=np.array([2.0], dtype=np.float32))])

        merged = list(merge_streams(reader1, reader2, time_tolerance=0.05))

        # With no overlap, we should get 2 separate samples
        assert len(merged) == 2
        assert merged[0].timestamp == 1.0
        assert merged[1].timestamp == 10.0

    def test_merge_streams_action_data(self) -> None:
        """Test merging with action data."""
        action = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        reader1 = MockReader([Sample(timestamp=1.0, action=action)])

        reader2 = MockReader([Sample(timestamp=1.01, proprio=np.array([1.0], dtype=np.float32))])

        merged = list(merge_streams(reader1, reader2, time_tolerance=0.05))

        assert len(merged) == 1
        assert merged[0].action is not None
        assert np.array_equal(merged[0].action, action)
        assert merged[0].proprio is not None

    def test_merge_streams_metadata(self) -> None:
        """Test that metadata is merged."""
        reader1 = MockReader([Sample(timestamp=1.0, metadata={"source": "video", "fps": 30})])

        reader2 = MockReader([Sample(timestamp=1.01, metadata={"episode_id": "test123"})])

        merged = list(merge_streams(reader1, reader2, time_tolerance=0.05))

        assert len(merged) == 1
        assert "source" in merged[0].metadata
        assert "fps" in merged[0].metadata
        assert "episode_id" in merged[0].metadata

    def test_merge_streams_duplicate_camera_names(self) -> None:
        """Test merging prevents duplicate camera names."""
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.ones((480, 640, 3), dtype=np.uint8)

        reader1 = MockReader([Sample(timestamp=1.0, cameras=[CameraImage(name="cam", image=img1)])])

        reader2 = MockReader([Sample(timestamp=1.01, cameras=[CameraImage(name="cam", image=img2)])])

        merged = list(merge_streams(reader1, reader2, time_tolerance=0.05))

        # Only the first camera should be kept (no duplicates)
        assert len(merged) == 1
        assert len(merged[0].cameras) == 1
        assert merged[0].cameras[0].name == "cam"
        # Should use first occurrence
        assert np.array_equal(merged[0].cameras[0].image, img1)

    def test_merge_streams_multiple_proprio(self) -> None:
        """Test merging selects nearest proprio (not concatenate)."""
        proprio1 = np.array([1.0, 2.0], dtype=np.float32)
        proprio2 = np.array([3.0, 4.0], dtype=np.float32)

        reader1 = MockReader([Sample(timestamp=1.0, proprio=proprio1)])

        reader2 = MockReader([Sample(timestamp=1.01, proprio=proprio2)])

        merged = list(merge_streams(reader1, reader2, time_tolerance=0.05))

        assert len(merged) == 1
        assert merged[0].proprio is not None
        assert len(merged[0].proprio) == 2  # NOT 4!
        # Should select closest (reader1 at 1.0 is the anchor)
        assert np.array_equal(merged[0].proprio, proprio1)

    def test_merge_streams_time_tolerance(self) -> None:
        """Test that time_tolerance parameter works correctly."""
        reader1 = MockReader([Sample(timestamp=1.0, proprio=np.array([1.0], dtype=np.float32))])

        reader2 = MockReader([Sample(timestamp=1.1, proprio=np.array([2.0], dtype=np.float32))])

        # With small tolerance, should NOT merge
        merged_small = list(merge_streams(reader1, reader2, time_tolerance=0.05))
        assert len(merged_small) == 2

        # Reset readers
        reader1 = MockReader([Sample(timestamp=1.0, proprio=np.array([1.0], dtype=np.float32))])
        reader2 = MockReader([Sample(timestamp=1.1, proprio=np.array([2.0], dtype=np.float32))])

        # With large tolerance, SHOULD merge
        merged_large = list(merge_streams(reader1, reader2, time_tolerance=0.15))
        assert len(merged_large) == 1

    def test_merge_multiple_proprio_uses_nearest(self) -> None:
        """CRITICAL BUG: Multiple proprio should select nearest, not concatenate.

        When multiple MCAP samples (e.g., at 350Hz) fall within tolerance of a video frame,
        we should select the temporally closest sample, not concatenate all of them.
        Concatenation creates wrong dimensions and corrupts training data.
        """
        # Three proprio samples within tolerance window
        reader1 = MockReader([Sample(timestamp=1.00, proprio=np.array([1.0], dtype=np.float32))])
        reader2 = MockReader([Sample(timestamp=1.01, proprio=np.array([2.0], dtype=np.float32))])
        reader3 = MockReader([Sample(timestamp=1.03, proprio=np.array([3.0], dtype=np.float32))])

        merged = list(merge_streams(reader1, reader2, reader3, time_tolerance=0.05))

        assert len(merged) == 1
        # Should keep original shape, NOT concatenate to (3,)
        assert merged[0].proprio.shape == (1,)
        # Should pick reader1 (1.00) as closest to the anchor timestamp (1.00)
        # The anchor is 1.00, so: reader1=0.0s away, reader2=0.01s away, reader3=0.03s away
        assert merged[0].proprio[0] == pytest.approx(1.0)

    def test_merge_multiple_actions_uses_nearest(self) -> None:
        """CRITICAL BUG: Multiple actions should select nearest, not first.

        Taking the first action in list order is arbitrary and temporally incorrect.
        For imitation learning, wrong action labels = catastrophic training failure.
        """
        # Three action samples within tolerance window
        reader1 = MockReader([Sample(timestamp=1.00, action=np.array([1.0], dtype=np.float32))])
        reader2 = MockReader([Sample(timestamp=1.01, action=np.array([2.0], dtype=np.float32))])
        reader3 = MockReader([Sample(timestamp=1.03, action=np.array([3.0], dtype=np.float32))])

        merged = list(merge_streams(reader1, reader2, reader3, time_tolerance=0.05))

        assert len(merged) == 1
        # Should pick reader1 (1.00) as closest to the anchor timestamp (1.00)
        # The anchor is 1.00, so: reader1=0.0s away, reader2=0.01s away, reader3=0.03s away
        assert merged[0].action[0] == pytest.approx(1.0)

    def test_merge_timestamp_matching_uses_closest(self) -> None:
        """BUG: Should match to closest timestamp, not first match.

        When a sample could match multiple existing groups within tolerance,
        it should join the temporally closest group, not the first one found.
        """
        # Create three samples that will all match within 0.6s tolerance
        # reader1 creates group at 1.00
        # reader2 at 1.50 is 0.50s from 1.00, so within tolerance -> joins group at 1.00
        # reader3 at 1.45 is 0.45s from 1.00, so within tolerance -> should also join group at 1.00
        reader1 = MockReader([Sample(timestamp=1.00, proprio=np.array([1.0], dtype=np.float32))])
        reader2 = MockReader([Sample(timestamp=1.50, proprio=np.array([2.0], dtype=np.float32))])
        reader3 = MockReader([Sample(timestamp=1.45, proprio=np.array([3.0], dtype=np.float32))])

        merged = list(merge_streams(reader1, reader2, reader3, time_tolerance=0.6))

        # All should merge into 1 group since all are within 0.6s of each other
        assert len(merged) == 1

        # The group timestamp is 1.00 (from reader1 which creates the group)
        assert merged[0].timestamp == pytest.approx(1.00)

        # With nearest neighbor selection, should pick reader1 (0.0s away from 1.00)
        assert merged[0].proprio[0] == pytest.approx(1.0)
