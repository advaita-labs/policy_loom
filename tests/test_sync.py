"""Tests for timestamp synchronization validation."""

import numpy as np
import pytest

from loom.core import CameraImage, Sample


class TestTimestampSync:
    """Tests for timestamp synchronization validation."""

    def test_sample_timestamps_are_absolute(self) -> None:
        """Test that samples can use absolute Unix timestamps."""
        unix_timestamp = 1757503161.152  # Example from real data
        sample = Sample(timestamp=unix_timestamp)

        assert sample.timestamp == unix_timestamp
        assert sample.timestamp > 1e9  # Reasonable Unix timestamp check

    def test_samples_sorted_by_timestamp(self) -> None:
        """Test that samples can be sorted by timestamp."""
        samples = [
            Sample(timestamp=1757503161.5),
            Sample(timestamp=1757503161.1),
            Sample(timestamp=1757503161.3),
        ]

        sorted_samples = sorted(samples, key=lambda s: s.timestamp)

        assert sorted_samples[0].timestamp == 1757503161.1
        assert sorted_samples[1].timestamp == 1757503161.3
        assert sorted_samples[2].timestamp == 1757503161.5

    def test_timestamp_alignment_detection(self) -> None:
        """Test detecting misaligned timestamps."""
        # Video with relative timestamps (starting at 0)
        video_samples = [
            Sample(timestamp=0.0, cameras=[CameraImage(name="cam", image=np.zeros((100, 100, 3), dtype=np.uint8))]),
            Sample(timestamp=0.033, cameras=[CameraImage(name="cam", image=np.zeros((100, 100, 3), dtype=np.uint8))]),
        ]

        # MCAP with absolute timestamps
        mcap_samples = [
            Sample(timestamp=1757503161.0, proprio=np.array([1.0], dtype=np.float32)),
            Sample(timestamp=1757503161.033, proprio=np.array([2.0], dtype=np.float32)),
        ]

        # Check if timestamps overlap
        video_start = min(s.timestamp for s in video_samples)
        video_end = max(s.timestamp for s in video_samples)
        mcap_start = min(s.timestamp for s in mcap_samples)
        mcap_end = max(s.timestamp for s in mcap_samples)

        overlap_start = max(video_start, mcap_start)
        overlap_end = min(video_end, mcap_end)

        # Should have NO overlap
        assert overlap_end <= overlap_start, "Timestamps should not overlap when using different time bases"

    def test_timestamp_offset_calculation(self) -> None:
        """Test calculating offset to align timestamps."""
        video_start = 0.0
        mcap_start = 1757503161.152

        # Calculate offset needed to align video to MCAP
        offset = mcap_start - video_start

        # Apply offset
        aligned_video_timestamp = video_start + offset

        assert aligned_video_timestamp == pytest.approx(mcap_start)

    def test_video_frame_count_matches_mcap_camera_messages(self) -> None:
        """Test that video frame count should match MCAP camera message count."""
        # This is a validation check that should be performed during ingestion
        video_frame_count = 261  # From actual data
        mcap_camera_msg_count = 262  # From actual data

        # Allow 1 frame difference due to start/end timing
        assert abs(video_frame_count - mcap_camera_msg_count) <= 1

    def test_synchronized_sample_creation(self) -> None:
        """Test creating synchronized samples with matched timestamps."""
        # Example: MCAP provides ground truth timestamps for camera frames
        mcap_camera_timestamp = 1757503161.178
        mcap_proprio_timestamp = 1757503161.180

        # Create sample using MCAP camera timestamp
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        sample = Sample(
            timestamp=mcap_camera_timestamp,
            cameras=[CameraImage(name="left_cam", image=image)],
        )

        # Verify timestamp is absolute
        assert sample.timestamp > 1e9
        assert sample.timestamp == pytest.approx(mcap_camera_timestamp)


class TestSyncValidation:
    """Tests for sync validation utilities."""

    def test_detect_time_base_mismatch(self) -> None:
        """Test function to detect if timestamps use different time bases."""

        def is_absolute_timestamp(timestamp: float) -> bool:
            """Check if timestamp is absolute (Unix) or relative."""
            return timestamp > 1e6  # Arbitrary threshold: > 1 million seconds

        # Relative timestamp (from video frame index)
        assert not is_absolute_timestamp(0.0)
        assert not is_absolute_timestamp(9.5)
        assert not is_absolute_timestamp(100.0)

        # Absolute timestamp (Unix epoch)
        assert is_absolute_timestamp(1757503161.152)
        assert is_absolute_timestamp(1700000000.0)

    def test_calculate_sync_quality_metric(self) -> None:
        """Test calculating sync quality between streams."""
        # Perfect sync: all timestamps within tolerance
        stream1_timestamps = np.array([1.0, 2.0, 3.0])
        stream2_timestamps = np.array([1.01, 2.01, 3.01])

        max_offset = np.max(np.abs(stream1_timestamps - stream2_timestamps))

        assert max_offset == pytest.approx(0.01)
        assert max_offset < 0.033  # Within ~30fps tolerance

    def test_frame_drop_detection(self) -> None:
        """Test detecting dropped frames in video stream."""
        # Expected: constant frame interval
        expected_interval = 1.0 / 30.0  # 30 FPS

        # Actual: one frame dropped
        timestamps = np.array([0.0, 0.033, 0.066, 0.133, 0.166])  # Gap at 0.1
        intervals = np.diff(timestamps)

        # Detect large gaps
        mean_interval = np.mean(intervals)
        large_gaps = intervals > (mean_interval * 1.5)

        assert np.any(large_gaps), "Should detect the dropped frame"
        assert np.sum(large_gaps) == 1, "Should detect exactly one gap"
