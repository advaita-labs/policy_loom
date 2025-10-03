"""Tests for synchronized video+MCAP reader."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from loom.io.synchronized import SynchronizedVideoMCAPReader


class TestSynchronizedVideoMCAPReader:
    """Tests for SynchronizedVideoMCAPReader."""

    def test_file_not_found_video(self) -> None:
        """Test that FileNotFoundError is raised for missing video file."""
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_mcap:
            mcap_path = Path(tmp_mcap.name)

        try:
            with pytest.raises(FileNotFoundError, match="Video file not found"):
                SynchronizedVideoMCAPReader(
                    video_path="/nonexistent/video.mp4",
                    mcap_path=mcap_path,
                    camera_topic="/cam/state",
                    camera_name="cam",
                )
        finally:
            mcap_path.unlink()

    def test_file_not_found_mcap(self) -> None:
        """Test that FileNotFoundError is raised for missing MCAP file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            video_path = Path(tmp_video.name)

        try:
            with pytest.raises(FileNotFoundError, match="MCAP file not found"):
                SynchronizedVideoMCAPReader(
                    video_path=video_path,
                    mcap_path="/nonexistent/data.mcap",
                    camera_topic="/cam/state",
                    camera_name="cam",
                )
        finally:
            video_path.unlink()

    @patch("loom.io.synchronized.make_reader")
    @patch("cv2.VideoCapture")
    def test_synchronized_reading(self, mock_capture: MagicMock, mock_make_reader: MagicMock) -> None:
        """Test synchronized reading with mocked MCAP and video."""
        # Create temporary files
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video,
            tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_mcap,
        ):
            video_path = Path(tmp_video.name)
            mcap_path = Path(tmp_mcap.name)

        try:
            # Mock MCAP reader to return camera timestamps
            mock_reader_instance = MagicMock()
            mock_make_reader.return_value = mock_reader_instance

            # Create mock messages with camera timestamps
            mock_messages = []
            timestamps_ns = [1757503161_000_000_000, 1757503161_033_000_000, 1757503161_066_000_000]

            for ts_ns in timestamps_ns:
                mock_message = MagicMock()
                mock_message.log_time = ts_ns

                mock_channel = MagicMock()
                mock_channel.topic = "/test/cam/state"

                mock_messages.append((None, mock_channel, mock_message))

            mock_reader_instance.iter_messages.return_value = mock_messages

            # Mock video capture
            mock_cap_instance = MagicMock()
            mock_capture.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 3,
            }.get(prop, 0)

            # Mock frame reading
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) * i for i in [1, 2, 3]]
            mock_cap_instance.read.side_effect = [
                (True, frames[0]),
                (True, frames[1]),
                (True, frames[2]),
                (False, None),
            ]

            # Mock BGR to RGB conversion
            with patch("cv2.cvtColor", side_effect=lambda img, code: img), patch("builtins.open", MagicMock()):
                reader = SynchronizedVideoMCAPReader(
                    video_path=video_path,
                    mcap_path=mcap_path,
                    camera_topic="/test/cam/state",
                    camera_name="test_cam",
                )

                samples = list(reader.read())

            # Verify results
            assert len(samples) == 3

            # Check timestamps are from MCAP (absolute time)
            assert samples[0].timestamp == pytest.approx(1757503161.0)
            assert samples[1].timestamp == pytest.approx(1757503161.033)
            assert samples[2].timestamp == pytest.approx(1757503161.066)

            # Check camera data
            for sample in samples:
                assert len(sample.cameras) == 1
                assert sample.cameras[0].name == "test_cam"
                assert sample.cameras[0].image.shape == (480, 640, 3)

            # Check metadata
            assert samples[0].metadata["synchronized"] is True
            assert samples[0].metadata["mcap_topic"] == "/test/cam/state"

        finally:
            video_path.unlink()
            mcap_path.unlink()

    @patch("loom.io.synchronized.make_reader")
    @patch("cv2.VideoCapture")
    def test_frame_count_mismatch_warning(
        self, mock_capture: MagicMock, mock_make_reader: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that warning is logged when frame counts don't match."""
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video,
            tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_mcap,
        ):
            video_path = Path(tmp_video.name)
            mcap_path = Path(tmp_mcap.name)

        try:
            # Mock MCAP with 5 timestamps
            mock_reader_instance = MagicMock()
            mock_make_reader.return_value = mock_reader_instance

            mock_messages = []
            for i in range(5):
                mock_message = MagicMock()
                mock_message.log_time = (1757503161 + i * 0.033) * 1e9

                mock_channel = MagicMock()
                mock_channel.topic = "/cam/state"

                mock_messages.append((None, mock_channel, mock_message))

            mock_reader_instance.iter_messages.return_value = mock_messages

            # Mock video with only 2 frames
            mock_cap_instance = MagicMock()
            mock_capture.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 2,  # Mismatch!
            }.get(prop, 0)

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap_instance.read.side_effect = [(True, frame), (True, frame), (False, None)]

            with patch("cv2.cvtColor", side_effect=lambda img, code: img), patch("builtins.open", MagicMock()):
                reader = SynchronizedVideoMCAPReader(
                    video_path=video_path,
                    mcap_path=mcap_path,
                    camera_topic="/cam/state",
                    camera_name="cam",
                )

                with caplog.at_level("WARNING"):
                    samples = list(reader.read())

                # Should log warning about mismatch
                assert "Frame count mismatch" in caplog.text
                # Should log warning about video ending early
                assert "Video ended at frame 2" in caplog.text

                # Should still return the frames that were read
                assert len(samples) == 2

        finally:
            video_path.unlink()
            mcap_path.unlink()

    @patch("loom.io.synchronized.make_reader")
    def test_no_camera_timestamps_error(self, mock_make_reader: MagicMock) -> None:
        """Test that ValueError is raised when no camera timestamps found."""
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video,
            tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp_mcap,
        ):
            video_path = Path(tmp_video.name)
            mcap_path = Path(tmp_mcap.name)

        try:
            # Mock MCAP with no messages for the requested topic
            mock_reader_instance = MagicMock()
            mock_make_reader.return_value = mock_reader_instance
            mock_reader_instance.iter_messages.return_value = []

            with patch("builtins.open", MagicMock()):
                reader = SynchronizedVideoMCAPReader(
                    video_path=video_path,
                    mcap_path=mcap_path,
                    camera_topic="/nonexistent/topic",
                    camera_name="cam",
                )

                with pytest.raises(ValueError, match="No camera timestamps found"):
                    list(reader.read())

        finally:
            video_path.unlink()
            mcap_path.unlink()
