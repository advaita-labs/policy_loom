"""Tests for I/O readers (MP4, MCAP)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from loom.io.mcap import MCAPReader
from loom.io.mp4 import MP4Reader


class TestMP4Reader:
    """Tests for MP4Reader."""

    def test_mp4_reader_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            MP4Reader("/nonexistent/path/video.mp4")

    @patch("cv2.VideoCapture")
    def test_mp4_reader_basic(self, mock_capture: MagicMock) -> None:
        """Test basic MP4 reading with mocked OpenCV."""
        # Create a temporary file to satisfy path existence check
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Mock VideoCapture behavior
            mock_cap_instance = MagicMock()
            mock_capture.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 3,
            }.get(prop, 0)

            # Mock frame reading: return 3 frames then False
            frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
            frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
            frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 255

            mock_cap_instance.read.side_effect = [
                (True, frame1),
                (True, frame2),
                (True, frame3),
                (False, None),
            ]

            # Mock BGR to RGB conversion
            with patch("cv2.cvtColor", side_effect=lambda img, code: img):
                reader = MP4Reader(tmp_path, camera_name="test_cam")
                samples = list(reader.read())

            assert len(samples) == 3

            # Check first sample
            assert samples[0].timestamp == 0.0
            assert len(samples[0].cameras) == 1
            assert samples[0].cameras[0].name == "test_cam"
            assert samples[0].cameras[0].image.shape == (480, 640, 3)
            assert samples[0].metadata["frame_idx"] == 0
            assert samples[0].metadata["fps"] == 30.0

            # Check second sample timestamp
            assert samples[1].timestamp == pytest.approx(1 / 30.0)

            # Check third sample timestamp
            assert samples[2].timestamp == pytest.approx(2 / 30.0)

        finally:
            tmp_path.unlink()

    @patch("cv2.VideoCapture")
    def test_mp4_reader_with_start_time(self, mock_capture: MagicMock) -> None:
        """Test MP4Reader with start_time offset."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            mock_cap_instance = MagicMock()
            mock_capture.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 2,
            }.get(prop, 0)

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap_instance.read.side_effect = [(True, frame), (True, frame), (False, None)]

            with patch("cv2.cvtColor", side_effect=lambda img, code: img):
                reader = MP4Reader(tmp_path, camera_name="test_cam", start_time=10.0)
                samples = list(reader.read())

            # Check timestamps are offset
            assert samples[0].timestamp == 10.0
            assert samples[1].timestamp == pytest.approx(10.0 + 1 / 30.0)

        finally:
            tmp_path.unlink()

    @patch("cv2.VideoCapture")
    def test_mp4_reader_default_camera_name(self, mock_capture: MagicMock) -> None:
        """Test MP4Reader with default camera name."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            mock_cap_instance = MagicMock()
            mock_capture.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 1,
            }.get(prop, 0)

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap_instance.read.side_effect = [(True, frame), (False, None)]

            with patch("cv2.cvtColor", side_effect=lambda img, code: img):
                reader = MP4Reader(tmp_path)  # No camera_name
                samples = list(reader.read())

            assert samples[0].cameras[0].name == "default"

        finally:
            tmp_path.unlink()

    @patch("cv2.VideoCapture")
    def test_mp4_reader_failed_to_open(self, mock_capture: MagicMock) -> None:
        """Test that IOError is raised if video cannot be opened."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            mock_cap_instance = MagicMock()
            mock_capture.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = False

            reader = MP4Reader(tmp_path)

            with pytest.raises(IOError, match="Failed to open video file"):
                list(reader.read())

        finally:
            tmp_path.unlink()

    @patch("cv2.VideoCapture")
    def test_mp4_reader_invalid_fps(self, mock_capture: MagicMock) -> None:
        """Test that ValueError is raised for invalid FPS."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            mock_cap_instance = MagicMock()
            mock_capture.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.return_value = 0.0  # Invalid FPS

            reader = MP4Reader(tmp_path)

            with pytest.raises(ValueError, match="Invalid FPS"):
                list(reader.read())

        finally:
            tmp_path.unlink()

    @patch("cv2.VideoCapture")
    def test_mp4_reader_context_manager(self, mock_capture: MagicMock) -> None:
        """Test MP4Reader as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            mock_cap_instance = MagicMock()
            mock_capture.return_value = mock_cap_instance
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 1,
            }.get(prop, 0)
            mock_cap_instance.read.return_value = (False, None)

            with patch("cv2.cvtColor", side_effect=lambda img, code: img):
                with MP4Reader(tmp_path) as reader:
                    assert reader is not None
                    # Call read to initialize _cap
                    list(reader.read())

            # After context exit, release should be called
            mock_cap_instance.release.assert_called_once()

        finally:
            tmp_path.unlink()


class TestMCAPReader:
    """Tests for MCAPReader."""

    def test_mcap_reader_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            MCAPReader("/nonexistent/path/data.mcap")

    def test_mcap_reader_basic(self) -> None:
        """Test basic MCAP reading with a real temp file."""
        # Create a minimal MCAP file structure for testing
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Write minimal MCAP header (this is a simplified test)
            tmp.write(b"\x89MCAP0\r\n")

        try:
            # Since we're creating a minimal file, the reader might fail gracefully
            # or we need to mock the internal mcap reader
            reader = MCAPReader(tmp_path)
            # For now, just verify it initializes without error
            assert reader.mcap_path == tmp_path

        finally:
            tmp_path.unlink()

    def test_mcap_reader_context_manager(self) -> None:
        """Test MCAPReader as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with MCAPReader(tmp_path) as reader:
                assert reader is not None
            # File should be closed after context exit

        finally:
            tmp_path.unlink()
