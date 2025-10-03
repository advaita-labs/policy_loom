"""Synchronized reader for matching video frames with MCAP camera timestamps."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
from mcap.reader import make_reader

from loom.core.ports import Reader
from loom.core.types import CameraImage, Sample

logger = logging.getLogger(__name__)


class SynchronizedVideoMCAPReader(Reader):
    """Read video frames with timestamps extracted from MCAP camera messages.

    Solves video (relative time) vs MCAP (absolute time) mismatch.
    """

    def __init__(
        self,
        video_path: str | Path,
        mcap_path: str | Path,
        camera_topic: str,
        camera_name: str,
    ) -> None:
        """Initialize synchronized reader."""
        self.video_path = Path(video_path)
        self.mcap_path = Path(mcap_path)
        self.camera_topic = camera_topic
        self.camera_name = camera_name

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.mcap_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {self.mcap_path}")

        self._cap: cv2.VideoCapture | None = None
        self._mcap_file: Any = None

    def read(self) -> Iterator[Sample]:
        """Yield video frames with MCAP timestamps."""
        # Extract camera timestamps from MCAP
        camera_timestamps = self._extract_camera_timestamps()

        if not camera_timestamps:
            raise ValueError(f"No camera timestamps found for topic: {self.camera_topic}")

        logger.info(
            f"Found {len(camera_timestamps)} camera timestamps for {self.camera_name} "
            f"({camera_timestamps[0]:.3f}s to {camera_timestamps[-1]:.3f}s)"
        )

        # Open video
        self._cap = cv2.VideoCapture(str(self.video_path))

        if not self._cap.isOpened():
            raise OSError(f"Failed to open video file: {self.video_path}")

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Video: {total_frames} frames at {fps:.2f} FPS")

        # Verify frame count roughly matches timestamp count
        if abs(total_frames - len(camera_timestamps)) > 2:
            logger.warning(
                f"Frame count mismatch: video has {total_frames} frames, "
                f"but MCAP has {len(camera_timestamps)} timestamps. "
                f"Difference: {abs(total_frames - len(camera_timestamps))}"
            )

        # Read frames and assign MCAP timestamps
        frame_idx = 0
        for timestamp in camera_timestamps:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"Video ended at frame {frame_idx}, but expected {len(camera_timestamps)} frames")
                break

            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create camera image
            camera = CameraImage(name=self.camera_name, image=rgb)

            # Build metadata
            metadata: dict[str, Any] = {
                "frame_idx": frame_idx,
                "source": str(self.video_path.name),
                "mcap_topic": self.camera_topic,
                "synchronized": True,
            }

            yield Sample(
                timestamp=timestamp,
                cameras=[camera],
                metadata=metadata,
            )

            frame_idx += 1

        logger.info(f"Read {frame_idx} synchronized frames from {self.video_path.name}")

    def _extract_camera_timestamps(self) -> list[float]:
        """Extract camera message timestamps from MCAP (in seconds, sorted)."""
        self._mcap_file = open(self.mcap_path, "rb")
        reader = make_reader(self._mcap_file)

        timestamps = []

        for _schema, channel, message in reader.iter_messages():
            if channel and channel.topic == self.camera_topic:
                timestamp = message.log_time / 1e9  # Convert nanoseconds to seconds
                timestamps.append(timestamp)

        self._mcap_file.close()
        self._mcap_file = None

        # Sort timestamps to ensure temporal order
        timestamps.sort()

        return timestamps

    def close(self) -> None:
        """Release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._mcap_file is not None:
            self._mcap_file.close()
            self._mcap_file = None
