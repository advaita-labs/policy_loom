"""MP4 video reader implementation."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2

from loom.core.ports import Reader
from loom.core.types import CameraImage, Sample

logger = logging.getLogger(__name__)


class MP4Reader(Reader):
    """Read RGB frames from MP4 video. Timestamps calculated from FPS."""

    def __init__(
        self,
        video_path: str | Path,
        camera_name: str | None = None,
        start_time: float = 0.0,
    ) -> None:
        """Initialize MP4 reader."""
        self.video_path = Path(video_path)
        self.camera_name = camera_name
        self.start_time = start_time
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._frame_count: int = 0

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    def read(self) -> Iterator[Sample]:
        """Yield video frames as Sample objects."""
        self._cap = cv2.VideoCapture(str(self.video_path))

        if not self._cap.isOpened():
            raise OSError(f"Failed to open video file: {self.video_path}")

        # Get video properties
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self._fps <= 0:
            raise ValueError(f"Invalid FPS: {self._fps}")

        logger.info(f"Reading {total_frames} frames from {self.video_path.name} at {self._fps:.2f} FPS")

        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV default) to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Calculate timestamp based on frame index and FPS
            timestamp = self.start_time + (frame_idx / self._fps)

            # Build metadata
            metadata: dict[str, Any] = {
                "frame_idx": frame_idx,
                "source": str(self.video_path.name),
                "fps": self._fps,
            }

            # Create camera image
            camera_name = self.camera_name or "default"
            camera = CameraImage(name=camera_name, image=rgb)

            yield Sample(
                timestamp=timestamp,
                cameras=[camera],
                metadata=metadata,
            )

            frame_idx += 1
            self._frame_count = frame_idx

        logger.info(f"Finished reading {self._frame_count} frames from {self.video_path.name}")

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
