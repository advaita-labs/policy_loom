"""MCAP reader implementation."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import make_reader

from loom.core.ports import Reader
from loom.core.types import Sample

logger = logging.getLogger(__name__)


class MCAPReader(Reader):
    """Read proprioceptive and action data from MCAP robotics telemetry files."""

    def __init__(
        self,
        mcap_path: str | Path,
        proprio_topic: str | None = None,
        action_topic: str | None = None,
        decode_json: bool = True,
    ) -> None:
        """Initialize MCAP reader."""
        self.mcap_path = Path(mcap_path)
        self.proprio_topic = proprio_topic
        self.action_topic = action_topic
        self.decode_json = decode_json
        self._file_handle: Any = None

        if not self.mcap_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {self.mcap_path}")

    def read(self) -> Iterator[Sample]:
        """Yield samples with proprio/action data grouped by timestamp."""
        self._file_handle = open(self.mcap_path, "rb")
        reader = make_reader(self._file_handle)

        # Log available topics
        summary = reader.get_summary()
        if summary and summary.statistics:
            logger.info(f"Reading MCAP file: {self.mcap_path.name}")
            logger.info(f"Message count: {summary.statistics.message_count}")
            if summary.statistics.channel_message_counts:
                logger.info("Available channels:")
                for channel_id, count in summary.statistics.channel_message_counts.items():
                    logger.info(f"  Channel {channel_id}: {count} messages")

        # Collect messages grouped by timestamp
        messages_by_time: dict[float, dict[str, Any]] = {}

        for _schema, channel, message in reader.iter_messages():
            # Convert nanosecond timestamp to seconds
            timestamp = message.log_time / 1e9

            if timestamp not in messages_by_time:
                messages_by_time[timestamp] = {}

            # Decode message data
            try:
                if self.decode_json:
                    data = json.loads(message.data.decode("utf-8"))
                else:
                    data = message.data
            except (UnicodeDecodeError, json.JSONDecodeError):
                # If JSON decoding fails, store raw bytes
                data = message.data

            # Store message by topic
            topic = channel.topic if channel else "unknown"
            messages_by_time[timestamp][topic] = data

        logger.info(f"Found {len(messages_by_time)} unique timestamps")

        # Yield samples in temporal order
        for timestamp in sorted(messages_by_time.keys()):
            message_data = messages_by_time[timestamp]

            # Extract proprioceptive data
            proprio = None
            if self.proprio_topic and self.proprio_topic in message_data:
                proprio = self._extract_array(message_data[self.proprio_topic])

            # Extract action data
            action = None
            if self.action_topic and self.action_topic in message_data:
                action = self._extract_array(message_data[self.action_topic])

            # Build metadata
            metadata: dict[str, Any] = {
                "source": str(self.mcap_path.name),
                "topics": list(message_data.keys()),
            }

            yield Sample(
                timestamp=timestamp,
                proprio=proprio,
                action=action,
                metadata=metadata,
            )

    def _extract_array(self, data: Any) -> np.ndarray | None:
        """Extract numpy array from message data.

        Handles various common formats:
        - Lists/tuples of numbers
        - Dicts with 'data', 'position', 'velocity', or 'values' keys
        - Already numpy arrays

        Args:
            data: Message data to extract array from

        Returns:
            Numpy array or None if extraction fails
        """
        if data is None:
            return None

        # Handle dict with common keys
        if isinstance(data, dict):
            for key in ["data", "position", "velocity", "values", "effort"]:
                if key in data:
                    return self._extract_array(data[key])
            # If no known keys, try to extract all numeric values
            numeric_values = [v for v in data.values() if isinstance(v, int | float)]
            if numeric_values:
                return np.array(numeric_values, dtype=np.float32)
            return None

        # Handle list/tuple
        if isinstance(data, list | tuple):
            try:
                return np.array(data, dtype=np.float32)
            except (ValueError, TypeError):
                return None

        # Handle numpy array
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)

        # Handle single number
        if isinstance(data, int | float):
            return np.array([data], dtype=np.float32)

        return None

    def close(self) -> None:
        """Release file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
