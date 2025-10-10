A"""Tests for LeRobot conversion utilities."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from loom.core.ports import Reader
from loom.core.types import CameraImage, Sample
from loom.io.lerobot.converter import convert_readers_to_lerobot


class DummyReader(Reader):
    """Simple reader that yields a fixed list of samples."""

    def __init__(self, samples: Sequence[Sample]):
        self._samples = list(samples)

    def read(self):
        yield from self._samples


@dataclass
class EpisodeCapture:
    """Stores arguments passed to the stub writer."""

    samples: list[Sample]
    task: str


class StubWriter:
    """Minimal writer stub for capturing add_episode calls."""

    def __init__(self) -> None:
        self.episodes: list[EpisodeCapture] = []

    def add_episode(self, samples: list[Sample], task: str) -> None:
        self.episodes.append(EpisodeCapture(samples=list(samples), task=task))


def make_sample(
    *,
    timestamp: float,
    camera_name: str,
    proprio_dim: int | None = None,
    action_dim: int | None = None,
) -> Sample:
    """Helper to build Sample objects for tests."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    proprio = np.ones(proprio_dim, dtype=np.float32) if proprio_dim else None
    action = np.ones(action_dim, dtype=np.float32) if action_dim else None
    return Sample(
        timestamp=timestamp,
        cameras=[CameraImage(name=camera_name, image=image)],
        proprio=proprio,
        action=action,
    )


class TestConvertReadersToLeRobot:
    """TDD tests for reader-to-LeRobot conversion pipeline."""

    def test_converts_single_reader_episode(self):
        samples = [make_sample(timestamp=0.0, camera_name="left", proprio_dim=2, action_dim=2)]
        readers = [DummyReader(samples)]
        writer = StubWriter()

        frame_count = convert_readers_to_lerobot(readers, writer, task="demo")

        assert frame_count == 1
        assert len(writer.episodes) == 1
        assert writer.episodes[0].task == "demo"
        assert writer.episodes[0].samples[0].get_camera("left") is not None

    def test_merges_multiple_readers(self):
        left_samples = [
            make_sample(timestamp=0.0, camera_name="left", proprio_dim=2, action_dim=2),
            make_sample(timestamp=0.033, camera_name="left", proprio_dim=2, action_dim=2),
        ]
        right_samples = [
            make_sample(timestamp=0.0, camera_name="right", proprio_dim=2, action_dim=2),
            make_sample(timestamp=0.033, camera_name="right", proprio_dim=2, action_dim=2),
        ]

        writer = StubWriter()
        convert_readers_to_lerobot(
            [DummyReader(left_samples), DummyReader(right_samples)],
            writer,
            task="bimanual",
        )

        assert len(writer.episodes) == 1
        merged_samples = writer.episodes[0].samples
        assert merged_samples[0].get_camera("left") is not None
        assert merged_samples[0].get_camera("right") is not None

    def test_applies_filters(self):
        samples = [
            make_sample(timestamp=0.0, camera_name="cam", proprio_dim=2, action_dim=2),
            make_sample(timestamp=0.033, camera_name="cam", proprio_dim=2, action_dim=2),
        ]
        writer = StubWriter()

        def drop_first(batch: list[Sample]) -> list[Sample]:
            return batch[1:]

        frame_count = convert_readers_to_lerobot(
            [DummyReader(samples)],
            writer,
            task="filtered",
            filters=[drop_first],
        )

        assert frame_count == 1
        assert len(writer.episodes[0].samples) == 1
        assert writer.episodes[0].samples[0].timestamp == pytest.approx(0.033)

    def test_raises_when_no_readers(self):
        writer = StubWriter()
        with pytest.raises(ValueError, match="No readers provided"):
            convert_readers_to_lerobot([], writer, task="empty")

    def test_raises_when_no_samples(self):
        writer = StubWriter()
        reader = DummyReader([])

        with pytest.raises(ValueError, match="No samples available"):
            convert_readers_to_lerobot([reader], writer, task="empty")
