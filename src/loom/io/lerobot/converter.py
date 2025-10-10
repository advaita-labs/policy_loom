"""Utilities for converting reader streams into LeRobot episodes."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from loom.core.ports import Reader
from loom.core.types import Sample
from loom.pipeline.merge import merge_streams

from loom.io.lerobot.writer import LeRobotDatasetWriter

SampleFilter = Callable[[list[Sample]], list[Sample]]


@dataclass(slots=True)
class LeRobotConversionConfig:
    """Configuration for converting merged samples into a LeRobot dataset episode."""

    task: str
    time_tolerance: float = 0.033
    filters: Sequence[SampleFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.task or not self.task.strip():
            raise ValueError("A non-empty task description must be provided.")


class _ReaderWrapper:
    """Wrap readers lacking context manager support."""

    def __init__(self, reader: Reader) -> None:
        self._reader = reader

    def __getattr__(self, item):
        return getattr(self._reader, item)

    def __enter__(self) -> Reader:
        return self._reader

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


def convert_readers_to_lerobot(
    readers: Sequence[Reader],
    writer: LeRobotDatasetWriter,
    config: LeRobotConversionConfig,
) -> int:
    """Convert one episode produced by readers into a LeRobot dataset episode."""

    if not readers:
        raise ValueError("No readers provided for conversion")

    wrapped: list[Reader] = []
    for reader in readers:
        if hasattr(reader, "__enter__") and hasattr(reader, "__exit__"):
            wrapped.append(reader)
        else:
            wrapped.append(_ReaderWrapper(reader))

    merged_samples = list(merge_streams(*wrapped, time_tolerance=config.time_tolerance))

    for filter_fn in config.filters:
        merged_samples = filter_fn(merged_samples)

    if not merged_samples:
        raise ValueError("No samples available after merging/filters")

    writer.add_episode(merged_samples, task=config.task)
    return len(merged_samples)
