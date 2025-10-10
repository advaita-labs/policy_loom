"""Utilities for converting reader streams into LeRobot episodes."""

from collections.abc import Callable, Sequence

from loom.core.ports import Reader
from loom.core.types import Sample
from loom.pipeline.merge import merge_streams

from loom.io.lerobot.writer import LeRobotDatasetWriter

SampleFilter = Callable[[list[Sample]], list[Sample]]


def convert_readers_to_lerobot(
    readers: Sequence[Reader],
    writer: LeRobotDatasetWriter,
    task: str,
    *,
    time_tolerance: float = 0.033,
    filters: Sequence[SampleFilter] | None = None,
) -> int:
    """Convert one episode produced by readers into a LeRobot dataset episode.

    Args:
        readers: Sequence of Reader instances providing synchronized samples.
        writer: LeRobot dataset writer to append the episode to.
        task: Task/episode description stored in the dataset.
        time_tolerance: Maximum timestamp difference (seconds) when merging streams.
        filters: Optional sequence of callables applied to the merged samples.

    Returns:
        Number of frames written to the dataset episode.

    Raises:
        ValueError: If no readers are provided or no samples are produced.
    """
    if not readers:
        raise ValueError("No readers provided for conversion")

    merged_samples = list(merge_streams(*readers, time_tolerance=time_tolerance))

    if filters:
        for filter_fn in filters:
            merged_samples = filter_fn(merged_samples)

    if not merged_samples:
        raise ValueError("No samples available after merging/filters")

    writer.add_episode(merged_samples, task=task)
    return len(merged_samples)
