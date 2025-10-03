"""Utilities for merging data streams from multiple readers."""

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from loom.core.ports import Reader
from loom.core.types import Sample

logger = logging.getLogger(__name__)


def merge_streams(
    *readers: Reader,
    time_tolerance: float = 0.033,  # ~30fps tolerance
    strategy: str = "nearest",
) -> Iterator[Sample]:
    """Merge multiple readers into temporally aligned samples using nearest neighbor matching."""
    if not readers:
        return

    if len(readers) == 1:
        # Single reader - just pass through
        with readers[0] as reader:
            yield from reader.read()
        return

    # Multiple readers - collect all samples with timestamps
    all_samples: list[tuple[float, Sample, int]] = []

    logger.info(f"Collecting samples from {len(readers)} readers...")

    for reader_idx, reader in enumerate(readers):
        with reader as r:
            for sample in r.read():
                all_samples.append((sample.timestamp, sample, reader_idx))

    logger.info(f"Collected {len(all_samples)} total samples")

    # Sort by timestamp
    all_samples.sort(key=lambda x: x[0])

    if not all_samples:
        logger.warning("No samples found in any reader")
        return

    # Group samples by timestamp using nearest neighbor strategy
    merged_groups: dict[float, list[Sample]] = {}

    for timestamp, sample, _reader_idx in all_samples:
        # Find all candidates within tolerance
        candidates = [(ts, abs(ts - timestamp)) for ts in merged_groups.keys() if abs(ts - timestamp) <= time_tolerance]

        if candidates:
            # Match to CLOSEST timestamp (not first match)
            closest_ts = min(candidates, key=lambda x: x[1])[0]
            merged_groups[closest_ts].append(sample)
        else:
            # Create new group
            merged_groups[timestamp] = [sample]

    logger.info(f"Merged into {len(merged_groups)} synchronized sample groups")

    # Yield merged samples in temporal order
    for timestamp in sorted(merged_groups.keys()):
        samples = merged_groups[timestamp]

        # Merge all samples into one
        merged = _merge_samples(samples)
        yield merged


def _merge_samples(samples: list[Sample]) -> Sample:
    """Merge samples: combine cameras, select nearest proprio/action."""
    if len(samples) == 1:
        return samples[0]

    # Use first sample's timestamp as base
    merged_timestamp = samples[0].timestamp

    # Merge cameras - collect all cameras from all samples
    from loom.core.types import CameraImage

    merged_cameras: list[CameraImage] = []
    seen_camera_names: set[str] = set()

    for sample in samples:
        for camera in sample.cameras:
            if camera.name not in seen_camera_names:
                merged_cameras.append(camera)
                seen_camera_names.add(camera.name)

    # Merge proprioceptive data - use temporally closest
    proprio_data: np.ndarray | None = None
    proprio_samples = [s for s in samples if s.proprio is not None]
    if proprio_samples:
        # Take temporally closest proprio to the merged timestamp
        closest = min(proprio_samples, key=lambda s: abs(s.timestamp - merged_timestamp))
        proprio_data = closest.proprio

    # Merge action data - use temporally closest
    action_data: np.ndarray | None = None
    action_samples = [s for s in samples if s.action is not None]
    if action_samples:
        # Take temporally closest action to the merged timestamp
        closest = min(action_samples, key=lambda s: abs(s.timestamp - merged_timestamp))
        action_data = closest.action

    # Merge metadata
    merged_metadata: dict[str, Any] = {}
    for sample in samples:
        merged_metadata.update(sample.metadata)

    return Sample(
        timestamp=merged_timestamp,
        cameras=merged_cameras,
        proprio=proprio_data,
        action=action_data,
        metadata=merged_metadata,
    )
