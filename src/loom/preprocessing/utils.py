"""Utilities for preprocessing pipelines."""

import logging
from collections.abc import Sequence

from loom.core.types import Sample

logger = logging.getLogger(__name__)


def filter_samples_by_cameras(
    samples: Sequence[Sample],
    required_cameras: Sequence[str],
) -> list[Sample]:
    """Filter samples to only include those with all required cameras.

    This is useful when dealing with robot data where camera synchronization
    is imperfect and some samples may be missing certain camera views.

    Args:
        samples: Input samples to filter
        required_cameras: List of camera names that must be present

    Returns:
        List of samples that have all required cameras

    Example:
        >>> from loom.preprocessing.utils import filter_samples_by_cameras
        >>> samples = [...]  # Some samples missing cameras
        >>> filtered = filter_samples_by_cameras(
        ...     samples,
        ...     required_cameras=["left_cam", "right_cam", "middle_cam"]
        ... )
        >>> # filtered only contains samples with all 3 cameras
    """
    valid_samples = []
    skipped_count = 0
    required_set = set(required_cameras)

    for sample in samples:
        available_cameras = {c.name for c in sample.cameras}

        if required_set.issubset(available_cameras):
            valid_samples.append(sample)
        else:
            skipped_count += 1
            missing = required_set - available_cameras
            logger.debug(f"Skipping sample at t={sample.timestamp:.4f}s: missing cameras {missing}")

    if skipped_count > 0:
        logger.info(
            f"Filtered {len(samples)} samples: "
            f"{len(valid_samples)} valid, {skipped_count} skipped "
            f"({skipped_count/len(samples)*100:.1f}% loss)"
        )

        # Warn if data loss is significant
        if skipped_count / len(samples) > 0.5:
            logger.warning(
                f"High data loss: {skipped_count/len(samples)*100:.1f}% of samples "
                f"missing required cameras. Consider investigating camera synchronization."
            )

    return valid_samples
