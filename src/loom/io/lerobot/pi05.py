"""Utilities for preparing LeRobot data for Pi0.5 training."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from loom.training.transforms.openpi_transform import OpenPITransform


def _ensure_prompts(batch: dict[str, Any], default_prompt: str | None) -> None:
    """Populate batch['prompt'] from metadata if missing."""
    if "prompt" in batch and batch["prompt"] is not None:
        return

    metadata = batch.get("metadata")
    if metadata:
        prompts: list[str] = []
        for meta in metadata:
            task = meta.get("task") if isinstance(meta, dict) else None
            if task is None:
                if default_prompt is None:
                    raise ValueError("Metadata missing 'task' entry and no default prompt provided.")
                task = default_prompt
            prompts.append(task)
        batch["prompt"] = prompts
        return

    task_values = batch.get("task")
    if task_values is not None:
        if isinstance(task_values, str):
            batch["prompt"] = [task_values] * len(batch["action"])
        elif isinstance(task_values, Sequence):
            batch["prompt"] = list(task_values)
        else:
            raise ValueError("Unsupported task field format; expected string or sequence of strings.")
        return

    if default_prompt is not None:
        batch["prompt"] = [default_prompt] * len(batch["action"])
    else:
        raise ValueError("No prompt data available; provide metadata with 'task' or default_prompt.")


def _resolve_transform(
    transform: "OpenPITransform" | None,
    *,
    tokenizer: Any | None,
    default_prompt: str | None,
    camera_name_mapping: dict[str, str] | None,
    image_size: tuple[int, int],
) -> "OpenPITransform":
    if transform is not None:
        return transform

    try:
        from loom.training.transforms.openpi_transform import OpenPITransform  # local import
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise ImportError(
            "OpenPITransform requires the 'openpi' extra; install with `uv sync --extra pi05`."
        ) from exc

    return OpenPITransform(
        tokenizer=tokenizer,
        image_size=image_size,
        default_prompt=default_prompt,
        camera_name_mapping=camera_name_mapping,
    )


def convert_lerobot_batch_to_pi05(
    batch: dict[str, Any],
    *,
    transform: "OpenPITransform" | None = None,
    tokenizer: Any | None = None,
    default_prompt: str | None = None,
    camera_name_mapping: dict[str, str] | None = None,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[Any, Any]:
    """Convert a LeRobot batch into Pi0.5/OpenPI training inputs.

    Args:
        batch: Batch dictionary produced by LeRobot dataloaders.
        transform: Optional pre-instantiated OpenPITransform. If not supplied,
            one will be created with the provided tokenizer/default_prompt settings.
        tokenizer: Tokenizer passed to OpenPITransform when auto-instantiated.
        default_prompt: Default task prompt if metadata does not include 'task'.
        camera_name_mapping: Optional mapping for camera keys (passed to transform).
        image_size: Target image size passed to transform when auto-instantiated.

    Returns:
        Tuple of (Observation, actions_tensor) produced by the transform.

    Raises:
        ValueError: If prompts cannot be inferred for the batch.
    """
    working_batch = deepcopy(batch)
    _ensure_prompts(working_batch, default_prompt)

    resolved_transform = _resolve_transform(
        transform,
        tokenizer=tokenizer,
        default_prompt=default_prompt,
        camera_name_mapping=camera_name_mapping,
        image_size=image_size,
    )

    observation, actions = resolved_transform(working_batch)
    return observation, actions
