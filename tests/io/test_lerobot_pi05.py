"""Tests for Pi0.5 conversion helpers."""

from __future__ import annotations

import numpy as np
import pytest

from loom.io.lerobot.pi05 import convert_lerobot_batch_to_pi05


class DummyTransform:
    """Stub OpenPITransform that records incoming batches."""

    def __init__(self):
        self.last_batch = None

    def __call__(self, batch):
        self.last_batch = batch
        return "obs", batch["action"]


def _make_batch(tasks):
    actions = np.zeros((len(tasks), 7), dtype=np.float32)
    images = [{"cam": np.zeros((4, 4, 3), dtype=np.uint8)} for _ in tasks]
    metadata = [{"task": t} for t in tasks]
    return {"action": actions, "images": images, "metadata": metadata}


class TestConvertLeRobotBatchToPi05:
    def test_populates_prompt_from_metadata(self):
        transform = DummyTransform()
        batch = _make_batch(["pick", "place"])

        convert_lerobot_batch_to_pi05(batch, transform=transform)

        assert transform.last_batch is not None
        assert transform.last_batch["prompt"] == ["pick", "place"]

    def test_keeps_existing_prompts(self):
        transform = DummyTransform()
        batch = _make_batch(["pick"])
        batch["prompt"] = ["custom"]

        convert_lerobot_batch_to_pi05(batch, transform=transform)

        assert transform.last_batch["prompt"] == ["custom"]

    def test_uses_task_field_when_present(self):
        transform = DummyTransform()
        actions = np.zeros((2, 7), dtype=np.float32)
        images = [{"cam": np.zeros((4, 4, 3), dtype=np.uint8)} for _ in range(2)]
        batch = {"action": actions, "images": images, "task": ["stack", "slide"]}

        convert_lerobot_batch_to_pi05(batch, transform=transform)

        assert transform.last_batch["prompt"] == ["stack", "slide"]

    def test_uses_default_prompt_when_missing_tasks(self):
        transform = DummyTransform()
        actions = np.zeros((2, 7), dtype=np.float32)
        images = [{"cam": np.zeros((4, 4, 3), dtype=np.uint8)} for _ in range(2)]
        batch = {"action": actions, "images": images, "metadata": [{}, {}]}

        convert_lerobot_batch_to_pi05(batch, transform=transform, default_prompt="generic")

        assert transform.last_batch["prompt"] == ["generic", "generic"]

    def test_raises_when_no_prompt_available(self):
        transform = DummyTransform()
        actions = np.zeros((1, 7), dtype=np.float32)
        images = [{"cam": np.zeros((4, 4, 3), dtype=np.uint8)}]
        batch = {"action": actions, "images": images}

        with pytest.raises(ValueError, match="prompt"):
            convert_lerobot_batch_to_pi05(batch, transform=transform)
