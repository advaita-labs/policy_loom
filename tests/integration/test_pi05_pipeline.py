"""Integration test covering reader -> LeRobot -> Pi0.5 pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from loom.core.types import CameraImage, Sample
from loom.io.lerobot import (
    LeRobotConversionConfig,
    LeRobotDatasetWriter,
    collate_lerobot_batch,
    convert_readers_to_lerobot,
)
from loom.io.lerobot.pi05 import convert_lerobot_batch_to_pi05


class DummyReader:
    def __init__(self, samples):
        self._samples = samples

    def read(self):
        yield from self._samples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class DummyTransform:
    def __init__(self):
        self.last_batch = None

    def __call__(self, batch):
        self.last_batch = batch
        return "obs", batch["action"]


def _make_sample(camera_name: str, timestamp: float) -> Sample:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return Sample(
        timestamp=timestamp,
        cameras=[CameraImage(name=camera_name, image=img)],
        proprio=np.ones(7, dtype=np.float32),
        action=np.ones(7, dtype=np.float32),
        metadata={"frame_idx": 0},
    )


def test_end_to_end_conversion_pipeline():
    left_samples = [_make_sample("left", 0.0), _make_sample("left", 0.033)]
    right_samples = [_make_sample("right", 0.0), _make_sample("right", 0.033)]

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir) / "lerobot_data"
        writer = LeRobotDatasetWriter(
            repo_id="tests/robot",
            robot_type="dummy",
            fps=30,
            camera_names=["left", "right"],
            action_dim=7,
            proprio_dim=7,
            root=dataset_root,
            use_videos=False,
        )

        convert_readers_to_lerobot(
            [DummyReader(left_samples), DummyReader(right_samples)],
            writer,
            LeRobotConversionConfig(task="stack boxes"),
        )

        frames = [writer.dataset[i] for i in range(len(writer))]
        for frame in frames:
            frame.setdefault("metadata", {})
            frame["metadata"]["task"] = frame.get("task", "stack boxes")

        batch = collate_lerobot_batch(frames)

        transform = DummyTransform()
        _, actions = convert_lerobot_batch_to_pi05(batch, transform=transform)

        assert transform.last_batch["prompt"] == ["stack boxes"] * len(actions)
        assert actions.shape[-1] == 7
