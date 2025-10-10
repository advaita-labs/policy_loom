"""Tests for LeRobot dataset writer.

Following TDD: Write tests first, then implement to make them pass.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from loom.core.types import CameraImage, Sample
from loom.io.lerobot import LeRobotDatasetWriter


class TestLeRobotDatasetWriter:
    """Test suite for LeRobot dataset writer."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary directory for test datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for dataset writer."""
        return {
            "repo_id": "test/robot_data",
            "robot_type": "test_robot",
            "fps": 30,
            "camera_names": ["left_cam", "right_cam"],
            "action_dim": 7,
            "proprio_dim": 7,
        }

    @pytest.fixture
    def sample_frames(self):
        """Create sample frames for testing."""
        samples = []
        for i in range(10):
            # Create sample with two cameras
            left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            cameras = [
                CameraImage(name="left_cam", image=left_img),
                CameraImage(name="right_cam", image=right_img),
            ]

            # Create sample with proprio and action
            sample = Sample(
                timestamp=1000.0 + i * 0.033,  # 30 FPS
                cameras=cameras,
                proprio=np.random.randn(7).astype(np.float32),
                action=np.random.randn(7).astype(np.float32),
                metadata={"frame_idx": i},
            )
            samples.append(sample)

        return samples

    def test_writer_initialization(self, temp_dataset_dir, sample_config):
        """Test 1: Writer initializes correctly."""
        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=sample_config["action_dim"],
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,  # Use images for faster tests
        )

        assert writer.repo_id == sample_config["repo_id"]
        assert writer.fps == sample_config["fps"]
        assert len(writer.camera_names) == 2

    def test_add_single_episode(self, temp_dataset_dir, sample_config, sample_frames):
        """Test 2: Can add a single episode."""
        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=sample_config["action_dim"],
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,
        )

        # Add episode
        writer.add_episode(sample_frames, task="pick_cube")

        # Verify episode was added
        assert writer.num_episodes == 1
        assert len(writer) == len(sample_frames)

    def test_add_multiple_episodes(self, temp_dataset_dir, sample_config, sample_frames):
        """Test 3: Can add multiple episodes."""
        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=sample_config["action_dim"],
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,
        )

        # Add two episodes
        writer.add_episode(sample_frames, task="pick_cube")
        writer.add_episode(sample_frames, task="place_cube")

        assert writer.num_episodes == 2
        assert len(writer) == len(sample_frames) * 2

    def test_sample_format_matches_lerobot(self, temp_dataset_dir, sample_config, sample_frames):
        """Test 4: Output format matches LeRobot expectations."""
        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=sample_config["action_dim"],
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,
        )

        writer.add_episode(sample_frames, task="test_task")

        # Access first sample
        sample = writer[0]

        # Verify LeRobot format
        assert "observation.images.left_cam" in sample
        assert "observation.images.right_cam" in sample
        assert "observation.state" in sample
        assert "action" in sample
        assert "task" in sample

        # Verify shapes
        assert sample["observation.state"].shape == (7,)
        assert sample["action"].shape == (7,)

    def test_allows_custom_camera_shapes(self, temp_dataset_dir, sample_config):
        """Writer accepts camera shapes overriding default resolution."""
        samples = []
        for i in range(3):
            img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            cameras = [
                CameraImage(name="left_cam", image=img),
                CameraImage(name="right_cam", image=img.copy()),
            ]
            samples.append(
                Sample(
                    timestamp=1000.0 + i * 0.033,
                    cameras=cameras,
                    proprio=np.random.randn(7).astype(np.float32),
                    action=np.random.randn(7).astype(np.float32),
                    metadata={"frame_idx": i},
                )
            )

        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=sample_config["action_dim"],
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,
            camera_shapes={"left_cam": (720, 1280, 3), "right_cam": (720, 1280, 3)},
        )

        writer.add_episode(samples, task="custom_shape_task")
        sample = writer[0]
        assert sample["observation.images.left_cam"].shape == (3, 720, 1280)
        assert sample["observation.images.right_cam"].shape == (3, 720, 1280)

    def test_consolidate_saves_dataset(self, temp_dataset_dir, sample_config, sample_frames):
        """Test 5: Consolidate saves dataset to disk."""
        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=sample_config["action_dim"],
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,
        )

        writer.add_episode(sample_frames, task="test_task")
        writer.consolidate()

        # Verify dataset files exist
        dataset_path = temp_dataset_dir / sample_config["repo_id"].replace("/", "__")
        assert dataset_path.exists()
        assert (dataset_path / "meta").exists()
        assert (dataset_path / "data").exists()

    def test_reload_saved_dataset(self, temp_dataset_dir, sample_config, sample_frames):
        """Test 6: Can reload dataset after saving."""
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=sample_config["action_dim"],
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,
        )

        writer.add_episode(sample_frames, task="test_task")
        writer.consolidate()

        dataset_root = temp_dataset_dir / sample_config["repo_id"].replace("/", "__")
        from datasets import Dataset

        hf_ds = Dataset.from_parquet(str(dataset_root / "data" / "chunk-000" / "*.parquet"))
        assert len(hf_ds) == len(sample_frames)

    def test_handles_missing_proprio(self, temp_dataset_dir, sample_config):
        """Test 7: Handles samples without proprioception."""
        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=sample_config["action_dim"],
            proprio_dim=0,  # No proprio
            root=temp_dataset_dir,
            use_videos=False,
        )

        # Create sample without proprio
        sample = Sample(
            timestamp=1000.0,
            cameras=[
                CameraImage(name="left_cam", image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
                CameraImage(name="right_cam", image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
            ],
            proprio=None,
            action=np.random.randn(7).astype(np.float32),
        )

        writer.add_episode([sample], task="test")

        # Should not raise error
        assert writer.num_episodes == 1

    def test_validates_camera_names(self, temp_dataset_dir, sample_config):
        """Test 8: Validates camera names match."""
        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=["left_cam"],  # Only one camera expected
            action_dim=sample_config["action_dim"],
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,
        )

        # Create sample with different camera name
        sample = Sample(
            timestamp=1000.0,
            cameras=[CameraImage(name="wrong_cam", image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))],
            action=np.random.randn(7).astype(np.float32),
        )

        with pytest.raises(ValueError, match="Camera.*not in expected"):
            writer.add_episode([sample], task="test")

    def test_validates_action_dimension(self, temp_dataset_dir, sample_config):
        """Test 9: Validates action dimension."""
        writer = LeRobotDatasetWriter(
            repo_id=sample_config["repo_id"],
            robot_type=sample_config["robot_type"],
            fps=sample_config["fps"],
            camera_names=sample_config["camera_names"],
            action_dim=7,  # Expect 7D actions
            proprio_dim=sample_config["proprio_dim"],
            root=temp_dataset_dir,
            use_videos=False,
        )

        # Create sample with wrong action dimension
        sample = Sample(
            timestamp=1000.0,
            cameras=[CameraImage(name="left_cam", image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))],
            proprio=np.random.randn(7).astype(np.float32),
            action=np.random.randn(14).astype(np.float32),  # Wrong dimension
        )

        with pytest.raises(ValueError, match="Action dimension"):
            writer.add_episode([sample], task="test")
