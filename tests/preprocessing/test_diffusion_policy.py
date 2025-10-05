"""Tests for Diffusion Policy preprocessing."""

import numpy as np
import pytest
import torch

from loom.core.types import CameraImage, DiffusionPolicyBatchInput, DiffusionPolicyInput, Sample
from loom.preprocessing.config import DiffusionPolicyPreprocessingConfig, ImagePreprocessingConfig
from loom.preprocessing.diffusion_policy import DiffusionPolicyPreprocessor


class TestDiffusionPolicyPreprocessorInit:
    """Test preprocessor initialization and configuration validation."""

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = DiffusionPolicyPreprocessingConfig(
            camera_names=["cam1", "cam2"],
            obs_horizon=2,
            action_horizon=4,
            state_mean=[0.0] * 7,
            state_std=[1.0] * 7,
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)
        assert preprocessor.config == config

    def test_init_validates_camera_list_not_empty(self):
        """Test that initialization fails if camera list is empty."""
        config = DiffusionPolicyPreprocessingConfig(camera_names=[])
        with pytest.raises(ValueError, match="camera_names cannot be empty"):
            DiffusionPolicyPreprocessor(config)

    def test_init_validates_obs_horizon_positive(self):
        """Test that obs_horizon must be positive."""
        config = DiffusionPolicyPreprocessingConfig(obs_horizon=0)
        with pytest.raises(ValueError, match="obs_horizon must be positive"):
            DiffusionPolicyPreprocessor(config)

    def test_init_validates_action_horizon_positive(self):
        """Test that action_horizon must be positive."""
        config = DiffusionPolicyPreprocessingConfig(action_horizon=0)
        with pytest.raises(ValueError, match="action_horizon must be positive"):
            DiffusionPolicyPreprocessor(config)

    def test_init_validates_state_mean_std_dimensions_match(self):
        """Test that state mean and std have matching dimensions."""
        config = DiffusionPolicyPreprocessingConfig(
            state_mean=[0.0] * 7,
            state_std=[1.0] * 5,  # Mismatch
        )
        with pytest.raises(ValueError, match="state_mean.*state_std"):
            DiffusionPolicyPreprocessor(config)

    def test_init_validates_action_mean_std_dimensions_match(self):
        """Test that action mean and std have matching dimensions."""
        config = DiffusionPolicyPreprocessingConfig(
            action_mean=[0.0] * 7,
            action_std=[1.0] * 5,  # Mismatch
        )
        with pytest.raises(ValueError, match="action_mean.*action_std"):
            DiffusionPolicyPreprocessor(config)


class TestPreprocessSampleSequence:
    """Test preprocessing of sample sequences."""

    def test_preprocess_sample_sequence_basic(self):
        """Test basic preprocessing of sample sequence."""
        config = DiffusionPolicyPreprocessingConfig(
            camera_names=["cam1"],
            obs_horizon=2,
            action_horizon=4,
            state_mean=[0.0] * 7,
            state_std=[1.0] * 7,
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        # Create sequence of samples
        samples = []
        for i in range(6):  # obs_horizon + action_horizon
            sample = Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="cam1", image=np.ones((480, 640, 3), dtype=np.uint8) * (i + 1) * 10)],
                proprio=np.random.randn(7).astype(np.float32),
                action=np.random.randn(7).astype(np.float32),
            )
            samples.append(sample)

        result = preprocessor.preprocess_sample_sequence(samples)

        assert isinstance(result, DiffusionPolicyInput)
        assert "cam1" in result.observation_images
        assert result.observation_images["cam1"].shape == (2, 3, 96, 96)  # obs_horizon, C, H, W
        assert result.state.shape == (2, 7)  # obs_horizon, state_dim
        assert result.action.shape == (4, 7)  # action_horizon, action_dim

    def test_preprocess_requires_sufficient_samples(self):
        """Test that preprocessing fails with insufficient samples."""
        config = DiffusionPolicyPreprocessingConfig(obs_horizon=2, action_horizon=4)
        preprocessor = DiffusionPolicyPreprocessor(config)

        # Only 3 samples, but need obs_horizon + action_horizon = 6
        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.zeros(7, dtype=np.float32),
                action=np.zeros(7, dtype=np.float32),
            )
            for i in range(3)
        ]

        with pytest.raises(ValueError, match="Need at least.*samples"):
            preprocessor.preprocess_sample_sequence(samples)

    def test_preprocess_validates_missing_camera(self):
        """Test that preprocessing fails if required camera is missing."""
        config = DiffusionPolicyPreprocessingConfig(
            camera_names=["cam1", "cam2"],
            obs_horizon=2,
            action_horizon=2,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        # Only cam1, missing cam2
        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="cam1", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.zeros(7, dtype=np.float32),
                action=np.zeros(7, dtype=np.float32),
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError, match="Camera.*not found"):
            preprocessor.preprocess_sample_sequence(samples)

    def test_preprocess_validates_missing_proprio(self):
        """Test that preprocessing fails if proprio is missing."""
        config = DiffusionPolicyPreprocessingConfig(obs_horizon=2, action_horizon=2)
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=None,  # Missing
                action=np.zeros(7, dtype=np.float32),
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError, match="proprio.*state"):
            preprocessor.preprocess_sample_sequence(samples)

    def test_preprocess_validates_missing_action(self):
        """Test that preprocessing fails if action is missing."""
        config = DiffusionPolicyPreprocessingConfig(obs_horizon=2, action_horizon=2)
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.zeros(7, dtype=np.float32),
                action=None,  # Missing
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError, match="action"):
            preprocessor.preprocess_sample_sequence(samples)


class TestImagePreprocessing:
    """Test image preprocessing."""

    def test_image_preprocessing_pipeline(self):
        """Test complete image preprocessing pipeline."""
        config = DiffusionPolicyPreprocessingConfig(
            image_config=ImagePreprocessingConfig(
                target_size=(96, 96),
                resize_with_padding=False,
                normalize=False,
            ),
            obs_horizon=2,
            action_horizon=2,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.full((480, 640, 3), 128, dtype=np.uint8))],
                proprio=np.zeros(7, dtype=np.float32),
                action=np.zeros(7, dtype=np.float32),
            )
            for i in range(4)
        ]

        result = preprocessor.preprocess_sample_sequence(samples)
        img_tensor = result.observation_images["observation.image"]

        # Check shape: (obs_horizon, C, H, W)
        assert img_tensor.shape == (2, 3, 96, 96)

        # Check normalized to [0, 1]
        assert img_tensor.min() >= 0.0
        assert img_tensor.max() <= 1.0
        assert torch.allclose(img_tensor[0], torch.tensor(128.0 / 255.0), atol=0.01)

    def test_image_channels_first_format(self):
        """Test that images are converted to channels-first format."""
        config = DiffusionPolicyPreprocessingConfig(obs_horizon=1, action_horizon=1)
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.zeros(7, dtype=np.float32),
                action=np.zeros(7, dtype=np.float32),
            )
            for i in range(2)
        ]

        result = preprocessor.preprocess_sample_sequence(samples)
        img_tensor = result.observation_images["observation.image"]

        # Shape should be (obs_horizon, C, H, W) not (obs_horizon, H, W, C)
        assert img_tensor.shape[1] == 3  # Channels
        assert img_tensor.shape[2:] == (96, 96)  # Height, Width


class TestStatePreprocessing:
    """Test state/proprio preprocessing."""

    def test_state_normalization_mean_std(self):
        """Test state normalization with mean and std."""
        state_mean = [1.0, 2.0, 3.0]
        state_std = [0.5, 0.5, 0.5]
        config = DiffusionPolicyPreprocessingConfig(
            obs_horizon=2,
            action_horizon=2,
            state_mean=state_mean,
            state_std=state_std,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        test_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=test_state.copy(),
                action=np.zeros(3, dtype=np.float32),
            )
            for i in range(4)
        ]

        result = preprocessor.preprocess_sample_sequence(samples)

        # (state - mean) / std = (1-1)/0.5 = 0, (2-2)/0.5 = 0, (3-3)/0.5 = 0
        expected = torch.zeros(2, 3)
        assert torch.allclose(result.state, expected, atol=0.01)

    def test_state_stacking_preserves_temporal_order(self):
        """Test that state stacking preserves temporal order."""
        config = DiffusionPolicyPreprocessingConfig(
            obs_horizon=3,
            action_horizon=2,
            state_mean=[0.0] * 7,
            state_std=[1.0] * 7,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = []
        for i in range(5):
            sample = Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.full(7, float(i), dtype=np.float32),  # State = [i, i, i, ...]
                action=np.zeros(7, dtype=np.float32),
            )
            samples.append(sample)

        result = preprocessor.preprocess_sample_sequence(samples)

        # Should have states [0, 1, 2] stacked
        assert result.state.shape == (3, 7)
        # First timestep should be 0, last should be 2
        assert torch.allclose(result.state[0], torch.zeros(7))
        assert torch.allclose(result.state[-1], torch.full((7,), 2.0))


class TestActionPreprocessing:
    """Test action preprocessing."""

    def test_action_normalization_mean_std(self):
        """Test action normalization with mean and std."""
        action_mean = [0.0, 0.0, 0.0]
        action_std = [2.0, 2.0, 2.0]
        config = DiffusionPolicyPreprocessingConfig(
            obs_horizon=2,
            action_horizon=3,
            action_mean=action_mean,
            action_std=action_std,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = []
        for i in range(5):
            sample = Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.zeros(3, dtype=np.float32),
                action=np.array([2.0, 4.0, 6.0], dtype=np.float32),
            )
            samples.append(sample)

        result = preprocessor.preprocess_sample_sequence(samples)

        # (action - mean) / std = ([2,4,6] - [0,0,0]) / [2,2,2] = [1,2,3]
        expected = torch.tensor([[1.0, 2.0, 3.0]] * 3)  # action_horizon=3
        assert torch.allclose(result.action, expected, atol=0.01)

    def test_action_chunking_preserves_temporal_order(self):
        """Test that action chunking preserves temporal order and starts at current timestep."""
        config = DiffusionPolicyPreprocessingConfig(
            obs_horizon=2,
            action_horizon=4,
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = []
        for i in range(6):
            sample = Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.zeros(7, dtype=np.float32),
                action=np.full(7, float(i), dtype=np.float32),  # Action = [i, i, i, ...]
            )
            samples.append(sample)

        result = preprocessor.preprocess_sample_sequence(samples)

        # Observations: [t0, t1]
        # Action chunk should start at current timestep t1: [t1, t2, t3, t4]
        assert result.action.shape == (4, 7)
        assert torch.allclose(result.action[0], torch.full((7,), 1.0))  # Action from t1 (current timestep)
        assert torch.allclose(result.action[-1], torch.full((7,), 4.0))  # Action from t4


class TestCollateFn:
    """Test batch collation."""

    def test_collate_single_camera(self):
        """Test collation with single camera."""
        config = DiffusionPolicyPreprocessingConfig(
            camera_names=["cam1"],
            obs_horizon=2,
            action_horizon=4,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        # Create batch of DiffusionPolicyInput
        batch = []
        for _ in range(3):
            item = DiffusionPolicyInput(
                observation_images={"cam1": torch.randn(2, 3, 96, 96)},
                state=torch.randn(2, 7),
                action=torch.randn(4, 7),
            )
            batch.append(item)

        result = preprocessor.collate_fn(batch)

        assert isinstance(result, DiffusionPolicyBatchInput)
        assert "cam1" in result.observation_images
        assert result.observation_images["cam1"].shape == (3, 2, 3, 96, 96)  # B, obs_horizon, C, H, W
        assert result.state.shape == (3, 2, 7)  # B, obs_horizon, state_dim
        assert result.action.shape == (3, 4, 7)  # B, action_horizon, action_dim

    def test_collate_multi_camera(self):
        """Test collation with multiple cameras."""
        config = DiffusionPolicyPreprocessingConfig(
            camera_names=["cam1", "cam2", "cam3"],
            obs_horizon=2,
            action_horizon=4,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        batch = []
        for _ in range(2):
            item = DiffusionPolicyInput(
                observation_images={
                    "cam1": torch.randn(2, 3, 96, 96),
                    "cam2": torch.randn(2, 3, 96, 96),
                    "cam3": torch.randn(2, 3, 96, 96),
                },
                state=torch.randn(2, 7),
                action=torch.randn(4, 7),
            )
            batch.append(item)

        result = preprocessor.collate_fn(batch)

        assert len(result.observation_images) == 3
        for cam in ["cam1", "cam2", "cam3"]:
            assert result.observation_images[cam].shape == (2, 2, 3, 96, 96)

    def test_collate_empty_batch_raises_error(self):
        """Test that collating empty batch raises error."""
        config = DiffusionPolicyPreprocessingConfig()
        preprocessor = DiffusionPolicyPreprocessor(config)

        with pytest.raises(ValueError, match="empty batch"):
            preprocessor.collate_fn([])

    def test_collate_inconsistent_camera_counts_raises_error(self):
        """Test that inconsistent camera counts raise error."""
        config = DiffusionPolicyPreprocessingConfig(obs_horizon=2, action_horizon=2)
        preprocessor = DiffusionPolicyPreprocessor(config)

        batch = [
            DiffusionPolicyInput(
                observation_images={"cam1": torch.randn(2, 3, 96, 96)},
                state=torch.randn(2, 7),
                action=torch.randn(2, 7),
            ),
            DiffusionPolicyInput(
                observation_images={
                    "cam1": torch.randn(2, 3, 96, 96),
                    "cam2": torch.randn(2, 3, 96, 96),  # Extra camera
                },
                state=torch.randn(2, 7),
                action=torch.randn(2, 7),
            ),
        ]

        with pytest.raises(ValueError, match="Inconsistent camera counts"):
            preprocessor.collate_fn(batch)


class TestIntegration:
    """Integration tests with realistic workflows."""

    def test_full_pipeline_with_dataloader(self):
        """Test full pipeline with PyTorch DataLoader."""
        from torch.utils.data import DataLoader, Dataset

        class SampleSequenceDataset(Dataset):
            def __init__(self, sequences, preprocessor):
                self.sequences = sequences
                self.preprocessor = preprocessor

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx):
                return self.preprocessor.preprocess_sample_sequence(self.sequences[idx])

        config = DiffusionPolicyPreprocessingConfig(
            camera_names=["cam1"],
            obs_horizon=2,
            action_horizon=4,
            state_mean=[0.0] * 7,
            state_std=[1.0] * 7,
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )
        preprocessor = DiffusionPolicyPreprocessor(config)

        # Create sample sequences
        sequences = []
        for _ in range(10):
            sequence = []
            for i in range(6):  # obs_horizon + action_horizon
                sample = Sample(
                    timestamp=i * 0.1,
                    cameras=[CameraImage(name="cam1", image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))],
                    proprio=np.random.randn(7).astype(np.float32),
                    action=np.random.randn(7).astype(np.float32),
                )
                sequence.append(sample)
            sequences.append(sequence)

        dataset = SampleSequenceDataset(sequences, preprocessor)
        loader = DataLoader(dataset, batch_size=4, collate_fn=preprocessor.collate_fn)

        # Test loading
        for batch in loader:
            assert isinstance(batch, DiffusionPolicyBatchInput)
            assert "cam1" in batch.observation_images
            assert batch.observation_images["cam1"].shape[0] <= 4  # batch_size
            assert batch.state.shape[1] == 2  # obs_horizon
            assert batch.action.shape[1] == 4  # action_horizon
            break  # Just test first batch


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_nan_in_state_raises_error(self):
        """Test that NaN in state raises error."""
        config = DiffusionPolicyPreprocessingConfig(obs_horizon=2, action_horizon=2)
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.array([1.0, np.nan, 3.0], dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError, match="NaN"):
            preprocessor.preprocess_sample_sequence(samples)

    def test_inf_in_action_raises_error(self):
        """Test that inf in action raises error."""
        config = DiffusionPolicyPreprocessingConfig(obs_horizon=2, action_horizon=2)
        preprocessor = DiffusionPolicyPreprocessor(config)

        samples = [
            Sample(
                timestamp=i * 0.1,
                cameras=[CameraImage(name="observation.image", image=np.zeros((480, 640, 3), dtype=np.uint8))],
                proprio=np.zeros(3, dtype=np.float32),
                action=np.array([1.0, np.inf, 3.0], dtype=np.float32),
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError, match="inf"):
            preprocessor.preprocess_sample_sequence(samples)
