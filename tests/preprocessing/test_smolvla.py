"""Tests for SmolVLA preprocessor.

This test suite follows TDD principles and specifies the expected behavior
of the SmolVLAPreprocessor class before implementation.

Test Coverage:
- Initialization and configuration validation
- Single sample preprocessing (images, state, action, language)
- Image preprocessing (resize, normalize, channels-first format)
- State/action normalization and padding
- Multi-camera handling with order preservation
- Batching and collation for PyTorch DataLoader
- Integration with PyTorch training pipeline
- Edge cases and error handling

Robotics-Specific Tests:
- Camera order preservation (critical for coordinate frames)
- Image format validation (RGB, uint8/float32)
- Normalization with dataset statistics
- NaN and inf handling in sensor data
- Variable-length sequences and padding
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from loom.core.types import CameraImage, Sample, SmolVLABatchInput, SmolVLAInput
from loom.preprocessing.config import SmolVLAPreprocessingConfig
from loom.preprocessing.smolvla import SmolVLAPreprocessor


class TestSmolVLAPreprocessorInit:
    """Test SmolVLAPreprocessor initialization."""

    def test_init_with_valid_config(self, basic_smolvla_config):
        """Test preprocessor initializes with valid config."""
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)
        assert preprocessor.config == basic_smolvla_config

    def test_init_loads_tokenizer(self, basic_smolvla_config):
        """Test that tokenizer is loaded during initialization."""
        with patch("loom.preprocessing.smolvla.AutoTokenizer") as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = Mock()
            _ = SmolVLAPreprocessor(basic_smolvla_config)
            mock_tokenizer.from_pretrained.assert_called_once_with(basic_smolvla_config.vlm_model_name)


class TestPreprocessSample:
    """Test single sample preprocessing."""

    def test_preprocess_returns_smolvla_input(self, simple_sample, basic_smolvla_config):
        """Test that preprocess_sample returns SmolVLAInput dataclass."""
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)
        result = preprocessor.preprocess_sample(simple_sample)

        assert isinstance(result, SmolVLAInput)
        assert hasattr(result, "images")
        assert hasattr(result, "language_instruction")
        assert hasattr(result, "state")
        assert hasattr(result, "action")

    def test_preprocess_extracts_correct_cameras(self, simple_sample, basic_smolvla_config):
        """Test that correct cameras are extracted in correct order."""
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)
        result = preprocessor.preprocess_sample(simple_sample)

        # Should have 2 cameras as per config
        assert len(result.images) == 2
        assert all(isinstance(img, torch.Tensor) for img in result.images)

    def test_preprocess_missing_camera_raises_error(self, simple_sample, single_camera_config):
        """Test that missing camera raises ValueError."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        # simple_sample has left_cam and right_cam, but config expects observation.image

        with pytest.raises(ValueError, match="Camera.*not found"):
            preprocessor.preprocess_sample(simple_sample)

    def test_preprocess_missing_task_raises_error(self, sample_without_task, basic_smolvla_config):
        """Test that missing task instruction raises ValueError."""
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)

        with pytest.raises(ValueError, match="task"):
            preprocessor.preprocess_sample(sample_without_task)

    def test_preprocess_extracts_language_instruction(self, simple_sample, basic_smolvla_config):
        """Test that language instruction is extracted from metadata."""
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)
        result = preprocessor.preprocess_sample(simple_sample)

        assert result.language_instruction == "Pick up the red cube"
        assert isinstance(result.language_instruction, str)

    def test_call_method_works(self, simple_sample, basic_smolvla_config):
        """Test that __call__ is equivalent to preprocess_sample."""
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)

        result1 = preprocessor.preprocess_sample(simple_sample)
        result2 = preprocessor(simple_sample)

        assert isinstance(result1, SmolVLAInput)
        assert isinstance(result2, SmolVLAInput)


class TestImagePreprocessing:
    """Test image preprocessing transformations."""

    def test_image_resized_to_target_size(self, single_camera_sample, single_camera_config):
        """Test that images are resized to configured target size."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        # Should be (C, H, W) format
        assert result.images[0].shape == (3, 512, 512)

    def test_image_has_correct_dtype(self, single_camera_sample, single_camera_config):
        """Test that images are converted to float32."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        assert result.images[0].dtype == torch.float32

    def test_image_normalized_with_imagenet_stats(self, single_camera_sample, single_camera_config):
        """Test that images are normalized with ImageNet mean/std."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        # Normalized images should have values roughly in [-3, 3] range
        # (since pixel values are 0-1 after scaling, and we subtract mean ~0.5 and divide by std ~0.2)
        assert result.images[0].min() >= -5.0
        assert result.images[0].max() <= 5.0

    def test_image_channels_first_format(self, single_camera_sample, single_camera_config):
        """Test that images are in channels-first (C, H, W) format."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        # First dimension should be 3 (RGB channels)
        assert result.images[0].shape[0] == 3


class TestStatePreprocessing:
    """Test state (proprioceptive) preprocessing."""

    def test_state_normalized_with_mean_std(self, single_camera_sample, single_camera_config):
        """Test that state is normalized using configured mean/std."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        assert isinstance(result.state, torch.Tensor)
        assert result.state.dtype == torch.float32

    def test_state_padded_to_max_dim(self, single_camera_sample, single_camera_config):
        """Test that state is padded to max_state_dim."""
        # Sample has 7-dim state, should be padded to 32
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        assert result.state.shape == (32,)

    def test_state_padding_values_are_zero(self, single_camera_sample, single_camera_config):
        """Test that padding values are zero."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        # Last 25 values should be zero (32 - 7 = 25)
        assert torch.all(result.state[7:] == 0.0)

    def test_state_normalization_applied_correctly(self, single_camera_sample):
        """Test that (x - mean) / std normalization is applied correctly."""
        config = SmolVLAPreprocessingConfig(
            camera_names=["observation.image"],
            state_mean=[1.0] * 7,  # Mean of 1
            state_std=[2.0] * 7,  # Std of 2
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )
        preprocessor = SmolVLAPreprocessor(config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        # Original: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        # Normalized: [(x - 1.0) / 2.0 for x in original]
        expected = torch.tensor(
            [(x - 1.0) / 2.0 for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]] + [0.0] * 25,
            dtype=torch.float32,
        )
        assert torch.allclose(result.state, expected, atol=1e-5)


class TestActionPreprocessing:
    """Test action preprocessing."""

    def test_action_normalized_with_mean_std(self, single_camera_sample, single_camera_config):
        """Test that action is normalized using configured mean/std."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        assert isinstance(result.action, torch.Tensor)
        assert result.action.dtype == torch.float32

    def test_action_padded_to_max_dim(self, single_camera_sample, single_camera_config):
        """Test that action is padded to max_action_dim."""
        # Sample has 7-dim action, should be padded to 32
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        assert result.action.shape == (32,)

    def test_action_padding_values_are_zero(self, single_camera_sample, single_camera_config):
        """Test that padding values are zero."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)
        result = preprocessor.preprocess_sample(single_camera_sample)

        # Last 25 values should be zero (32 - 7 = 25)
        assert torch.all(result.action[7:] == 0.0)


class TestCollateFn:
    """Test batching/collation functionality."""

    def test_collate_returns_batch_input(self, single_camera_config):
        """Test that collate_fn returns SmolVLABatchInput dataclass."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        # Create batch of 2 preprocessed samples
        batch = [
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Pick up the red cube",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Place the blue box",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
        ]

        result = preprocessor.collate_fn(batch)
        assert isinstance(result, SmolVLABatchInput)

    def test_collate_adds_batch_dimension(self, single_camera_config):
        """Test that collate_fn adds batch dimension to all tensors."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        batch = [
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Pick up the red cube",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Place the blue box",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
        ]

        result = preprocessor.collate_fn(batch)

        # All tensors should have batch size 2 as first dimension
        assert result.observation_images["observation.image"].shape[0] == 2
        assert result.language_tokens.shape[0] == 2
        assert result.language_attention_mask.shape[0] == 2
        assert result.state.shape[0] == 2
        assert result.action.shape[0] == 2

    def test_collate_stacks_images_correctly(self, single_camera_config):
        """Test that images are stacked correctly in batch."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        batch = [
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Task 1",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Task 2",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
        ]

        result = preprocessor.collate_fn(batch)

        # Image should be (B, C, H, W)
        assert result.observation_images["observation.image"].shape == (2, 3, 512, 512)

    def test_collate_handles_multiple_cameras(self, basic_smolvla_config):
        """Test that collate_fn handles multiple cameras correctly."""
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)

        batch = [
            SmolVLAInput(
                images=[torch.rand(3, 512, 512), torch.rand(3, 512, 512)],
                language_instruction="Task 1",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
            SmolVLAInput(
                images=[torch.rand(3, 512, 512), torch.rand(3, 512, 512)],
                language_instruction="Task 2",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
        ]

        result = preprocessor.collate_fn(batch)

        # Should preserve camera names from config: ["left_cam", "right_cam"]
        assert "left_cam" in result.observation_images
        assert "right_cam" in result.observation_images
        assert result.observation_images["left_cam"].shape == (2, 3, 512, 512)
        assert result.observation_images["right_cam"].shape == (2, 3, 512, 512)

    def test_collate_tokenizes_language(self, single_camera_config):
        """Test that language instructions are tokenized in batch."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        batch = [
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Pick up the red cube",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Place the blue box",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
        ]

        result = preprocessor.collate_fn(batch)

        # Should have token IDs and attention mask
        assert result.language_tokens.dtype == torch.long
        assert result.language_attention_mask.dtype == torch.long
        assert result.language_tokens.shape[0] == 2  # Batch size
        assert result.language_attention_mask.shape[0] == 2  # Batch size

    def test_collate_reshapes_state_for_obs_steps(self, single_camera_config):
        """Test that state is reshaped to (B, n_obs_steps, state_dim)."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        batch = [
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Task 1",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Task 2",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
        ]

        result = preprocessor.collate_fn(batch)

        # State should be (B, n_obs_steps, state_dim) = (2, 1, 32)
        assert result.state.shape == (2, 1, 32)

    def test_collate_reshapes_action_for_chunk_size(self, single_camera_config):
        """Test that single action is reshaped to (B, 1, action_dim).

        Since samples contain single actions (not sequences), the collate function
        should reshape them to (B, 1, action_dim) for model compatibility.
        """
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        batch = [
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Task 1",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],
                language_instruction="Task 2",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
        ]

        result = preprocessor.collate_fn(batch)

        # Single actions should be reshaped to (B, 1, action_dim)
        assert result.action.shape == (2, 1, 32)


class TestIntegration:
    """Integration tests with PyTorch DataLoader."""

    def test_works_with_pytorch_dataloader(self, simple_sample, basic_smolvla_config):
        """Test that preprocessor works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader, Dataset

        class SimpleDataset(Dataset):
            def __init__(self, samples, preprocessor):
                self.samples = samples
                self.preprocessor = preprocessor

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                return self.preprocessor.preprocess_sample(self.samples[idx])

        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)
        dataset = SimpleDataset([simple_sample] * 4, preprocessor)
        loader = DataLoader(dataset, batch_size=2, collate_fn=preprocessor.collate_fn)

        # Test that we can iterate through loader
        batch_count = 0
        for batch in loader:
            assert isinstance(batch, SmolVLABatchInput)
            assert batch.observation_images["left_cam"].shape[0] == 2
            batch_count += 1

        assert batch_count == 2  # 4 samples / batch_size 2


class TestRoboticsSpecific:
    """Robotics-specific tests for preprocessing."""

    def test_camera_order_preserved(self, basic_smolvla_config):
        """Test that camera order from config is strictly preserved.

        Critical for multi-camera coordinate frame consistency. If config specifies
        ["left_cam", "right_cam"], output must be in that exact order regardless
        of sample camera order.
        """
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)

        # Create sample with cameras in WRONG order (right before left)
        sample = Sample(
            timestamp=1.0,
            cameras=[
                CameraImage(
                    name="right_cam",
                    image=np.ones((480, 640, 3), dtype=np.uint8) * 255,  # All white
                ),
                CameraImage(
                    name="left_cam",
                    image=np.zeros((480, 640, 3), dtype=np.uint8),  # All black
                ),
            ],
            proprio=np.zeros(7, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            metadata={"task": "test"},
        )

        result = preprocessor.preprocess_sample(sample)

        # Output order must match config order: ["left_cam", "right_cam"]
        # left_cam (black) should be first, right_cam (white) should be second
        assert result.images[0].mean() < 0.5  # left_cam is dark (black)
        assert result.images[1].mean() > 0.5  # right_cam is bright (white)

    def test_image_preprocessing_pipeline_order(self):
        """Test complete image preprocessing pipeline in correct order.

        Verifies: uint8 [0,255] → float32 → /255 → [0,1] → (x-mean)/std → normalized
        """
        # Create deterministic gray image (128 = mid-gray)
        # Use square image to avoid padding effects
        image = np.full((512, 512, 3), 128, dtype=np.uint8)
        sample = Sample(
            timestamp=1.0,
            cameras=[CameraImage(name="observation.image", image=image)],
            proprio=np.zeros(7, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            metadata={"task": "test"},
        )

        # Use config without padding to ensure deterministic values
        from loom.preprocessing.config import ImagePreprocessingConfig

        config = SmolVLAPreprocessingConfig(
            camera_names=["observation.image"],
            image_config=ImagePreprocessingConfig(
                target_size=(512, 512),
                resize_with_padding=False,  # No padding for deterministic test
                normalize=True,
            ),
            state_mean=[0.0] * 7,
            state_std=[1.0] * 7,
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )

        preprocessor = SmolVLAPreprocessor(config)
        result = preprocessor.preprocess_sample(sample)

        # Compute expected values for each channel
        # 128 / 255 = 0.5019...
        # R: (0.5019 - 0.485) / 0.229 ≈ 0.074
        # G: (0.5019 - 0.456) / 0.224 ≈ 0.205
        # B: (0.5019 - 0.406) / 0.225 ≈ 0.426
        expected_r = (128.0 / 255.0 - 0.485) / 0.229
        expected_g = (128.0 / 255.0 - 0.456) / 0.224
        expected_b = (128.0 / 255.0 - 0.406) / 0.225

        # Check channel means are close to expected
        assert torch.allclose(result.images[0][0].mean(), torch.tensor(expected_r), atol=0.01)
        assert torch.allclose(result.images[0][1].mean(), torch.tensor(expected_g), atol=0.01)
        assert torch.allclose(result.images[0][2].mean(), torch.tensor(expected_b), atol=0.01)

    def test_state_with_nan_raises_error(self, single_camera_config):
        """Test that NaN in proprioceptive state raises clear error.

        Robot datasets sometimes have NaN for missing/failed sensors.
        """
        sample = Sample(
            timestamp=1.0,
            cameras=[
                CameraImage(
                    name="observation.image",
                    image=np.zeros((480, 640, 3), dtype=np.uint8),
                )
            ],
            proprio=np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            metadata={"task": "test"},
        )

        preprocessor = SmolVLAPreprocessor(single_camera_config)

        with pytest.raises(ValueError, match="NaN.*proprio|state"):
            preprocessor.preprocess_sample(sample)

    def test_action_with_nan_raises_error(self, single_camera_config):
        """Test that NaN in action raises clear error."""
        sample = Sample(
            timestamp=1.0,
            cameras=[
                CameraImage(
                    name="observation.image",
                    image=np.zeros((480, 640, 3), dtype=np.uint8),
                )
            ],
            proprio=np.zeros(7, dtype=np.float32),
            action=np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
            metadata={"task": "test"},
        )

        preprocessor = SmolVLAPreprocessor(single_camera_config)

        with pytest.raises(ValueError, match="NaN|action"):
            preprocessor.preprocess_sample(sample)


class TestConfigValidation:
    """Test configuration validation."""

    def test_config_with_zero_std_raises_error(self):
        """Test that zero std raises error (prevents division by zero)."""
        config = SmolVLAPreprocessingConfig(
            camera_names=["observation.image"],
            state_mean=[0.0] * 7,
            state_std=[0.0] * 7,  # Invalid: will cause division by zero
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )

        with pytest.raises(ValueError, match="std.*zero"):
            SmolVLAPreprocessor(config)

    def test_config_with_empty_camera_list_raises_error(self):
        """Test that empty camera_names list raises error."""
        config = SmolVLAPreprocessingConfig(
            camera_names=[],  # Invalid
            state_mean=[0.0] * 7,
            state_std=[1.0] * 7,
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )

        with pytest.raises(ValueError, match="camera.*empty"):
            SmolVLAPreprocessor(config)

    def test_config_with_mismatched_mean_std_lengths_raises_error(self):
        """Test that mismatched mean/std lengths raise error."""
        config = SmolVLAPreprocessingConfig(
            camera_names=["observation.image"],
            state_mean=[0.0] * 5,
            state_std=[1.0] * 7,  # Different length
            action_mean=[0.0] * 7,
            action_std=[1.0] * 7,
        )

        with pytest.raises(ValueError, match="mean.*std.*length|mismatch"):
            SmolVLAPreprocessor(config)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_batch_raises_error(self, single_camera_config):
        """Test that empty batch raises appropriate error."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        with pytest.raises((ValueError, IndexError)):
            preprocessor.collate_fn([])

    def test_mismatched_camera_counts_in_batch(self, basic_smolvla_config):
        """Test that mismatched camera counts raise error."""
        preprocessor = SmolVLAPreprocessor(basic_smolvla_config)

        batch = [
            SmolVLAInput(
                images=[torch.rand(3, 512, 512)],  # 1 camera
                language_instruction="Task 1",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
            SmolVLAInput(
                images=[torch.rand(3, 512, 512), torch.rand(3, 512, 512)],  # 2 cameras
                language_instruction="Task 2",
                state=torch.rand(32),
                action=torch.rand(32),
            ),
        ]

        with pytest.raises(ValueError, match="camera"):
            preprocessor.collate_fn(batch)

    def test_sample_without_proprio_raises_error(self, single_camera_config):
        """Test that sample without proprio raises error."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        sample = Sample(
            timestamp=1000.0,
            cameras=[
                CameraImage(
                    name="observation.image",
                    image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                )
            ],
            proprio=None,  # Missing proprio
            action=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            metadata={"task": "Pick up the red cube"},
        )

        with pytest.raises(ValueError):
            preprocessor.preprocess_sample(sample)

    def test_sample_without_action_raises_error(self, single_camera_config):
        """Test that sample without action raises error."""
        preprocessor = SmolVLAPreprocessor(single_camera_config)

        sample = Sample(
            timestamp=1000.0,
            cameras=[
                CameraImage(
                    name="observation.image",
                    image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                )
            ],
            proprio=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            action=None,  # Missing action
            metadata={"task": "Pick up the red cube"},
        )

        with pytest.raises(ValueError):
            preprocessor.preprocess_sample(sample)
