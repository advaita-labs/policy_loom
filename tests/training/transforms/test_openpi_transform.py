"""Tests for OpenPI data transformation."""

import numpy as np
import pytest
import torch

from loom.training.transforms.openpi_transform import OpenPITransform


class TestOpenPITransform:
    """Test OpenPI transformation from LeRobot format."""

    def test_basic_transformation(self):
        """Test basic transformation without images or prompts."""
        transform = OpenPITransform()

        # Create simple batch
        batch = {
            "observation": torch.randn(4, 8),  # (B=4, state_dim=8)
            "action": torch.randn(4, 7),  # (B=4, action_dim=7)
            "images": [],  # No images
        }

        obs_dict, actions = transform(batch)

        # Check state is passed through
        assert "state" in obs_dict
        assert obs_dict["state"].shape == (4, 8)
        assert torch.allclose(obs_dict["state"], batch["observation"].float())

        # Check actions
        assert actions.shape == (4, 7)
        assert torch.allclose(actions, batch["action"])

    def test_image_transformation(self):
        """Test image normalization from uint8 to float32[-1,1]."""
        transform = OpenPITransform(image_size=(224, 224))

        # Create batch with images
        images_list = [
            {"cam0": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)},
            {"cam0": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)},
        ]

        batch = {
            "observation": torch.randn(2, 8),
            "action": torch.randn(2, 7),
            "images": images_list,
        }

        obs_dict, actions = transform(batch)

        # Check images are present
        assert "images" in obs_dict
        assert "image_masks" in obs_dict
        assert "cam0" in obs_dict["images"]

        # Check normalization
        img_tensor = obs_dict["images"]["cam0"]
        assert img_tensor.shape == (2, 224, 224, 3)
        assert img_tensor.dtype == torch.float32
        assert img_tensor.min() >= -1.0
        assert img_tensor.max() <= 1.0

        # Check masks
        assert obs_dict["image_masks"]["cam0"].shape == (2,)
        assert obs_dict["image_masks"]["cam0"].all()

    def test_multiple_cameras(self):
        """Test transformation with multiple cameras."""
        transform = OpenPITransform(image_size=(224, 224))

        images_list = [
            {
                "cam0": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
                "cam1": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            },
            {
                "cam0": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
                "cam1": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            },
        ]

        batch = {
            "observation": torch.randn(2, 8),
            "action": torch.randn(2, 7),
            "images": images_list,
        }

        obs_dict, actions = transform(batch)

        # Check both cameras
        assert "cam0" in obs_dict["images"]
        assert "cam1" in obs_dict["images"]
        assert "cam0" in obs_dict["image_masks"]
        assert "cam1" in obs_dict["image_masks"]

        # Check shapes
        assert obs_dict["images"]["cam0"].shape == (2, 224, 224, 3)
        assert obs_dict["images"]["cam1"].shape == (2, 224, 224, 3)

    def test_missing_image_handling(self):
        """Test handling of missing images in some samples."""
        transform = OpenPITransform(image_size=(224, 224))

        images_list = [
            {"cam0": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)},
            {},  # Missing camera in second sample
        ]

        batch = {
            "observation": torch.randn(2, 8),
            "action": torch.randn(2, 7),
            "images": images_list,
        }

        obs_dict, actions = transform(batch)

        # Should still work, creating black image for missing sample
        assert "images" in obs_dict or "state" in obs_dict
        assert actions.shape == (2, 7)

    def test_action_validation(self):
        """Test that missing action raises error."""
        transform = OpenPITransform()

        batch = {
            "observation": torch.randn(2, 8),
            # Missing action!
        }

        with pytest.raises(ValueError, match="action"):
            transform(batch)

    def test_numpy_to_tensor_conversion(self):
        """Test conversion from numpy arrays to tensors."""
        transform = OpenPITransform()

        batch = {
            "observation": np.random.randn(2, 8).astype(np.float32),
            "action": np.random.randn(2, 7).astype(np.float32),
            "images": [],
        }

        obs_dict, actions = transform(batch)

        # Should be converted to tensors
        assert isinstance(obs_dict["state"], torch.Tensor)
        assert isinstance(actions, torch.Tensor)
        assert obs_dict["state"].dtype == torch.float32
        assert actions.dtype == torch.float32
