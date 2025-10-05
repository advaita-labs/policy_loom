"""Base preprocessor with reusable utilities for VLA models."""

import numpy as np
import numpy.typing as npt
from PIL import Image


class BasePreprocessor:
    """Base class with common preprocessing utilities.

    Provides reusable methods for:
    - Normalization (mean/std, min/max)
    - Image preprocessing (resize, normalize, to_tensor)
    - Padding
    - Validation
    """

    @staticmethod
    def normalize_mean_std(
        data: npt.NDArray[np.float32],
        mean: npt.NDArray[np.float32],
        std: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Z-score normalization: (x - mean) / std.

        Args:
            data: Input data to normalize
            mean: Mean for each dimension
            std: Standard deviation for each dimension

        Returns:
            Normalized data

        Raises:
            ValueError: If std contains zeros
        """
        if np.any(std == 0):
            raise ValueError("Standard deviation contains zeros, cannot normalize")

        return (data - mean) / std

    @staticmethod
    def pad_vector(
        vec: npt.NDArray[np.float32],
        target_dim: int,
        pad_value: float = 0.0,
    ) -> npt.NDArray[np.float32]:
        """Pad 1D vector to target dimension.

        Args:
            vec: Input vector
            target_dim: Target dimension
            pad_value: Value to use for padding

        Returns:
            Padded vector

        Raises:
            ValueError: If vec is longer than target_dim
        """
        if vec.shape[0] > target_dim:
            raise ValueError(f"Vector dimension {vec.shape[0]} exceeds target dimension {target_dim}")

        if vec.shape[0] == target_dim:
            return vec

        padded = np.full(target_dim, pad_value, dtype=np.float32)
        padded[: vec.shape[0]] = vec
        return padded

    @staticmethod
    def resize_image_with_padding(
        image: npt.NDArray[np.uint8] | npt.NDArray[np.float32],
        target_size: tuple[int, int],
        pad_value: int | float = 0,
    ) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
        """Resize image maintaining aspect ratio with padding (letterbox).

        Args:
            image: Input image (H, W, C)
            target_size: Target size (height, width)
            pad_value: Value to use for padding

        Returns:
            Resized and padded image

        Raises:
            ValueError: If image shape is invalid
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {image.shape}")

        h, w, c = image.shape
        target_h, target_w = target_size

        # Calculate scaling factor to fit image in target size
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        pil_image = Image.fromarray(image)
        resized = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        resized_np = np.array(resized)

        # Create padded canvas
        if image.dtype == np.uint8:
            padded = np.full((target_h, target_w, c), pad_value, dtype=np.uint8)
        else:
            padded = np.full((target_h, target_w, c), pad_value, dtype=np.float32)

        # Calculate padding offsets (center the image)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        # Place resized image in center
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_np

        return padded

    @staticmethod
    def resize_image_distort(
        image: npt.NDArray[np.uint8] | npt.NDArray[np.float32],
        target_size: tuple[int, int],
    ) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
        """Resize image by distorting to exact target size.

        Args:
            image: Input image (H, W, C)
            target_size: Target size (height, width)

        Returns:
            Resized image (may be distorted)

        Raises:
            ValueError: If image shape is invalid
        """
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (H, W, C), got shape {image.shape}")

        target_h, target_w = target_size

        pil_image = Image.fromarray(image)
        resized = pil_image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.array(resized)

    @staticmethod
    def validate_no_nan(data: npt.NDArray[np.float32], name: str) -> None:
        """Validate that data contains no NaN values.

        Args:
            data: Data to validate
            name: Name of data (for error message)

        Raises:
            ValueError: If data contains NaN
        """
        if np.isnan(data).any():
            raise ValueError(f"{name} contains NaN values")

    @staticmethod
    def validate_no_inf(data: npt.NDArray[np.float32], name: str) -> None:
        """Validate that data contains no infinite values.

        Args:
            data: Data to validate
            name: Name of data (for error message)

        Raises:
            ValueError: If data contains inf
        """
        if np.isinf(data).any():
            raise ValueError(f"{name} contains infinite values")
