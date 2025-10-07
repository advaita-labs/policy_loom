"""Abstract base classes defining the contracts for pipeline components."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from loom.core.types import Sample

# Type variables for generic Preprocessor
TInput = TypeVar("TInput")  # Single sample input type
TBatchInput = TypeVar("TBatchInput")  # Batched input type


class Reader(ABC):
    """Abstract base class for data ingest.

    A Reader is responsible for:
    - Opening a data source (mp4, mcap, directory of images, etc.)
    - Yielding a stream of Sample objects
    - Managing resources (file handles, connections)
    """

    @abstractmethod
    def read(self) -> Iterator[Sample]:
        """Yield samples from the data source.

        Yields:
            Sample objects in temporal order

        Raises:
            IOError: If the data source cannot be read
            ValueError: If the data format is invalid
        """
        ...

    def __enter__(self) -> "Reader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Release resources. Override if needed."""
        return


class Transform(ABC):
    """Abstract base class for stateless sample transformations.

    A Transform:
    - Takes a Sample and returns a modified Sample
    - Is stateless (no memory between calls)
    - Preserves temporal order unless explicitly documented otherwise
    """

    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        """Apply transformation to a sample.

        Args:
            sample: Input sample

        Returns:
            Transformed sample

        Raises:
            ValueError: If sample cannot be transformed
        """
        ...


class Preprocessor(ABC, Generic[TInput, TBatchInput]):
    """Abstract base class for model-specific preprocessing.

    A Preprocessor is responsible for:
    - Converting Sample objects to model-specific input format
    - Handling tokenization, normalization, padding
    - Batching and collation for DataLoader
    - Managing model-specific state (tokenizers, normalization stats)

    Type Parameters:
        TInput: Type of single sample input (unbatched)
        TBatchInput: Type of batched input for model

    Example:
        >>> from loom.preprocessing.smolvla import SmolVLAPreprocessor
        >>> from loom.core.types import SmolVLAInput, SmolVLABatchInput
        >>>
        >>> preprocessor: Preprocessor[SmolVLAInput, SmolVLABatchInput]
        >>> preprocessor = SmolVLAPreprocessor(config)
    """

    @abstractmethod
    def preprocess_sample(self, sample: Sample) -> TInput:
        """Convert a single Sample to model input format (unbatched).

        Args:
            sample: Input sample with cameras, proprio, action, metadata

        Returns:
            Model-specific input dataclass without batch dimension

        Raises:
            ValueError: If sample is missing required fields
        """
        ...

    @abstractmethod
    def collate_fn(self, batch: list[TInput]) -> TBatchInput:
        """Collate list of preprocessed samples into batched model input.

        This is designed to be used as collate_fn in PyTorch DataLoader.

        Args:
            batch: List of preprocessed inputs from preprocess_sample()

        Returns:
            Batched model input dataclass with all tensors having batch dimension

        Raises:
            ValueError: If batch contains inconsistent shapes
        """
        ...

    def __call__(self, sample: Sample) -> TInput:
        """Convenience method. Equivalent to preprocess_sample()."""
        return self.preprocess_sample(sample)
