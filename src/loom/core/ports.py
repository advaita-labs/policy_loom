"""Abstract base classes defining the contracts for pipeline components."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loom.core.types import Sample


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


class Writer(ABC):
    """Abstract base class for writing processed data.

    A Writer is responsible for:
    - Creating output directories/files
    - Writing samples to disk in the target format
    - Ensuring atomic writes and proper cleanup
    - Generating manifests/metadata
    """

    @abstractmethod
    def write(self, sample: Sample) -> None:
        """Write a single sample to the output.

        Args:
            sample: Sample to write

        Raises:
            IOError: If write fails
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Finalize writes and release resources.

        This should:
        - Flush any buffered data
        - Write manifests/metadata
        - Close file handles
        - Perform atomic renames if using temp files
        """
        ...

    def __enter__(self) -> "Writer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


class Exporter(ABC):
    """Abstract base class for export format handlers.

    An Exporter defines:
    - How to map canonical Sample → target schema
    - Directory layout conventions
    - File format specifications
    """

    @abstractmethod
    def create_writer(self, output_path: Path, **kwargs: Any) -> Writer:
        """Create a writer for this export format.

        Args:
            output_path: Root directory for exported data
            **kwargs: Format-specific configuration

        Returns:
            A Writer instance configured for this export format
        """
        ...

    @abstractmethod
    def validate_output(self, output_path: Path) -> bool:
        """Validate that exported data conforms to the schema.

        Args:
            output_path: Root directory of exported data

        Returns:
            True if valid, False otherwise
        """
        ...
