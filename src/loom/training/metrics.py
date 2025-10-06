"""Metrics tracking for training."""

from collections import defaultdict


class MetricsTracker:
    """Track and average metrics over training steps.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update({"loss": 0.5, "acc": 0.8})
        >>> tracker.update({"loss": 0.3, "acc": 0.9})
        >>> tracker.compute()
        {'loss': 0.4, 'acc': 0.85}
        >>> tracker.reset()
    """

    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self._metrics: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        """Update metrics with new values.

        Args:
            metrics: Dictionary of metric names and values
        """
        for key, value in metrics.items():
            self._metrics[key].append(float(value))

    def compute(self) -> dict[str, float]:
        """Compute average of all tracked metrics.

        Returns:
            Dictionary of metric names and averaged values
        """
        return {key: sum(values) / len(values) for key, values in self._metrics.items() if values}

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._metrics.clear()

    def __len__(self) -> int:
        """Return number of updates for first metric."""
        if not self._metrics:
            return 0
        return len(next(iter(self._metrics.values())))
