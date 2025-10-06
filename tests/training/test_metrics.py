"""Tests for metrics tracking."""

import pytest

from loom.training.metrics import MetricsTracker


class TestMetricsTracker:
    """Test MetricsTracker class."""

    def test_initialization(self):
        """Test tracker initializes empty."""
        tracker = MetricsTracker()
        assert len(tracker) == 0
        assert tracker.compute() == {}

    def test_single_update(self):
        """Test updating with single metrics."""
        tracker = MetricsTracker()
        tracker.update({"loss": 0.5, "accuracy": 0.8})

        assert len(tracker) == 1
        metrics = tracker.compute()
        assert metrics["loss"] == 0.5
        assert metrics["accuracy"] == 0.8

    def test_multiple_updates(self):
        """Test averaging over multiple updates."""
        tracker = MetricsTracker()
        tracker.update({"loss": 1.0, "accuracy": 0.6})
        tracker.update({"loss": 0.5, "accuracy": 0.8})
        tracker.update({"loss": 0.5, "accuracy": 0.9})

        assert len(tracker) == 3
        metrics = tracker.compute()
        assert metrics["loss"] == pytest.approx(0.6667, abs=0.001)
        assert metrics["accuracy"] == pytest.approx(0.7667, abs=0.001)

    def test_reset(self):
        """Test resetting tracker."""
        tracker = MetricsTracker()
        tracker.update({"loss": 0.5})
        tracker.update({"loss": 0.3})

        assert len(tracker) == 2

        tracker.reset()

        assert len(tracker) == 0
        assert tracker.compute() == {}

    def test_different_metrics_per_update(self):
        """Test handling different metrics in different updates."""
        tracker = MetricsTracker()
        tracker.update({"loss": 0.5, "accuracy": 0.8})
        tracker.update({"loss": 0.3})  # No accuracy
        tracker.update({"accuracy": 0.9})  # No loss

        metrics = tracker.compute()
        assert metrics["loss"] == pytest.approx(0.4)  # (0.5 + 0.3) / 2
        assert metrics["accuracy"] == pytest.approx(0.85)  # (0.8 + 0.9) / 2

    def test_integer_values(self):
        """Test handling integer metric values."""
        tracker = MetricsTracker()
        tracker.update({"epoch": 1})
        tracker.update({"epoch": 2})

        metrics = tracker.compute()
        assert metrics["epoch"] == 1.5

    def test_empty_compute(self):
        """Test computing metrics when no updates."""
        tracker = MetricsTracker()
        assert tracker.compute() == {}

    def test_compute_after_reset(self):
        """Test computing after reset gives correct new values."""
        tracker = MetricsTracker()
        tracker.update({"loss": 1.0})
        tracker.reset()
        tracker.update({"loss": 0.5})

        metrics = tracker.compute()
        assert metrics["loss"] == 0.5
