"""Checkpoint management for model training."""

import json
import logging
from pathlib import Path
from typing import Any

import torch

from loom.training.config import CheckpointConfig

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints with top-K and last-K retention.

    Handles:
    - Saving checkpoints with metadata
    - Loading checkpoints
    - Pruning old checkpoints (keep top-K by metric + last-K)

    Example:
        >>> manager = CheckpointManager(config)
        >>> manager.save(model, optimizer, step=1000, metrics={"eval/loss": 0.5})
        >>> step = manager.load_latest(model, optimizer)
    """

    def __init__(self, config: CheckpointConfig):
        """Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration
        """
        self.config = config
        self.checkpoint_dir = Path(config.dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoints
        self._checkpoints: list[dict[str, Any]] = []
        self._load_checkpoint_index()

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        metrics: dict[str, float],
    ) -> Path:
        """Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            epoch: Current epoch
            metrics: Current metrics

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Update index
        self._checkpoints.append(
            {"path": checkpoint_path, "step": step, "epoch": epoch, "metrics": metrics, "name": checkpoint_name}
        )
        self._save_checkpoint_index()

        # Prune old checkpoints
        self._prune_checkpoints()

        return checkpoint_path

    def load(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """Load checkpoint from path.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Step number from checkpoint

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint path is outside checkpoint directory or is invalid
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Security: Ensure checkpoint is within our checkpoint directory
        try:
            checkpoint_path.resolve().relative_to(self.checkpoint_dir.resolve())
        except ValueError as e:
            raise ValueError(
                f"Checkpoint path {checkpoint_path} is outside checkpoint directory {self.checkpoint_dir}"
            ) from e

        # Use weights_only=True to prevent arbitrary code execution (PyTorch 2.6+)
        try:
            checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions without weights_only
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            logger.warning("Loading checkpoint without weights_only=True. Upgrade to PyTorch 2.6+ for better security.")

        # Validate checkpoint structure
        required_keys = ["step", "model_state_dict"]
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise ValueError(
                f"Checkpoint missing required keys: {missing_keys}. "
                f"Found keys: {list(checkpoint.keys())}. "
                "Checkpoint may be corrupted or from an incompatible version."
            )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        step: int = checkpoint["step"]
        logger.info(f"Loaded checkpoint from {checkpoint_path} (step {step})")

        return step

    def load_latest(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
        """Load most recent checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Step number from checkpoint (0 if no checkpoints)
        """
        if not self._checkpoints:
            logger.info("No checkpoints found, starting from scratch")
            return 0

        latest = max(self._checkpoints, key=lambda x: x["step"])
        return self.load(latest["path"], model, optimizer)

    def _prune_checkpoints(self) -> None:
        """Prune checkpoints keeping top-K by metric and last-K."""
        # Need at least keep_top_k + keep_last_k checkpoints before pruning
        min_checkpoints = max(1, self.config.keep_top_k + self.config.keep_last_k)
        if len(self._checkpoints) <= min_checkpoints:
            return

        # Sort by metric for top-K
        metric_key = self.config.metric_for_best
        reverse = self.config.mode == "max"

        checkpoints_with_metric = [ckpt for ckpt in self._checkpoints if metric_key in ckpt["metrics"]]

        if checkpoints_with_metric:
            top_k = sorted(checkpoints_with_metric, key=lambda x: x["metrics"][metric_key], reverse=reverse)[
                : self.config.keep_top_k
            ]
        else:
            top_k = []

        # Get last-K
        last_k = sorted(self._checkpoints, key=lambda x: x["step"])[-(self.config.keep_last_k) :]

        # Combine and keep unique
        keep = {ckpt["name"]: ckpt for ckpt in top_k + last_k}

        # Delete checkpoints not in keep set
        for ckpt in self._checkpoints:
            if ckpt["name"] not in keep:
                try:
                    ckpt["path"].unlink()
                    logger.info(f"Pruned checkpoint: {ckpt['name']}")
                except FileNotFoundError:
                    pass

        # Update checkpoint list
        self._checkpoints = list(keep.values())
        self._save_checkpoint_index()

    def _save_checkpoint_index(self) -> None:
        """Save checkpoint index to JSON."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        index_data = [
            {
                "name": ckpt["name"],
                "step": ckpt["step"],
                "epoch": ckpt["epoch"],
                "metrics": ckpt["metrics"],
            }
            for ckpt in self._checkpoints
        ]

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

    def _load_checkpoint_index(self) -> None:
        """Load checkpoint index from JSON."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        if not index_path.exists():
            return

        with open(index_path) as f:
            index_data = json.load(f)

        self._checkpoints = [
            {
                "path": self.checkpoint_dir / ckpt["name"],
                "step": ckpt["step"],
                "epoch": ckpt["epoch"],
                "metrics": ckpt["metrics"],
                "name": ckpt["name"],
            }
            for ckpt in index_data
            if (self.checkpoint_dir / ckpt["name"]).exists()
        ]
