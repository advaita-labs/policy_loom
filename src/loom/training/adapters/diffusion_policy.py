"""Diffusion Policy model adapter for training."""

import logging
from typing import Any

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset

from loom.training.adapter import register_adapter

logger = logging.getLogger(__name__)


class DiffusionPolicyUNet(nn.Module):
    """Simple UNet for Diffusion Policy.

    Predicts noise in action space conditioned on observations.

    Args:
        obs_dim: Observation dimension (flattened images + state)
        action_dim: Action dimension
        action_horizon: Number of future actions to predict
        hidden_dim: Hidden dimension size
    """

    def __init__(self, obs_dim: int, action_dim: int, action_horizon: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Encoder for observations
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Time embedding for diffusion timestep
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Noise prediction network
        self.noise_pred = nn.Sequential(
            nn.Linear(action_horizon * action_dim + hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_horizon * action_dim),
        )

    def forward(self, noisy_actions: torch.Tensor, obs: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy_actions: Noisy actions, shape (B, action_horizon, action_dim)
            obs: Observations, shape (B, obs_dim)
            timestep: Diffusion timestep, shape (B,)

        Returns:
            Predicted noise, shape (B, action_horizon, action_dim)
        """
        batch_size = noisy_actions.shape[0]

        # Encode observations
        obs_embed = self.obs_encoder(obs)  # (B, hidden_dim)

        # Embed timestep
        t_embed = self.time_embed(timestep.unsqueeze(-1).float())  # (B, hidden_dim)

        # Flatten noisy actions
        noisy_actions_flat = noisy_actions.reshape(batch_size, -1)  # (B, action_horizon * action_dim)

        # Concatenate all inputs
        x = torch.cat([noisy_actions_flat, obs_embed, t_embed], dim=-1)

        # Predict noise
        noise = self.noise_pred(x)

        # Reshape to (B, action_horizon, action_dim)
        noise = noise.reshape(batch_size, self.action_horizon, self.action_dim)

        return noise


@register_adapter("diffusion_policy")
class DiffusionPolicyAdapter:
    """Adapter for Diffusion Policy model training.

    Config should include:
        - obs_dim: Observation dimension
        - action_dim: Action dimension
        - action_horizon: Number of future actions
        - hidden_dim: Hidden layer size (default: 256)
        - num_diffusion_steps: Number of diffusion steps (default: 100)
        - beta_schedule: Diffusion beta schedule (default: "squaredcos_cap_v2")

    Example config:
        ```yaml
        model:
          type: diffusion_policy
          obs_dim: 512
          action_dim: 7
          action_horizon: 8
          hidden_dim: 256
          num_diffusion_steps: 100
        ```
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize adapter.

        Args:
            config: Model configuration dictionary
        """
        self.config = config

        # Extract config values
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.action_horizon = config["action_horizon"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_diffusion_steps = config.get("num_diffusion_steps", 100)
        self.beta_schedule = config.get("beta_schedule", "squaredcos_cap_v2")

        # Create diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            beta_schedule=self.beta_schedule,
            clip_sample=True,
            prediction_type="epsilon",
        )

        logger.info("Initialized DiffusionPolicyAdapter:")
        logger.info(f"  obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        logger.info(f"  action_horizon={self.action_horizon}, hidden_dim={self.hidden_dim}")
        logger.info(f"  diffusion_steps={self.num_diffusion_steps}")

    def create_model(self) -> nn.Module:
        """Create Diffusion Policy model.

        Returns:
            Initialized DiffusionPolicyUNet model
        """
        model = DiffusionPolicyUNet(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            hidden_dim=self.hidden_dim,
        )
        return model

    def create_optimizer(self, model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
        """Create AdamW optimizer.

        Args:
            model: Model to optimize
            lr: Learning rate
            weight_decay: Weight decay

        Returns:
            AdamW optimizer
        """
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Execute one training step with diffusion loss.

        Args:
            model: DiffusionPolicyUNet model
            batch: Batch dict with keys:
                - 'observation': Observations (B, obs_dim)
                - 'action': Ground truth actions (B, action_horizon, action_dim)
            device: Device to run on

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        # Move data to device
        obs = batch["observation"].to(device)  # (B, obs_dim)
        action = batch["action"].to(device)  # (B, action_horizon, action_dim)

        batch_size = action.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()

        # Sample noise
        noise = torch.randn_like(action)

        # Add noise to actions
        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)

        # Predict noise
        noise_pred = model(noisy_actions, obs, timesteps)

        # Compute loss (MSE between predicted and actual noise)
        loss = nn.functional.mse_loss(noise_pred, noise)

        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "noise_mse": loss.item(),
        }

        return loss, metrics

    def eval_step(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        device: torch.device,
    ) -> dict[str, float]:
        """Execute one evaluation step.

        Evaluates action prediction MSE using full diffusion sampling.

        Args:
            model: DiffusionPolicyUNet model
            batch: Batch dict with 'observation' and 'action'
            device: Device to run on

        Returns:
            Metrics dict
        """
        # Move data to device
        obs = batch["observation"].to(device)
        action_gt = batch["action"].to(device)

        batch_size = action_gt.shape[0]

        # Start from random noise
        action_pred = torch.randn_like(action_gt)

        # Denoise iteratively
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict noise
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(action_pred, obs, timesteps)

            # Denoise step
            action_pred = self.noise_scheduler.step(noise_pred, t, action_pred).prev_sample

        # Compute prediction error
        mse = nn.functional.mse_loss(action_pred, action_gt)

        metrics = {
            "eval/action_mse": mse.item(),
            "eval/loss": mse.item(),
        }

        return metrics

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        batch_size: int,
        num_workers: int,
    ) -> tuple[DataLoader, DataLoader | None]:
        """Create train and eval dataloaders.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            batch_size: Batch size
            num_workers: Number of dataloader workers

        Returns:
            Tuple of (train_loader, eval_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        eval_loader = None
        if eval_dataset is not None:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        return train_loader, eval_loader
