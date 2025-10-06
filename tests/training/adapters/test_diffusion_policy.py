"""Tests for DiffusionPolicyAdapter."""

import pytest
import torch
from torch.utils.data import TensorDataset

from loom.training.adapters.diffusion_policy import DiffusionPolicyAdapter, DiffusionPolicyUNet


class TestDiffusionPolicyUNet:
    """Test DiffusionPolicyUNet model."""

    def test_initialization(self):
        """Test model initialization."""
        model = DiffusionPolicyUNet(
            obs_dim=128,
            action_dim=7,
            action_horizon=8,
            hidden_dim=256,
        )

        assert model.obs_dim == 128
        assert model.action_dim == 7
        assert model.action_horizon == 8

    def test_forward_pass(self):
        """Test forward pass shape."""
        model = DiffusionPolicyUNet(
            obs_dim=128,
            action_dim=7,
            action_horizon=8,
            hidden_dim=256,
        )

        batch_size = 4
        noisy_actions = torch.randn(batch_size, 8, 7)
        obs = torch.randn(batch_size, 128)
        timestep = torch.randint(0, 100, (batch_size,))

        noise_pred = model(noisy_actions, obs, timestep)

        assert noise_pred.shape == (batch_size, 8, 7)

    def test_forward_pass_gradients(self):
        """Test that gradients flow through the model."""
        model = DiffusionPolicyUNet(
            obs_dim=128,
            action_dim=7,
            action_horizon=8,
        )

        noisy_actions = torch.randn(2, 8, 7, requires_grad=True)
        obs = torch.randn(2, 128, requires_grad=True)
        timestep = torch.tensor([10, 50])

        noise_pred = model(noisy_actions, obs, timestep)
        loss = noise_pred.sum()
        loss.backward()

        # Check gradients exist
        assert noisy_actions.grad is not None
        assert obs.grad is not None


class TestDiffusionPolicyAdapter:
    """Test DiffusionPolicyAdapter."""

    @pytest.fixture
    def adapter_config(self):
        """Create adapter configuration."""
        return {
            "type": "diffusion_policy",
            "obs_dim": 128,
            "action_dim": 7,
            "action_horizon": 8,
            "hidden_dim": 256,
            "num_diffusion_steps": 100,
        }

    @pytest.fixture
    def adapter(self, adapter_config):
        """Create adapter instance."""
        return DiffusionPolicyAdapter(adapter_config)

    def test_initialization(self, adapter, adapter_config):
        """Test adapter initialization."""
        assert adapter.obs_dim == 128
        assert adapter.action_dim == 7
        assert adapter.action_horizon == 8
        assert adapter.hidden_dim == 256
        assert adapter.num_diffusion_steps == 100
        assert adapter.noise_scheduler is not None

    def test_create_model(self, adapter):
        """Test model creation."""
        model = adapter.create_model()

        assert isinstance(model, DiffusionPolicyUNet)
        assert model.obs_dim == 128
        assert model.action_dim == 7
        assert model.action_horizon == 8

    def test_create_optimizer(self, adapter):
        """Test optimizer creation."""
        model = adapter.create_model()
        optimizer = adapter.create_optimizer(model, lr=1e-4, weight_decay=1e-6)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 1e-6

    def test_training_step(self, adapter):
        """Test training step."""
        model = adapter.create_model()
        model.eval()  # Disable dropout if any

        batch = {
            "observation": torch.randn(4, 128),
            "action": torch.randn(4, 8, 7),
        }

        loss, metrics = adapter.training_step(model, batch, torch.device("cpu"))

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert "loss" in metrics
        assert "noise_mse" in metrics
        assert metrics["loss"] > 0

    def test_training_step_backprop(self, adapter):
        """Test that training step loss can backpropagate."""
        model = adapter.create_model()
        optimizer = adapter.create_optimizer(model, lr=1e-4, weight_decay=1e-6)

        batch = {
            "observation": torch.randn(4, 128),
            "action": torch.randn(4, 8, 7),
        }

        # Forward pass
        loss, _ = adapter.training_step(model, batch, torch.device("cpu"))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients

    def test_eval_step(self, adapter):
        """Test evaluation step."""
        model = adapter.create_model()
        model.eval()

        batch = {
            "observation": torch.randn(2, 128),
            "action": torch.randn(2, 8, 7),
        }

        with torch.no_grad():
            metrics = adapter.eval_step(model, batch, torch.device("cpu"))

        assert "eval/action_mse" in metrics
        assert "eval/loss" in metrics
        assert metrics["eval/action_mse"] >= 0

    def test_create_dataloaders(self, adapter):
        """Test dataloader creation."""
        # Create dummy datasets
        obs = torch.randn(20, 128)
        actions = torch.randn(20, 8, 7)
        train_dataset = TensorDataset(obs, actions)
        eval_dataset = TensorDataset(obs[:10], actions[:10])

        train_loader, eval_loader = adapter.create_dataloaders(
            train_dataset,
            eval_dataset,
            batch_size=4,
            num_workers=0,
        )

        assert train_loader is not None
        assert eval_loader is not None
        assert len(train_loader) == 5  # 20 / 4
        assert len(eval_loader) == 3  # 10 / 4 (rounded up)

    def test_create_dataloaders_no_eval(self, adapter):
        """Test dataloader creation without eval dataset."""
        obs = torch.randn(20, 128)
        actions = torch.randn(20, 8, 7)
        train_dataset = TensorDataset(obs, actions)

        train_loader, eval_loader = adapter.create_dataloaders(
            train_dataset,
            None,
            batch_size=4,
            num_workers=0,
        )

        assert train_loader is not None
        assert eval_loader is None

    def test_default_config_values(self):
        """Test default configuration values."""
        config = {
            "type": "diffusion_policy",
            "obs_dim": 128,
            "action_dim": 7,
            "action_horizon": 8,
        }

        adapter = DiffusionPolicyAdapter(config)

        # Check defaults
        assert adapter.hidden_dim == 256
        assert adapter.num_diffusion_steps == 100
        assert adapter.beta_schedule == "squaredcos_cap_v2"

    def test_custom_diffusion_steps(self):
        """Test custom number of diffusion steps."""
        config = {
            "type": "diffusion_policy",
            "obs_dim": 128,
            "action_dim": 7,
            "action_horizon": 8,
            "num_diffusion_steps": 50,
        }

        adapter = DiffusionPolicyAdapter(config)
        assert adapter.num_diffusion_steps == 50
        assert adapter.noise_scheduler.config.num_train_timesteps == 50

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, adapter, batch_size):
        """Test training with different batch sizes."""
        model = adapter.create_model()

        batch = {
            "observation": torch.randn(batch_size, 128),
            "action": torch.randn(batch_size, 8, 7),
        }

        loss, metrics = adapter.training_step(model, batch, torch.device("cpu"))

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar loss

    def test_eval_deterministic(self, adapter):
        """Test that eval with same input gives same output."""
        model = adapter.create_model()
        model.eval()

        batch = {
            "observation": torch.randn(1, 128),
            "action": torch.randn(1, 8, 7),
        }

        # Set seed for reproducibility
        torch.manual_seed(42)
        with torch.no_grad():
            metrics1 = adapter.eval_step(model, batch, torch.device("cpu"))

        torch.manual_seed(42)
        with torch.no_grad():
            metrics2 = adapter.eval_step(model, batch, torch.device("cpu"))

        # Should be deterministic with same seed
        assert abs(metrics1["eval/action_mse"] - metrics2["eval/action_mse"]) < 1e-5
