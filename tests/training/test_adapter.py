"""Tests for model adapter protocol and registry."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from loom.training.adapter import get_adapter, list_adapters, register_adapter


# Test implementation of ModelAdapter (not a pytest test class)
@register_adapter("test_adapter")
class MockAdapter:
    """Test adapter implementation."""

    def __init__(self, config):
        self.config = config

    def create_model(self):
        return nn.Linear(10, 5)

    def create_optimizer(self, model, lr, weight_decay):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def training_step(self, model, batch, device):
        loss = torch.tensor(0.5)
        metrics = {"loss": 0.5}
        return loss, metrics

    def eval_step(self, model, batch, device):
        return {"eval/loss": 0.3}

    def create_dataloaders(self, train_dataset, eval_dataset, batch_size, num_workers):
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size) if eval_dataset else None
        return train_loader, eval_loader


class TestAdapterRegistry:
    """Test adapter registration and retrieval."""

    def test_register_adapter(self):
        """Test registering an adapter."""
        assert "test_adapter" in list_adapters()

    def test_get_adapter(self):
        """Test getting registered adapter."""
        config = {"param1": "value1"}
        adapter = get_adapter("test_adapter", config)

        assert isinstance(adapter, MockAdapter)
        assert adapter.config == config

    def test_get_nonexistent_adapter(self):
        """Test getting non-existent adapter raises error."""
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("nonexistent_adapter", {})

    def test_list_adapters(self):
        """Test listing all registered adapters."""
        adapters = list_adapters()
        assert isinstance(adapters, list)
        assert "test_adapter" in adapters

    def test_register_duplicate_adapter(self):
        """Test registering duplicate adapter name raises error."""
        with pytest.raises(ValueError, match="already registered"):

            @register_adapter("test_adapter")
            class DuplicateAdapter:
                pass

    def test_adapter_create_model(self):
        """Test adapter can create model."""
        adapter = get_adapter("test_adapter", {})
        model = adapter.create_model()

        assert isinstance(model, nn.Module)

    def test_adapter_create_optimizer(self):
        """Test adapter can create optimizer."""
        adapter = get_adapter("test_adapter", {})
        model = adapter.create_model()
        optimizer = adapter.create_optimizer(model, lr=1e-3, weight_decay=1e-6)

        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_adapter_training_step(self):
        """Test adapter training step."""
        adapter = get_adapter("test_adapter", {})
        model = adapter.create_model()
        batch = {"data": torch.randn(4, 10)}
        device = torch.device("cpu")

        loss, metrics = adapter.training_step(model, batch, device)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_adapter_eval_step(self):
        """Test adapter eval step."""
        adapter = get_adapter("test_adapter", {})
        model = adapter.create_model()
        batch = {"data": torch.randn(4, 10)}
        device = torch.device("cpu")

        metrics = adapter.eval_step(model, batch, device)

        assert isinstance(metrics, dict)
        assert "eval/loss" in metrics

    def test_adapter_create_dataloaders(self):
        """Test adapter creates dataloaders."""
        adapter = get_adapter("test_adapter", {})

        class DummyDataset(Dataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {"data": torch.randn(10)}

        train_dataset = DummyDataset()
        eval_dataset = DummyDataset()

        train_loader, eval_loader = adapter.create_dataloaders(
            train_dataset, eval_dataset, batch_size=16, num_workers=0
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(eval_loader, DataLoader)

    def test_adapter_create_dataloaders_without_eval(self):
        """Test adapter creates dataloaders without eval dataset."""
        adapter = get_adapter("test_adapter", {})

        class DummyDataset(Dataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {"data": torch.randn(10)}

        train_dataset = DummyDataset()

        train_loader, eval_loader = adapter.create_dataloaders(train_dataset, None, batch_size=16, num_workers=0)

        assert isinstance(train_loader, DataLoader)
        assert eval_loader is None
