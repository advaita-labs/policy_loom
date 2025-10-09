"""CLI for policy_loom using Typer.

Provides commands for:
- Training models
- Preprocessing data
- Running transformations
- Evaluating models
"""

import logging
from pathlib import Path
from typing import Annotated, Any

import torch
import typer

from loom.training import Trainer, TrainingConfig, list_adapters

# Create Typer app
app = typer.Typer(
    name="loom",
    help="Policy Loom - VLA model training toolkit",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo("policy-loom version 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version and exit"),
    ] = None,
) -> None:
    """Policy Loom - VLA model training toolkit."""
    pass


@app.command()
def train(
    config_path: Annotated[Path, typer.Argument(help="Path to training config YAML file")],
    output_dir: Annotated[Path | None, typer.Option(help="Override output directory for checkpoints/logs")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
) -> None:
    """Train a VLA model using the specified configuration.

    Example:
        loom train config.yaml
        loom train config.yaml --output-dir ./runs/experiment1 --verbose
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Validate config file exists
        if not config_path.exists():
            typer.echo(f"Error: Config file not found: {config_path}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading configuration from {config_path}")

        # Load config
        config = TrainingConfig.from_yaml(config_path)

        # Override output dir if specified
        if output_dir:
            config.checkpoints.dir = output_dir / "checkpoints"
            config.logging.log_dir = output_dir / "logs"
            typer.echo(f"Output directory: {output_dir}")

        # Load data based on type
        typer.echo("Loading datasets...")
        data_type = config.data.get("type", "file")

        if data_type == "lerobot":
            # Load LeRobot dataset from HuggingFace
            from loom.io.lerobot import LeRobotDatasetLoader

            dataset_repo = config.data.get("dataset")
            if not dataset_repo:
                typer.echo("Error: 'dataset' field required for lerobot data type", err=True)
                raise typer.Exit(1)

            train_split = config.data.get("train_split", "train")
            eval_split = config.data.get("eval_split")
            local_dir = Path(config.data["local_dir"]) if config.data.get("local_dir") else None

            # Load training data
            typer.echo(f"Loading LeRobot dataset: {dataset_repo} (split={train_split})")
            train_loader = LeRobotDatasetLoader(dataset_repo, split=train_split, local_dir=local_dir)
            train_dataset = train_loader.to_torch_dataset()

            # Load eval data if specified
            eval_dataset = None
            if eval_split:
                typer.echo(f"Loading eval dataset: {dataset_repo} (split={eval_split})")
                eval_loader = LeRobotDatasetLoader(dataset_repo, split=eval_split, local_dir=local_dir)
                eval_dataset = eval_loader.to_torch_dataset()
                typer.echo(f"Loaded {len(train_dataset)} training samples, {len(eval_dataset)} eval samples")
            else:
                typer.echo(f"Loaded {len(train_dataset)} training samples (no eval set)")

        else:
            # Load from local files (legacy path-based loading)
            train_data_path = Path(str(config.data.get("train_path")))
            eval_data_path = Path(str(config.data.get("eval_path"))) if config.data.get("eval_path") else None

            if not train_data_path.exists():
                typer.echo(f"Error: Training data not found: {train_data_path}", err=True)
                raise typer.Exit(1)

            # Load training data
            train_data = torch.load(train_data_path)
            train_dataset = _create_dataset(train_data)

            # Load eval data if specified
            eval_dataset = None
            if eval_data_path and eval_data_path.exists():
                eval_data = torch.load(eval_data_path)
                eval_dataset = _create_dataset(eval_data)
                typer.echo(f"Loaded {len(train_dataset)} training samples, {len(eval_dataset)} eval samples")
            else:
                typer.echo(f"Loaded {len(train_dataset)} training samples (no eval set)")

        # Create trainer
        typer.echo(f"Initializing trainer with model type: {config.model['type']}")
        trainer = Trainer(config, train_dataset, eval_dataset)

        # Start training
        typer.echo("Starting training...")
        trainer.train()

        typer.echo("Training completed successfully!")

    except Exception as e:
        typer.echo(f"Error during training: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def eval(
    config_path: Annotated[Path, typer.Argument(help="Path to training config YAML file")],
    checkpoint: Annotated[Path, typer.Option(help="Path to checkpoint file to evaluate")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
) -> None:
    """Evaluate a trained model on the eval dataset.

    Example:
        loom eval config.yaml --checkpoint checkpoints/checkpoint_step_1000.pt
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    try:
        if not config_path.exists():
            typer.echo(f"Error: Config file not found: {config_path}", err=True)
            raise typer.Exit(1)

        if not checkpoint.exists():
            typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading configuration from {config_path}")
        config = TrainingConfig.from_yaml(config_path)

        # Load eval data
        eval_data_path = Path(str(config.data.get("eval_path")))
        if not eval_data_path.exists():
            typer.echo(f"Error: Eval data not found: {eval_data_path}", err=True)
            raise typer.Exit(1)

        typer.echo("Loading eval dataset...")
        eval_data = torch.load(eval_data_path)
        eval_dataset = _create_dataset(eval_data)

        # Create dummy train dataset (not used for eval)
        train_dataset = eval_dataset

        typer.echo("Initializing trainer...")
        trainer = Trainer(config, train_dataset, eval_dataset)

        # Load checkpoint
        typer.echo(f"Loading checkpoint from {checkpoint}")
        trainer.checkpoint_manager.load(checkpoint, trainer.model)

        # Run evaluation
        typer.echo("Running evaluation...")
        metrics = trainer._evaluate()

        # Print results
        typer.echo("\n" + "=" * 50)
        typer.echo("Evaluation Results:")
        typer.echo("=" * 50)
        for key, value in metrics.items():
            typer.echo(f"{key}: {value:.6f}")
        typer.echo("=" * 50)

    except Exception as e:
        typer.echo(f"Error during evaluation: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def preprocess(
    config_path: Annotated[Path, typer.Argument(help="Path to preprocessing config YAML file")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
) -> None:
    """Run preprocessing on raw data.

    This command will be implemented to preprocess raw robot data
    into the format expected by VLA models.

    Example:
        loom preprocess preprocess_config.yaml
    """
    typer.echo("Preprocessing command coming soon!", err=True)
    typer.echo("This will preprocess raw robot data for model training.", err=True)
    raise typer.Exit(1)


@app.command()
def transform(
    config_path: Annotated[Path, typer.Argument(help="Path to transform config YAML file")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
) -> None:
    """Run transformations on data.

    This command will be implemented to run time/vision transforms
    on preprocessed data.

    Example:
        loom transform transform_config.yaml
    """
    typer.echo("Transform command coming soon!", err=True)
    typer.echo("This will apply transforms to preprocessed data.", err=True)
    raise typer.Exit(1)


@app.command(name="list-adapters")
def list_adapters_command() -> None:
    """List all available model adapters.

    Shows which model types can be used in the config file.

    Example:
        loom list-adapters
    """
    adapters = list_adapters()

    if not adapters:
        typer.echo("No model adapters registered.")
        return

    typer.echo("Available model adapters:")
    typer.echo("=" * 40)
    for adapter_name in sorted(adapters):
        typer.echo(f"  - {adapter_name}")
    typer.echo("=" * 40)
    typer.echo(f"\nTotal: {len(adapters)} adapter(s)")
    typer.echo("\nUse these names in your config file under 'model.type'")


def _create_dataset(data: dict[str, Any]) -> Any:
    """Create a dataset from loaded data.

    Args:
        data: Dict with 'observation' and 'action' keys

    Returns:
        Dataset that yields dicts with observation and action
    """

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, observations: Any, actions: Any) -> None:
            self.observations = torch.tensor(observations, dtype=torch.float32)
            self.actions = torch.tensor(actions, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.observations)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "observation": self.observations[idx],
                "action": self.actions[idx],
            }

    return DictDataset(data["observation"], data["action"])


@app.command()
def train_pi05(
    dataset: Annotated[str, typer.Argument(help="HuggingFace LeRobot dataset")],
    config_name: Annotated[str, typer.Option(help="OpenPI config")] = "pi05_libero",
    custom_config: Annotated[Path | None, typer.Option(help="Path to custom Python config module")] = None,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 256,
    steps: Annotated[int, typer.Option(help="Training steps")] = 30000,
    output_dir: Annotated[Path, typer.Option(help="Checkpoint output directory")] = Path("./checkpoints"),
    lr: Annotated[float | None, typer.Option(help="Learning rate (default: use config)")] = None,
) -> None:
    """Train Pi0.5 using OpenPI (simplified integration).

    Example: loom train-pi05 gauravpradeep/t02_piper_pick_and_place_bimanual --steps 30000

    For custom datasets, provide a custom config module:
        loom train-pi05 DATASET --custom-config configs/pi05_gauravpradeep_simple.py
    """
    try:
        from openpi.training import config as openpi_config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        import dataclasses
        import jax
        import numpy as np
    except ImportError as e:
        typer.echo(f"Error: OpenPI not installed: {e}", err=True)
        typer.echo("Install with: uv sync --extra pi05", err=True)
        raise typer.Exit(1)

    # Patch torch.stack to handle HuggingFace Column objects
    # This fixes an issue in lerobot's LeRobotDataset.__init__ which calls
    # torch.stack on Column objects
    _original_torch_stack = torch.stack

    def _patched_torch_stack(tensors, *args, **kwargs):
        """Patched torch.stack that handles HuggingFace Column objects."""
        # Check if tensors is a HuggingFace Column
        if hasattr(tensors, '__class__') and 'Column' in tensors.__class__.__name__:
            # Convert Column to list first
            tensors = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in tensors]
        return _original_torch_stack(tensors, *args, **kwargs)

    torch.stack = _patched_torch_stack

    # Patch OpenPI's collate function to handle HuggingFace Column objects
    import openpi.training.data_loader as openpi_data_loader

    def _patched_collate_fn(items):
        """Collate the batch elements into batched numpy arrays.

        This patched version handles HuggingFace Column objects that don't
        properly convert with np.asarray(). Instead, we use np.array() which
        handles more cases correctly.
        """
        def to_numpy(x):
            """Convert any array-like object to numpy array."""
            if isinstance(x, np.ndarray):
                return x
            # Use np.array() instead of np.asarray() to handle HuggingFace Columns
            return np.array(x)

        return jax.tree.map(lambda *xs: np.stack([to_numpy(x) for x in xs], axis=0), *items)

    # Replace the collate function before it's used
    openpi_data_loader._collate_fn = _patched_collate_fn

    # Now import create_data_loader
    from openpi.training.data_loader import create_data_loader

    # Load and configure
    if custom_config:
        typer.echo(f"Loading custom config from '{custom_config}'...")
        try:
            # Load custom config module
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location("custom_config", custom_config)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {custom_config}")

            custom_module = importlib.util.module_from_spec(spec)
            sys.modules["custom_config"] = custom_module
            spec.loader.exec_module(custom_module)

            # Get the data config class (should end with DataConfig)
            data_config_class = None
            for name in dir(custom_module):
                obj = getattr(custom_module, name)
                if isinstance(obj, type) and name.endswith("DataConfig") and name != "DataConfig":
                    data_config_class = obj
                    break

            if data_config_class is None:
                raise ValueError(f"No DataConfig class found in {custom_config}")

            typer.echo(f"  Found config class: {data_config_class.__name__}")

            # Create data config instance
            data_config_instance = data_config_class(repo_id=dataset, assets=None, base_config=None)

            # Use pi05_libero as base and replace data config
            config = openpi_config.get_config('pi05_libero')
            config = dataclasses.replace(config, data=data_config_instance)

        except Exception as e:
            typer.echo(f"Error loading custom config: {e}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo(f"Loading OpenPI config '{config_name}'...")
        try:
            config = openpi_config.get_config(config_name)
        except KeyError:
            typer.echo(f"Error: Unknown config '{config_name}'", err=True)
            typer.echo("Available configs: pi05_libero, pi05_droid, pi0_libero, etc.", err=True)
            raise typer.Exit(1)

    # Update config with training parameters
    # Only update repo_id if not using custom config (custom config already has it set)
    if custom_config:
        config = dataclasses.replace(
            config,
            batch_size=batch_size,
            num_train_steps=steps,
            num_workers=0,  # Disable multiprocessing to avoid pickling issues with patched functions
        )
    else:
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, repo_id=dataset),
            batch_size=batch_size,
            num_train_steps=steps,
            num_workers=0,  # Disable multiprocessing to avoid pickling issues with patched functions
        )

    typer.echo(f"\nTraining Pi0.5:")
    typer.echo(f"  Dataset: {dataset}")
    typer.echo(f"  Steps: {steps}")
    typer.echo(f"  Batch size: {batch_size}")
    typer.echo(f"  Output: {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"  Device: {device}")

    # Create model
    typer.echo("\nCreating model...")
    model = PI0Pytorch(config.model).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    typer.echo(f"  Model size: {num_params:.1f}B parameters")

    # Create optimizer with config's learning rate or override
    learning_rate = lr if lr is not None else config.lr_schedule.peak_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    typer.echo(f"  Learning rate: {learning_rate}")

    # Data loader
    typer.echo("\nCreating data loader...")
    try:
        loader = create_data_loader(config, framework="pytorch", shuffle=True, skip_norm_stats=True)
        typer.echo("  Data loader ready")
    except Exception as e:
        typer.echo(f"Error: Data loading failed: {e}", err=True)
        typer.echo("\nPossible issues:", err=True)
        typer.echo("  - Dataset not found on HuggingFace", err=True)
        typer.echo("  - Dataset format incompatible", err=True)
        raise typer.Exit(1)

    # Create checkpoint directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    typer.echo(f"\nTraining started...\n")
    for step, (obs, actions) in enumerate(loader):
        if step >= steps:
            break

        obs, actions = obs.to(device), actions.to(device)
        loss = model.forward(obs, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if step % 100 == 0:
            typer.echo(f"Step {step}/{steps}: Loss = {loss.item():.4f}")

        # Save checkpoint every 1000 steps
        if step > 0 and step % 1000 == 0:
            checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'config': config,
            }, checkpoint_path)
            typer.echo(f"  Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    final_path = output_dir / "final_checkpoint.pt"
    torch.save({
        'step': steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)

    typer.echo(f"\nTraining complete!")
    typer.echo(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    app()
