#!/usr/bin/env python3
"""Train DiffusionPolicy on eval_t10_pick_and_place dataset.

This script tests DiffusionPolicy training on a LeRobot format dataset
on Apple Silicon (MPS) or CUDA GPUs.

Usage:
    source .venv-diffusion/bin/activate
    python scripts/train_diffusion_eval_t10.py

Example with options:
    python scripts/train_diffusion_eval_t10.py \\
        --epochs 10 \\
        --batch-size 8 \\
        --output checkpoints/diffusion_eval_t10
"""

import argparse
import logging
from pathlib import Path

import torch

from loom.io.lerobot import LeRobotDatasetLoader
from loom.training.config import (
    CheckpointConfig,
    EvaluationConfig,
    LoggingConfig,
    TrainingConfig,
    TrainingParams,
    WandbConfig,
)
from loom.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train DiffusionPolicy on eval_t10 dataset")

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5 for quick test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4 for MPS)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./checkpoints/diffusion_eval_t10"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    args = parser.parse_args()

    # Check device availability
    if torch.cuda.is_available():
        logger.info("CUDA available - will train on GPU")
    elif torch.backends.mps.is_available():
        logger.info("MPS available - will train on Apple Silicon")
    else:
        logger.warning("No GPU available - training will be slow on CPU")

    # Load dataset
    logger.info("Loading eval_t10_pick_and_place dataset from HuggingFace Hub")
    logger.info("This will download ~2GB of data on first run")

    train_loader = LeRobotDatasetLoader(
        repo_id="LBST/eval_t10_pick_and_place",
        split="train",
    )
    train_dataset = train_loader.to_torch_dataset()
    logger.info(f"Loaded {len(train_dataset)} training samples")

    # Check dataset structure
    sample = train_dataset[0]
    logger.info(f"Sample structure:")
    logger.info(f"  Observation shape: {sample['observation'].shape if sample['observation'] is not None else None}")
    logger.info(f"  Action shape: {sample['action'].shape if sample['action'] is not None else None}")

    # Create training configuration
    # Note: DiffusionPolicy uses observation history (obs_horizon) and action chunking (action_horizon)
    config = TrainingConfig(
        model={
            "type": "diffusion_policy",
            "obs_horizon": 2,  # 2 past observations
            "action_horizon": 8,  # 8 future actions
            "camera_names": ["front", "up"],  # eval_t10 has front and up cameras
            "obs_dim": 6,  # Match DiffusionPolicyAdapter config key
            "action_dim": 6,  # eval_t10 has 6D actions
        },
        training=TrainingParams(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_workers=2,  # Lower for MPS
            weight_decay=1e-6,
        ),
        checkpoints=CheckpointConfig(
            dir=args.output,
            save_every_steps=500,
            keep_top_k=2,
            keep_last_k=1,
        ),
        evaluation=EvaluationConfig(
            eval_every_steps=None,  # No eval split for this dataset
        ),
        logging=LoggingConfig(
            log_every_steps=10,
            wandb=WandbConfig(
                enabled=args.wandb,
                project="policy_loom_diffusion",
                name="eval_t10_test",
            ),
        ),
        data={
            "dataset": "LBST/eval_t10_pick_and_place",
            "train_split": "train",
        },
    )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = args.output / "training_config.yaml"
    config.to_yaml(config_path)
    logger.info(f"Saved training config to {config_path}")

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    # Start training
    logger.info("=" * 80)
    logger.info("Starting DiffusionPolicy training on eval_t10_pick_and_place")
    logger.info(f"Device: {trainer.device}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    logger.info(f"Dataset: {len(train_dataset)} samples")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)

    try:
        trainer.train()
        logger.info("✓ Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
