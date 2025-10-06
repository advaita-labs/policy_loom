#!/usr/bin/env python3
"""Train pi0.5 model on LeRobot dataset.

This script downloads a LeRobot dataset from HuggingFace Hub and trains pi0.5.

⚠️  IMPORTANT: Run this in a separate virtualenv with pi05 dependencies:
    python -m venv venv-pi05
    source venv-pi05/bin/activate
    uv sync --extra pi05
    uv run python scripts/train_pi05.py --dataset lerobot/koch_test

Example:
    # Train on Koch test dataset
    python scripts/train_pi05.py --dataset lerobot/koch_test --output checkpoints/pi05

    # Fine-tune from pretrained checkpoint
    python scripts/train_pi05.py \\
        --dataset lerobot/aloha_sim_transfer_cube_scripted \\
        --model lerobot/pi05_base \\
        --epochs 50 \\
        --batch-size 16 \\
        --output checkpoints/pi05_aloha

Usage:
    python scripts/train_pi05.py --help
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
    parser = argparse.ArgumentParser(description="Train pi0.5 on LeRobot dataset")

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="LeRobot dataset repository ID (e.g., 'lerobot/koch_test')",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Training split name (default: train)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default=None,
        help="Evaluation split name (default: None, no eval)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="lerobot/pi05_base",
        help="Pretrained model name or path (default: lerobot/pi05_base)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze VLM backbone during training",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-6,
        help="Weight decay (default: 1e-6)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./checkpoints/pi05"),
        help="Output directory for checkpoints (default: ./checkpoints/pi05)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)",
    )

    # Logging arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="policy_loom_pi05",
        help="W&B project name (default: policy_loom_pi05)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be very slow on CPU.")
        logger.warning("Pi0.5 requires significant GPU memory (>22GB for full fine-tuning)")

    # Load dataset
    logger.info(f"Loading training dataset: {args.dataset} (split={args.train_split})")
    train_loader = LeRobotDatasetLoader(
        repo_id=args.dataset,
        split=args.train_split,
    )
    train_dataset = train_loader.to_torch_dataset()
    logger.info(f"Loaded {len(train_dataset)} training samples")

    # Load eval dataset if specified
    eval_dataset = None
    if args.eval_split:
        logger.info(f"Loading eval dataset: {args.dataset} (split={args.eval_split})")
        eval_loader = LeRobotDatasetLoader(
            repo_id=args.dataset,
            split=args.eval_split,
        )
        eval_dataset = eval_loader.to_torch_dataset()
        logger.info(f"Loaded {len(eval_dataset)} eval samples")

    # Create training configuration
    config = TrainingConfig(
        model={
            "type": "pi05",
            "pretrained_model_name_or_path": args.model,
            "freeze_backbone": args.freeze_backbone,
        },
        training=TrainingParams(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            num_workers=4,
        ),
        checkpoints=CheckpointConfig(
            dir=args.output,
            save_every_steps=args.save_every,
            keep_top_k=3,
            keep_last_k=2,
        ),
        evaluation=EvaluationConfig(
            eval_every_steps=args.eval_every if eval_dataset else None,
        ),
        logging=LoggingConfig(
            log_every_steps=10,
            wandb=WandbConfig(
                enabled=args.wandb,
                project=args.wandb_project,
                name=args.wandb_name,
            ),
        ),
        data={
            "dataset": args.dataset,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
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
        eval_dataset=eval_dataset,
    )

    # Start training
    logger.info("=" * 80)
    logger.info(f"Starting pi0.5 training on {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 80)

    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
