"""Training configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration.

    Attributes:
        type: Scheduler type (cosine, step, constant, plateau)
        warmup_steps: Number of warmup steps
        min_lr: Minimum learning rate for cosine schedule
        step_size: Step size for step scheduler
        gamma: Multiplicative factor for step/plateau scheduler
        patience: Patience for plateau scheduler (epochs to wait before reducing LR)
    """

    type: Literal["cosine", "step", "constant", "plateau"] = "cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    step_size: int | None = None
    gamma: float = 0.1
    patience: int = 10


@dataclass
class TrainingParams:
    """Training hyperparameters.

    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        lr_scheduler: Learning rate scheduler config
        gradient_clip_norm: Max gradient norm for clipping
        mixed_precision: Use automatic mixed precision
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch
        optimizer: Optimizer type (adamw or adam)
        betas: Beta parameters for Adam/AdamW
        seed: Random seed for reproducibility
    """

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    optimizer: Literal["adamw", "adam"] = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    seed: int = 42


@dataclass
class CheckpointConfig:
    """Checkpoint management configuration.

    Attributes:
        dir: Directory to save checkpoints
        save_every_steps: Save checkpoint every N steps (None to disable)
        save_every_epochs: Save checkpoint every N epochs (None to disable)
        keep_top_k: Number of best checkpoints to keep
        keep_last_k: Number of recent checkpoints to keep
        metric_for_best: Metric to determine best checkpoint
        mode: Whether to minimize or maximize metric
        resume_from: Path to checkpoint to resume from
    """

    dir: Path = Path("./checkpoints")
    save_every_steps: int | None = 1000
    save_every_epochs: int | None = None
    keep_top_k: int = 3
    keep_last_k: int = 2
    metric_for_best: str = "eval/loss"
    mode: Literal["min", "max"] = "min"
    resume_from: Path | None = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration.

    Attributes:
        eval_every_steps: Evaluate every N steps (None to disable)
        eval_every_epochs: Evaluate every N epochs (None to disable)
        eval_batches: Max batches per eval (None = full eval set)
    """

    eval_every_steps: int | None = 500
    eval_every_epochs: int | None = None
    eval_batches: int | None = None


@dataclass
class WandbConfig:
    """Weights & Biases configuration.

    Attributes:
        enabled: Enable wandb logging
        project: WandB project name
        entity: WandB entity/username
        name: Run name (None = auto-generated)
        tags: List of tags for the run
        notes: Run notes/description
        group: Group name for organizing runs
    """

    enabled: bool = False
    project: str = "policy_loom"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    group: str | None = None


@dataclass
class LoggingConfig:
    """Logging configuration.

    Attributes:
        log_every_steps: Log to console every N steps
        wandb: WandB configuration
        log_dir: Directory for local logs
        save_logs: Save logs to file
    """

    log_every_steps: int = 10
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_dir: Path = Path("./logs")
    save_logs: bool = True


@dataclass
class TrainingConfig:
    """Complete training configuration.

    Attributes:
        model: Model-specific configuration dict
        training: Training hyperparameters
        checkpoints: Checkpoint management config
        evaluation: Evaluation config
        logging: Logging config
        data: Data paths and config dict
    """

    model: dict[str, Any]
    training: TrainingParams
    checkpoints: CheckpointConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    data: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        # Parse nested configs
        training_dict = config_dict.get("training", {}).copy()
        lr_scheduler_dict = training_dict.pop("lr_scheduler", {})

        # Ensure numeric types are correctly converted (dataclasses don't enforce types)
        if "warmup_steps" in lr_scheduler_dict:
            lr_scheduler_dict["warmup_steps"] = int(lr_scheduler_dict["warmup_steps"])
        if "min_lr" in lr_scheduler_dict:
            lr_scheduler_dict["min_lr"] = float(lr_scheduler_dict["min_lr"])
        if "step_size" in lr_scheduler_dict and lr_scheduler_dict["step_size"] is not None:
            lr_scheduler_dict["step_size"] = int(lr_scheduler_dict["step_size"])
        if "gamma" in lr_scheduler_dict:
            lr_scheduler_dict["gamma"] = float(lr_scheduler_dict["gamma"])
        if "patience" in lr_scheduler_dict:
            lr_scheduler_dict["patience"] = int(lr_scheduler_dict["patience"])

        # Ensure training params are correctly typed
        if "epochs" in training_dict:
            training_dict["epochs"] = int(training_dict["epochs"])
        if "batch_size" in training_dict:
            training_dict["batch_size"] = int(training_dict["batch_size"])
        if "learning_rate" in training_dict:
            training_dict["learning_rate"] = float(training_dict["learning_rate"])
        if "weight_decay" in training_dict:
            training_dict["weight_decay"] = float(training_dict["weight_decay"])
        if "gradient_clip_norm" in training_dict:
            training_dict["gradient_clip_norm"] = float(training_dict["gradient_clip_norm"])
        if "num_workers" in training_dict:
            training_dict["num_workers"] = int(training_dict["num_workers"])

        training_params = TrainingParams(
            **training_dict,
            lr_scheduler=LRSchedulerConfig(**lr_scheduler_dict),
        )

        # Ensure checkpoint config types are correct
        checkpoint_dict = config_dict.get("checkpoints", {}).copy()
        if "save_every_steps" in checkpoint_dict and checkpoint_dict["save_every_steps"] is not None:
            checkpoint_dict["save_every_steps"] = int(checkpoint_dict["save_every_steps"])
        if "save_every_epochs" in checkpoint_dict and checkpoint_dict["save_every_epochs"] is not None:
            checkpoint_dict["save_every_epochs"] = int(checkpoint_dict["save_every_epochs"])
        if "keep_top_k" in checkpoint_dict:
            checkpoint_dict["keep_top_k"] = int(checkpoint_dict["keep_top_k"])
        if "keep_last_k" in checkpoint_dict:
            checkpoint_dict["keep_last_k"] = int(checkpoint_dict["keep_last_k"])

        checkpoint_config = CheckpointConfig(**checkpoint_dict)
        # Convert string path to Path object
        if "dir" in config_dict.get("checkpoints", {}):
            checkpoint_config.dir = Path(checkpoint_config.dir)
        if checkpoint_config.resume_from is not None:
            checkpoint_config.resume_from = Path(checkpoint_config.resume_from)

        # Ensure evaluation config types are correct
        eval_dict = config_dict.get("evaluation", {}).copy()
        if "eval_every_steps" in eval_dict and eval_dict["eval_every_steps"] is not None:
            eval_dict["eval_every_steps"] = int(eval_dict["eval_every_steps"])
        if "eval_every_epochs" in eval_dict and eval_dict["eval_every_epochs"] is not None:
            eval_dict["eval_every_epochs"] = int(eval_dict["eval_every_epochs"])
        if "eval_batches" in eval_dict and eval_dict["eval_batches"] is not None:
            eval_dict["eval_batches"] = int(eval_dict["eval_batches"])

        eval_config = EvaluationConfig(**eval_dict)

        # Ensure logging config types are correct
        logging_dict = {k: v for k, v in config_dict.get("logging", {}).items() if k != "wandb"}
        if "log_every_steps" in logging_dict:
            logging_dict["log_every_steps"] = int(logging_dict["log_every_steps"])

        logging_config = LoggingConfig(
            **logging_dict,
            wandb=WandbConfig(**config_dict.get("logging", {}).get("wandb", {})),
        )
        if "log_dir" in config_dict.get("logging", {}):
            logging_config.log_dir = Path(logging_config.log_dir)

        return cls(
            model=config_dict.get("model", {}),
            training=training_params,
            checkpoints=checkpoint_config,
            evaluation=eval_config,
            logging=logging_config,
            data=config_dict.get("data", {}),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML config
        """
        # Convert to dict
        config_dict = {
            "model": self.model,
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "lr_scheduler": {
                    "type": self.training.lr_scheduler.type,
                    "warmup_steps": self.training.lr_scheduler.warmup_steps,
                    "min_lr": self.training.lr_scheduler.min_lr,
                    "step_size": self.training.lr_scheduler.step_size,
                    "gamma": self.training.lr_scheduler.gamma,
                },
                "gradient_clip_norm": self.training.gradient_clip_norm,
                "mixed_precision": self.training.mixed_precision,
                "num_workers": self.training.num_workers,
                "pin_memory": self.training.pin_memory,
                "prefetch_factor": self.training.prefetch_factor,
                "optimizer": self.training.optimizer,
                "betas": list(self.training.betas),
                "seed": self.training.seed,
            },
            "checkpoints": {
                "dir": str(self.checkpoints.dir),
                "save_every_steps": self.checkpoints.save_every_steps,
                "save_every_epochs": self.checkpoints.save_every_epochs,
                "keep_top_k": self.checkpoints.keep_top_k,
                "keep_last_k": self.checkpoints.keep_last_k,
                "metric_for_best": self.checkpoints.metric_for_best,
                "mode": self.checkpoints.mode,
                "resume_from": str(self.checkpoints.resume_from) if self.checkpoints.resume_from else None,
            },
            "evaluation": {
                "eval_every_steps": self.evaluation.eval_every_steps,
                "eval_every_epochs": self.evaluation.eval_every_epochs,
                "eval_batches": self.evaluation.eval_batches,
            },
            "logging": {
                "log_every_steps": self.logging.log_every_steps,
                "wandb": {
                    "enabled": self.logging.wandb.enabled,
                    "project": self.logging.wandb.project,
                    "entity": self.logging.wandb.entity,
                    "name": self.logging.wandb.name,
                    "tags": self.logging.wandb.tags,
                    "notes": self.logging.wandb.notes,
                    "group": self.logging.wandb.group,
                },
                "log_dir": str(self.logging.log_dir),
                "save_logs": self.logging.save_logs,
            },
            "data": self.data,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
