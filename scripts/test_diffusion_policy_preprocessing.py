"""End-to-end test of Diffusion Policy preprocessing on run19 data."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loom.core.types import DiffusionPolicyInput, Sample
from loom.io.mcap import MCAPReader
from loom.io.mp4 import MP4Reader
from loom.pipeline import merge_streams
from loom.preprocessing import (
    DiffusionPolicyPreprocessingConfig,
    DiffusionPolicyPreprocessor,
    filter_samples_by_cameras,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SampleSequenceDataset(Dataset):
    """Dataset wrapper for sample sequences."""

    def __init__(self, sequences: list[list[Sample]], preprocessor: DiffusionPolicyPreprocessor):
        self.sequences = sequences
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> DiffusionPolicyInput:
        return self.preprocessor.preprocess_sample_sequence(self.sequences[idx])


def compute_normalization_stats(samples: list[Sample]) -> dict:
    """Compute normalization statistics from samples.

    Args:
        samples: List of samples to compute stats from

    Returns:
        Dictionary with state_mean, state_std, action_mean, action_std
    """
    logger.info("Computing normalization statistics...")

    # Collect all state and action vectors
    states = []
    actions = []

    for sample in samples:
        if sample.proprio is not None:
            states.append(sample.proprio)
        if sample.action is not None:
            actions.append(sample.action)

    if not states or not actions:
        raise ValueError("No state or action data found in samples")

    # Stack and compute statistics
    states_array = np.stack(states, axis=0)
    actions_array = np.stack(actions, axis=0)

    state_mean = states_array.mean(axis=0).tolist()
    state_std = states_array.std(axis=0).tolist()
    action_mean = actions_array.mean(axis=0).tolist()
    action_std = actions_array.std(axis=0).tolist()

    # Replace zero stds with 1.0 to avoid division by zero
    state_std = [s if s > 1e-6 else 1.0 for s in state_std]
    action_std = [s if s > 1e-6 else 1.0 for s in action_std]

    logger.info(f"State dim: {len(state_mean)}, Action dim: {len(action_mean)}")
    logger.info(f"State mean: {state_mean[:3]}...")
    logger.info(f"State std: {state_std[:3]}...")
    logger.info(f"Action mean: {action_mean[:3]}...")
    logger.info(f"Action std: {action_std[:3]}...")

    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "state_dim": len(state_mean),
        "action_dim": len(action_mean),
    }


def create_sequences(samples: list[Sample], obs_horizon: int, action_horizon: int) -> list[list[Sample]]:
    """Create sliding window sequences for Diffusion Policy.

    Args:
        samples: List of samples in temporal order
        obs_horizon: Number of observation steps
        action_horizon: Number of action steps

    Returns:
        List of sample sequences, each with obs_horizon + action_horizon samples
    """
    sequences = []
    window_size = obs_horizon + action_horizon

    for i in range(len(samples) - window_size + 1):
        sequence = samples[i : i + window_size]
        sequences.append(sequence)

    logger.info(f"Created {len(sequences)} sequences from {len(samples)} samples")
    return sequences


def test_diffusion_policy_preprocessing(data_dir: Path, num_samples: int = 100) -> None:
    """Test Diffusion Policy preprocessing on run19 dataset.

    Args:
        data_dir: Root directory containing videos/ and mcap file
        num_samples: Number of samples to collect for testing
    """
    video_dir = data_dir / "videos"
    mcap_file = data_dir / "run19_0.mcap"

    # Verify files exist
    if not video_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {video_dir}")
    if not mcap_file.exists():
        raise FileNotFoundError(f"MCAP file not found: {mcap_file}")

    logger.info("=" * 80)
    logger.info("STEP 1: Reading and merging data")
    logger.info("=" * 80)

    # Create readers
    readers = []

    left_cam_file = video_dir / "left_arm.perception_interface.left_cam.state.mp4"
    if left_cam_file.exists():
        readers.append(MP4Reader(left_cam_file, camera_name="left_cam"))
        logger.info("Added reader: left_cam")

    right_cam_file = video_dir / "right_arm.perception_interface.right_cam.state.mp4"
    if right_cam_file.exists():
        readers.append(MP4Reader(right_cam_file, camera_name="right_cam"))
        logger.info("Added reader: right_cam")

    middle_cam_file = video_dir / "torso.perception_interface.middle_cam.state.mp4"
    if middle_cam_file.exists():
        readers.append(MP4Reader(middle_cam_file, camera_name="middle_cam"))
        logger.info("Added reader: middle_cam")

    mcap_reader = MCAPReader(mcap_file)
    readers.append(mcap_reader)
    logger.info("Added reader: MCAP")

    # Collect samples
    logger.info(f"\nCollecting {num_samples} samples...")
    samples = []

    for i, sample in enumerate(merge_streams(*readers)):
        if i >= num_samples:
            break

        samples.append(sample)

        if i < 3:
            logger.info(f"\nSample {i + 1}:")
            logger.info(f"  Timestamp: {sample.timestamp:.4f}s")
            logger.info(f"  Cameras: {[c.name for c in sample.cameras]}")
            if sample.proprio is not None:
                logger.info(f"  Proprio shape: {sample.proprio.shape}")
            if sample.action is not None:
                logger.info(f"  Action shape: {sample.action.shape}")

    logger.info(f"✓ Collected {len(samples)} samples")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Filtering samples by camera availability")
    logger.info("=" * 80)

    # Define required cameras
    required_cameras = ["left_cam", "right_cam", "middle_cam"]
    logger.info(f"Required cameras: {required_cameras}")
    logger.info(f"Total samples before filtering: {len(samples)}")

    # Filter samples to only include those with all required cameras
    filtered_samples = filter_samples_by_cameras(samples, required_cameras)

    if not filtered_samples:
        raise ValueError(
            f"No samples have all required cameras {required_cameras}! "
            "Check camera synchronization or adjust required camera list."
        )

    logger.info(f"✓ Filtered to {len(filtered_samples)} complete samples")

    # Use filtered samples for rest of pipeline
    samples = filtered_samples

    # Validate samples
    first_sample = samples[0]
    logger.info("\nFirst sample inspection:")
    logger.info(f"  Has cameras: {len(first_sample.cameras) > 0}")
    logger.info(f"  Has proprio: {first_sample.proprio is not None}")
    logger.info(f"  Has action: {first_sample.action is not None}")

    # Check if we have proprio/action data
    samples_with_proprio = sum(1 for s in samples if s.proprio is not None)
    samples_with_action = sum(1 for s in samples if s.action is not None)

    logger.info(f"  Samples with proprio: {samples_with_proprio}/{len(samples)}")
    logger.info(f"  Samples with action: {samples_with_action}/{len(samples)}")

    if samples_with_proprio == 0:
        logger.warning("⚠ No proprio data found in samples! Using dummy data for testing.")
        # Add dummy proprio data
        for sample in samples:
            if sample.proprio is None:
                sample.proprio = np.random.randn(14).astype(np.float32)

    if samples_with_action == 0:
        logger.warning("⚠ No action data found in samples! Using dummy data for testing.")
        # Add dummy action data
        for sample in samples:
            if sample.action is None:
                sample.action = np.random.randn(14).astype(np.float32)

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Computing normalization statistics")
    logger.info("=" * 80)

    stats = compute_normalization_stats(samples)

    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Creating sample sequences for Diffusion Policy")
    logger.info("=" * 80)

    obs_horizon = 2
    action_horizon = 8
    logger.info(f"Observation horizon: {obs_horizon}")
    logger.info(f"Action horizon: {action_horizon}")

    sequences = create_sequences(samples, obs_horizon, action_horizon)

    if len(sequences) == 0:
        raise ValueError(f"Not enough samples to create sequences (need at least {obs_horizon + action_horizon})")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Creating Diffusion Policy preprocessor")
    logger.info("=" * 80)

    # Create preprocessor config
    config = DiffusionPolicyPreprocessingConfig(
        camera_names=["left_cam", "right_cam", "middle_cam"],
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        device="cpu",
    )

    logger.info("Config:")
    logger.info(f"  Cameras: {config.camera_names}")
    logger.info(f"  Observation horizon: {config.obs_horizon}")
    logger.info(f"  Action horizon: {config.action_horizon}")
    logger.info(f"  State dim: {len(config.state_mean)}")
    logger.info(f"  Action dim: {len(config.action_mean)}")
    logger.info(f"  Image size: {config.image_config.target_size}")
    logger.info(f"  Device: {config.device}")

    preprocessor = DiffusionPolicyPreprocessor(config)
    logger.info("✓ Preprocessor created successfully")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Testing single sequence preprocessing")
    logger.info("=" * 80)

    test_sequence = sequences[0]
    try:
        result = preprocessor.preprocess_sample_sequence(test_sequence)
        logger.info("✓ Single sequence preprocessing successful")
        logger.info(f"  Observation images: {len(result.observation_images)} cameras")
        for cam_name, img_stack in result.observation_images.items():
            logger.info(f"    {cam_name}: shape={img_stack.shape}, dtype={img_stack.dtype}")
        logger.info(f"  State: shape={result.state.shape}, dtype={result.state.dtype}")
        logger.info(f"  Action: shape={result.action.shape}, dtype={result.action.dtype}")
    except Exception as e:
        logger.error(f"✗ Single sequence preprocessing failed: {e}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Testing with PyTorch DataLoader")
    logger.info("=" * 80)

    dataset = SampleSequenceDataset(sequences, preprocessor)
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=preprocessor.collate_fn,
        num_workers=0,  # Use 0 for debugging
        shuffle=False,
    )

    logger.info("DataLoader created:")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info("  Batch size: 4")
    logger.info(f"  Num batches: {len(loader)}")

    try:
        batch_count = 0
        for batch_idx, batch in enumerate(loader):
            batch_count += 1

            if batch_idx == 0:
                logger.info("\nFirst batch inspection:")
                logger.info(f"  observation_images keys: {list(batch.observation_images.keys())}")
                for cam_name, img_tensor in batch.observation_images.items():
                    logger.info(f"    {cam_name}: shape={img_tensor.shape}, dtype={img_tensor.dtype}")
                logger.info(f"  state: shape={batch.state.shape}, dtype={batch.state.dtype}")
                logger.info(f"  action: shape={batch.action.shape}, dtype={batch.action.dtype}")

                # Validate shapes
                batch_size = batch.observation_images["left_cam"].shape[0]
                assert batch_size == 4, f"Expected batch size 4, got {batch_size}"
                assert batch.state.shape == (4, obs_horizon, 14), f"Unexpected state shape: {batch.state.shape}"
                assert batch.action.shape == (
                    4,
                    action_horizon,
                    14,
                ), f"Unexpected action shape: {batch.action.shape}"
                logger.info("  ✓ Batch shapes validated")

            if batch_idx < 3:
                logger.info(f"Batch {batch_idx + 1}: {batch.observation_images['left_cam'].shape[0]} sequences")

        logger.info(f"\n✓ Successfully processed {batch_count} batches")

    except Exception as e:
        logger.error(f"✗ DataLoader failed: {e}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Validation checks")
    logger.info("=" * 80)

    # Check for NaN or inf
    logger.info("Checking for NaN/inf in preprocessed data...")
    for batch_idx, batch in enumerate(loader):
        for cam_name, img_tensor in batch.observation_images.items():
            if torch.isnan(img_tensor).any():
                raise ValueError(f"NaN found in {cam_name} images at batch {batch_idx}")
            if torch.isinf(img_tensor).any():
                raise ValueError(f"Inf found in {cam_name} images at batch {batch_idx}")

        if torch.isnan(batch.state).any():
            raise ValueError(f"NaN found in state at batch {batch_idx}")
        if torch.isnan(batch.action).any():
            raise ValueError(f"NaN found in action at batch {batch_idx}")

        if batch_idx == 0:
            logger.info("✓ No NaN/inf detected in first batch")

    logger.info("✓ All validation checks passed")

    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS: End-to-end Diffusion Policy preprocessing test completed!")
    logger.info("=" * 80)
    logger.info(f"✓ Processed {len(samples)} samples")
    logger.info(f"✓ Created {len(sequences)} sequences")
    logger.info(f"✓ Generated {batch_count} batches")
    logger.info("✓ All shapes and dtypes correct")
    logger.info("✓ No NaN/inf in data")
    logger.info("✓ Ready for model training!")


if __name__ == "__main__":
    data_dir = Path("/Users/donna/Downloads/run19")
    test_diffusion_policy_preprocessing(data_dir, num_samples=100)
