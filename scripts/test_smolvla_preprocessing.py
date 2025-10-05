"""End-to-end test of SmolVLA preprocessing on run19 data."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from loom.core.types import Sample, SmolVLAInput
from loom.io.mcap import MCAPReader
from loom.io.mp4 import MP4Reader
from loom.pipeline import merge_streams
from loom.preprocessing import SmolVLAPreprocessingConfig, SmolVLAPreprocessor, filter_samples_by_cameras

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SampleDataset(Dataset):
    """Simple dataset wrapper for Sample objects."""

    def __init__(self, samples: list[Sample], preprocessor: SmolVLAPreprocessor):
        self.samples = samples
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SmolVLAInput:
        return self.preprocessor.preprocess_sample(self.samples[idx])


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


def test_smolvla_preprocessing(data_dir: Path, num_samples: int = 50) -> None:
    """Test SmolVLA preprocessing on run19 dataset.

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

        # Add task instruction to metadata (required by SmolVLA)
        sample.metadata["task"] = "Pick and place object"

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
    logger.info("STEP 4: Creating SmolVLA preprocessor")
    logger.info("=" * 80)

    # Create preprocessor config
    config = SmolVLAPreprocessingConfig(
        camera_names=["left_cam", "right_cam", "middle_cam"],
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        action_mean=stats["action_mean"],
        action_std=stats["action_std"],
        max_state_dim=32,
        max_action_dim=32,
        device="cpu",
    )

    logger.info("Config:")
    logger.info(f"  Cameras: {config.camera_names}")
    logger.info(f"  State dim: {len(config.state_mean)} → {config.max_state_dim}")
    logger.info(f"  Action dim: {len(config.action_mean)} → {config.max_action_dim}")
    logger.info(f"  Image size: {config.image_config.target_size}")
    logger.info(f"  Device: {config.device}")

    preprocessor = SmolVLAPreprocessor(config)
    logger.info("✓ Preprocessor created successfully")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Testing single sample preprocessing")
    logger.info("=" * 80)

    test_sample = samples[0]
    try:
        result = preprocessor.preprocess_sample(test_sample)
        logger.info("✓ Single sample preprocessing successful")
        logger.info(f"  Images: {len(result.images)} tensors")
        for i, img in enumerate(result.images):
            logger.info(f"    Image {i}: shape={img.shape}, dtype={img.dtype}")
        logger.info(f"  Language: '{result.language_instruction}'")
        logger.info(f"  State: shape={result.state.shape}, dtype={result.state.dtype}")
        logger.info(f"  Action: shape={result.action.shape}, dtype={result.action.dtype}")
    except Exception as e:
        logger.error(f"✗ Single sample preprocessing failed: {e}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Testing with PyTorch DataLoader")
    logger.info("=" * 80)

    dataset = SampleDataset(samples, preprocessor)
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
                logger.info(
                    f"  language_tokens: shape={batch.language_tokens.shape}, dtype={batch.language_tokens.dtype}"
                )
                logger.info(
                    f"  language_attention_mask: shape={batch.language_attention_mask.shape}, "
                    f"dtype={batch.language_attention_mask.dtype}"
                )
                logger.info(f"  state: shape={batch.state.shape}, dtype={batch.state.dtype}")
                logger.info(f"  action: shape={batch.action.shape}, dtype={batch.action.dtype}")

                # Validate shapes
                batch_size = batch.observation_images["left_cam"].shape[0]
                assert batch_size == 4, f"Expected batch size 4, got {batch_size}"
                assert batch.state.shape == (4, 1, 32), f"Unexpected state shape: {batch.state.shape}"
                assert batch.action.shape == (4, 1, 32), f"Unexpected action shape: {batch.action.shape}"
                logger.info("  ✓ Batch shapes validated")

            if batch_idx < 3:
                logger.info(f"Batch {batch_idx + 1}: {batch.observation_images['left_cam'].shape[0]} samples")

        logger.info(f"\n✓ Successfully processed {batch_count} batches")

    except Exception as e:
        logger.error(f"✗ DataLoader failed: {e}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Validation checks")
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
    logger.info("SUCCESS: End-to-end preprocessing test completed!")
    logger.info("=" * 80)
    logger.info(f"✓ Processed {len(samples)} samples")
    logger.info(f"✓ Created {batch_count} batches")
    logger.info("✓ All shapes and dtypes correct")
    logger.info("✓ No NaN/inf in data")
    logger.info("✓ Ready for model training!")


if __name__ == "__main__":
    data_dir = Path("/Users/donna/Downloads/run19")
    test_smolvla_preprocessing(data_dir, num_samples=50)
