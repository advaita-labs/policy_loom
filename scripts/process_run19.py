"""Script to process run19 data (videos + MCAP)."""

import logging
from pathlib import Path

from loom.io.mcap import MCAPReader
from loom.io.mp4 import MP4Reader
from loom.pipeline import merge_streams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_run19(data_dir: Path) -> None:
    """Process run19 dataset with multiple cameras and MCAP data.

    Args:
        data_dir: Root directory containing videos/ and mcap file
    """
    video_dir = data_dir / "videos"
    mcap_file = data_dir / "run19_0.mcap"

    # Verify files exist
    if not video_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {video_dir}")
    if not mcap_file.exists():
        raise FileNotFoundError(f"MCAP file not found: {mcap_file}")

    # List video files
    video_files = sorted(video_dir.glob("*.mp4"))
    logger.info(f"Found {len(video_files)} video files:")
    for vf in video_files:
        logger.info(f"  - {vf.name}")

    # Create readers for each video
    readers = []

    # Left arm camera
    left_cam_file = video_dir / "left_arm.perception_interface.left_cam.state.mp4"
    if left_cam_file.exists():
        readers.append(MP4Reader(left_cam_file, camera_name="left_cam"))

    # Right arm camera
    right_cam_file = video_dir / "right_arm.perception_interface.right_cam.state.mp4"
    if right_cam_file.exists():
        readers.append(MP4Reader(right_cam_file, camera_name="right_cam"))

    # Torso camera
    torso_cam_file = video_dir / "torso.perception_interface.middle_cam.state.mp4"
    if torso_cam_file.exists():
        readers.append(MP4Reader(torso_cam_file, camera_name="middle_cam"))

    # MCAP reader - we'll inspect available topics first
    logger.info(f"\nInspecting MCAP file: {mcap_file.name}")
    mcap_reader = MCAPReader(mcap_file)
    readers.append(mcap_reader)

    # Process merged streams
    logger.info("\nMerging streams...")
    sample_count = 0

    for sample in merge_streams(*readers):
        sample_count += 1

        # Log first few samples for inspection
        if sample_count <= 3:
            logger.info(f"\nSample {sample_count}:")
            logger.info(f"  Timestamp: {sample.timestamp:.4f}s")

            if sample.cameras:
                logger.info(f"  Cameras ({len(sample.cameras)}):")
                for camera in sample.cameras:
                    logger.info(f"    {camera.name}: {camera.image.shape}")

            if sample.proprio is not None:
                logger.info(f"  Proprio: shape={sample.proprio.shape}, values={sample.proprio[:5]}...")

            if sample.action is not None:
                logger.info(f"  Action: shape={sample.action.shape}, values={sample.action[:5]}...")

            logger.info(f"  Metadata keys: {list(sample.metadata.keys())}")

        # Process every 100 samples
        if sample_count % 100 == 0:
            logger.info(f"Processed {sample_count} samples...")

    logger.info(f"\nFinished processing {sample_count} total samples")


if __name__ == "__main__":
    data_dir = Path("/Users/donna/Downloads/run19")
    process_run19(data_dir)
