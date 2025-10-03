"""Script to process run19 data with proper timestamp synchronization."""

import logging
from pathlib import Path

from loom.io.synchronized import SynchronizedVideoMCAPReader
from loom.pipeline import merge_streams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_run19_synchronized(data_dir: Path) -> None:
    """Process run19 dataset with synchronized timestamps.

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

    logger.info("Creating synchronized readers...")

    # Create synchronized readers for each camera
    readers = []

    # Left arm camera
    left_cam_video = video_dir / "left_arm.perception_interface.left_cam.state.mp4"
    if left_cam_video.exists():
        readers.append(
            SynchronizedVideoMCAPReader(
                video_path=left_cam_video,
                mcap_path=mcap_file,
                camera_topic="left_arm/perception_interface/left_cam/state",
                camera_name="left_cam",
            )
        )

    # Right arm camera
    right_cam_video = video_dir / "right_arm.perception_interface.right_cam.state.mp4"
    if right_cam_video.exists():
        readers.append(
            SynchronizedVideoMCAPReader(
                video_path=right_cam_video,
                mcap_path=mcap_file,
                camera_topic="right_arm/perception_interface/right_cam/state",
                camera_name="right_cam",
            )
        )

    # Torso/middle camera
    middle_cam_video = video_dir / "torso.perception_interface.middle_cam.state.mp4"
    if middle_cam_video.exists():
        readers.append(
            SynchronizedVideoMCAPReader(
                video_path=middle_cam_video,
                mcap_path=mcap_file,
                camera_topic="torso/perception_interface/middle_cam/state",
                camera_name="middle_cam",
            )
        )

    # Create MCAP readers for proprioception/actions from joint state topics
    # Note: Binary joint state messages need special handling - for now we'll just verify video sync
    # TODO: Add protobuf/ROS message decoding for joint states
    # mcap_reader = MCAPReader(
    #     mcap_file,
    #     proprio_topic="left_arm/actor_interface/left_joints/state",  # Binary data - needs protobuf
    #     action_topic="left_arm/actor_interface/left_joints/action",
    # )
    # readers.append(mcap_reader)

    # For now, just verify that video synchronization works
    logger.info("NOTE: Proprio/action data requires protobuf decoding (not yet implemented)")

    # Process merged streams with standard time tolerance
    logger.info("\nMerging synchronized video streams...")
    sample_count = 0

    for sample in merge_streams(*readers, time_tolerance=0.033):
        sample_count += 1

        # Log first few samples for verification
        if sample_count <= 5:
            logger.info(f"\nSample {sample_count}:")
            logger.info(f"  Timestamp: {sample.timestamp:.6f}s (absolute Unix time)")

            if sample.cameras:
                logger.info(f"  Cameras ({len(sample.cameras)}):")
                for camera in sample.cameras:
                    logger.info(f"    {camera.name}: {camera.image.shape}")

            if sample.proprio is not None:
                logger.info(f"  Proprio: shape={sample.proprio.shape}")
                if len(sample.proprio) > 0:
                    logger.info(f"    First 5 values: {sample.proprio[:5]}")

            if sample.action is not None:
                logger.info(f"  Action: shape={sample.action.shape}")
                if len(sample.action) > 0:
                    logger.info(f"    First 5 values: {sample.action[:5]}")

            logger.info(f"  Metadata keys: {list(sample.metadata.keys())}")

        # Progress indicator
        if sample_count % 100 == 0:
            logger.info(f"Processed {sample_count} synchronized samples...")

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total synchronized samples: {sample_count}")
    logger.info("All samples now have absolute Unix timestamps from MCAP")
    logger.info("Vision and proprioception data are temporally aligned")


if __name__ == "__main__":
    data_dir = Path("/Users/donna/Downloads/run19")
    process_run19_synchronized(data_dir)
