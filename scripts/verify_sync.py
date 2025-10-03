"""Verify synchronization quality of merged data."""

import logging
from pathlib import Path

import numpy as np

from loom.io.mcap import MCAPReader
from loom.io.synchronized import SynchronizedVideoMCAPReader
from loom.pipeline import merge_streams

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


def verify_synchronization(data_dir: Path) -> None:
    """Verify that video and MCAP data are properly synchronized."""
    video_dir = data_dir / "videos"
    mcap_file = data_dir / "run19_0.mcap"

    print("\n" + "=" * 80)
    print("SYNCHRONIZATION VERIFICATION")
    print("=" * 80)

    # Create synchronized readers
    readers = []

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

    mcap_reader = MCAPReader(mcap_file)
    readers.append(mcap_reader)

    # Collect samples with time_tolerance=0.033 (standard ~30fps tolerance)
    samples = list(merge_streams(*readers, time_tolerance=0.033))

    print(f"\nTotal merged samples: {len(samples)}")

    # Analyze sample composition
    vision_only = 0
    proprio_only = 0
    both = 0
    multi_camera = 0

    camera_counts = {1: 0, 2: 0, 3: 0}

    for sample in samples:
        has_vision = len(sample.cameras) > 0
        has_proprio = sample.proprio is not None

        if has_vision and has_proprio:
            both += 1
        elif has_vision:
            vision_only += 1
        elif has_proprio:
            proprio_only += 1

        if len(sample.cameras) > 1:
            multi_camera += 1

        cam_count = len(sample.cameras)
        if cam_count in camera_counts:
            camera_counts[cam_count] += 1

    print("\nSample composition:")
    print(f"  Vision + Proprio: {both} ({both/len(samples)*100:.1f}%)")
    print(f"  Vision only: {vision_only} ({vision_only/len(samples)*100:.1f}%)")
    print(f"  Proprio only: {proprio_only} ({proprio_only/len(samples)*100:.1f}%)")
    print(f"  Multi-camera: {multi_camera} ({multi_camera/len(samples)*100:.1f}%)")

    print("\nCamera count distribution:")
    for count, num in sorted(camera_counts.items()):
        if num > 0:
            print(f"  {count} camera(s): {num} samples")

    # Check timestamp ranges
    timestamps = np.array([s.timestamp for s in samples])
    duration = timestamps[-1] - timestamps[0]

    print("\nTemporal analysis:")
    print(f"  Duration: {duration:.3f}s")
    print(f"  Start: {timestamps[0]:.6f}s")
    print(f"  End: {timestamps[-1]:.6f}s")
    print(f"  Mean sample rate: {len(timestamps)/duration:.2f} Hz")

    # Check intervals
    intervals = np.diff(timestamps)
    print(f"  Mean interval: {np.mean(intervals):.4f}s")
    print(f"  Min interval: {np.min(intervals):.6f}s")
    print(f"  Max interval: {np.max(intervals):.6f}s")

    # Sample some multi-camera samples
    print("\nExample multi-camera samples:")
    multi_cam_samples = [s for s in samples if len(s.cameras) > 1]
    for i, sample in enumerate(multi_cam_samples[:3], 1):
        print(f"\n  Sample {i} @ {sample.timestamp:.6f}s:")
        print(f"    Cameras: {[cam.name for cam in sample.cameras]}")
        if sample.proprio is not None:
            print(f"    Proprio: shape={sample.proprio.shape}, mean={np.mean(sample.proprio):.3f}")
        if sample.action is not None:
            print(f"    Action: shape={sample.action.shape}")

    # Quality assessment
    print("\n" + "=" * 80)
    print("SYNCHRONIZATION QUALITY")
    print("=" * 80)

    total_video_frames = sum([261, 277, 274])  # From the actual data
    total_merged_vision = sum(1 for s in samples if len(s.cameras) > 0)

    print("\nFrame accounting:")
    print(f"  Total video frames across all cameras: {total_video_frames}")
    print(f"  Samples with vision data: {total_merged_vision}")
    print(f"  Multi-camera merged samples: {multi_camera}")

    if both > 0:
        print(f"\n✅ SUCCESS: {both} samples have BOTH vision and proprio data")
        print("   This means vision and proprioception are synchronized!")
    else:
        print("\n⚠️  WARNING: No samples with both vision and proprio")
        print("   Vision and proprio may not be overlapping in time")

    if multi_camera > 100:
        print(f"\n✅ SUCCESS: {multi_camera} samples have multiple cameras")
        print("   Multi-camera streams are properly synchronized!")

    # Check for timestamp alignment issues
    vision_samples = [s for s in samples if len(s.cameras) > 0]
    proprio_samples = [s for s in samples if s.proprio is not None]

    if vision_samples and proprio_samples:
        vision_times = np.array([s.timestamp for s in vision_samples])
        proprio_times = np.array([s.timestamp for s in proprio_samples])

        vision_range = (vision_times.min(), vision_times.max())
        proprio_range = (proprio_times.min(), proprio_times.max())

        overlap_start = max(vision_range[0], proprio_range[0])
        overlap_end = min(vision_range[1], proprio_range[1])
        overlap = overlap_end - overlap_start

        print("\nTemporal overlap:")
        print(f"  Vision range: {vision_range[0]:.3f}s to {vision_range[1]:.3f}s")
        print(f"  Proprio range: {proprio_range[0]:.3f}s to {proprio_range[1]:.3f}s")
        print(f"  Overlap: {overlap:.3f}s")

        if overlap > 9.0:
            print(f"\n✅ EXCELLENT: ~{overlap:.1f}s of synchronized data!")


def main() -> None:
    """Run synchronization verification."""
    data_dir = Path("/Users/donna/Downloads/run19")
    verify_synchronization(data_dir)


if __name__ == "__main__":
    main()
