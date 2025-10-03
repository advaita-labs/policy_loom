"""Diagnostic script to analyze timestamp synchronization between videos and MCAP."""

import logging
from pathlib import Path

import numpy as np

from loom.io.mcap import MCAPReader
from loom.io.mp4 import MP4Reader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_video_timestamps(video_path: Path, camera_name: str) -> dict:
    """Analyze timestamp distribution in a video file.

    Args:
        video_path: Path to video file
        camera_name: Camera identifier

    Returns:
        Dict with timestamp statistics
    """
    logger.info(f"\nAnalyzing {camera_name}: {video_path.name}")

    with MP4Reader(video_path, camera_name=camera_name) as reader:
        timestamps = []
        for sample in reader.read():
            timestamps.append(sample.timestamp)

    timestamps = np.array(timestamps)

    if len(timestamps) < 2:
        return {"camera": camera_name, "count": len(timestamps)}

    # Calculate frame intervals
    intervals = np.diff(timestamps)

    stats = {
        "camera": camera_name,
        "frame_count": len(timestamps),
        "duration": timestamps[-1] - timestamps[0],
        "start_time": timestamps[0],
        "end_time": timestamps[-1],
        "mean_interval": np.mean(intervals),
        "std_interval": np.std(intervals),
        "min_interval": np.min(intervals),
        "max_interval": np.max(intervals),
        "fps_calculated": 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0,
    }

    logger.info(f"  Frame count: {stats['frame_count']}")
    logger.info(f"  Duration: {stats['duration']:.3f}s")
    logger.info(f"  Start: {stats['start_time']:.3f}s, End: {stats['end_time']:.3f}s")
    logger.info(f"  Mean interval: {stats['mean_interval']:.4f}s ({stats['fps_calculated']:.2f} FPS)")
    logger.info(f"  Interval std dev: {stats['std_interval']:.4f}s")
    logger.info(f"  Min interval: {stats['min_interval']:.4f}s, Max: {stats['max_interval']:.4f}s")

    # Check for large gaps (potential dropped frames)
    large_gaps = intervals > (stats["mean_interval"] * 2)
    if np.any(large_gaps):
        gap_count = np.sum(large_gaps)
        logger.warning(f"  ⚠️  Found {gap_count} large gaps (>2x mean interval)")
        gap_indices = np.where(large_gaps)[0]
        for idx in gap_indices[:5]:  # Show first 5
            logger.warning(
                f"    Gap at frame {idx}: {intervals[idx]:.4f}s " f"(expected ~{stats['mean_interval']:.4f}s)"
            )

    return stats


def analyze_mcap_timestamps(mcap_path: Path) -> dict:
    """Analyze timestamp distribution in MCAP file.

    Args:
        mcap_path: Path to MCAP file

    Returns:
        Dict with timestamp statistics
    """
    logger.info(f"\nAnalyzing MCAP: {mcap_path.name}")

    with MCAPReader(mcap_path) as reader:
        timestamps = []
        for sample in reader.read():
            timestamps.append(sample.timestamp)

    timestamps = np.array(timestamps)

    if len(timestamps) < 2:
        return {"count": len(timestamps)}

    # Calculate message intervals
    timestamps_sorted = np.sort(timestamps)
    intervals = np.diff(timestamps_sorted)

    stats = {
        "message_count": len(timestamps),
        "duration": timestamps_sorted[-1] - timestamps_sorted[0],
        "start_time": timestamps_sorted[0],
        "end_time": timestamps_sorted[-1],
        "mean_interval": np.mean(intervals),
        "std_interval": np.std(intervals),
        "min_interval": np.min(intervals),
        "max_interval": np.max(intervals),
        "rate_hz": 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0,
    }

    logger.info(f"  Message count: {stats['message_count']}")
    logger.info(f"  Duration: {stats['duration']:.3f}s")
    logger.info(f"  Start: {stats['start_time']:.3f}s, End: {stats['end_time']:.3f}s")
    logger.info(f"  Mean interval: {stats['mean_interval']:.4f}s ({stats['rate_hz']:.2f} Hz)")
    logger.info(f"  Interval std dev: {stats['std_interval']:.4f}s")

    return stats


def check_synchronization(video_stats: list[dict], mcap_stats: dict) -> None:
    """Check if video and MCAP timestamps are properly synchronized.

    Args:
        video_stats: List of video statistics
        mcap_stats: MCAP statistics
    """
    logger.info("\n" + "=" * 80)
    logger.info("SYNCHRONIZATION ANALYSIS")
    logger.info("=" * 80)

    # Check time alignment
    video_start = min(s["start_time"] for s in video_stats)
    video_end = max(s["end_time"] for s in video_stats)
    mcap_start = mcap_stats["start_time"]
    mcap_end = mcap_stats["end_time"]

    logger.info("\nTime ranges:")
    logger.info(f"  Videos: {video_start:.3f}s to {video_end:.3f}s (duration: {video_end - video_start:.3f}s)")
    logger.info(f"  MCAP:   {mcap_start:.3f}s to {mcap_end:.3f}s (duration: {mcap_end - mcap_start:.3f}s)")

    # Check overlap
    overlap_start = max(video_start, mcap_start)
    overlap_end = min(video_end, mcap_end)
    overlap_duration = overlap_end - overlap_start

    if overlap_duration > 0:
        logger.info(f"\n✅ Overlap: {overlap_duration:.3f}s ({overlap_start:.3f}s to {overlap_end:.3f}s)")
    else:
        logger.error("\n❌ NO OVERLAP! Video and MCAP do not share any time range!")
        return

    # Check camera synchronization
    logger.info("\nCamera synchronization:")
    for i, cam1 in enumerate(video_stats):
        for cam2 in video_stats[i + 1 :]:
            time_diff = abs(cam1["start_time"] - cam2["start_time"])
            logger.info(f"  {cam1['camera']} vs {cam2['camera']}: {time_diff:.4f}s offset")

            if time_diff > 0.1:  # More than 100ms offset
                logger.warning("    ⚠️  Large offset between cameras!")

    # Check frame rate consistency
    logger.info("\nFrame rate analysis:")
    for stat in video_stats:
        variation_pct = (stat["std_interval"] / stat["mean_interval"]) * 100 if stat["mean_interval"] > 0 else 0
        logger.info(f"  {stat['camera']}: {stat['fps_calculated']:.2f} FPS (variation: {variation_pct:.1f}%)")

        if variation_pct > 5:
            logger.warning("    ⚠️  High frame time variation - possible dropped frames!")

    # Calculate expected vs actual sync quality
    logger.info("\nSync quality metrics:")
    time_tolerance = 0.033  # 33ms tolerance (as used in merge_streams)

    for stat in video_stats:
        # How many video frames fall within the overlap period?
        frames_in_overlap = sum(
            1
            for _ in range(stat["frame_count"])
            if overlap_start <= stat["start_time"] + _ * stat["mean_interval"] <= overlap_end
        )
        logger.info(f"  {stat['camera']}: ~{frames_in_overlap} frames in overlap period")

    # Estimate merge quality
    total_video_samples = sum(s["frame_count"] for s in video_stats)
    mcap_samples_in_overlap = int(overlap_duration / mcap_stats["mean_interval"])

    logger.info("\nExpected merge results:")
    logger.info(f"  Total video samples: {total_video_samples}")
    logger.info(f"  MCAP samples in overlap: ~{mcap_samples_in_overlap}")
    logger.info(f"  Time tolerance for merging: {time_tolerance * 1000:.1f}ms")

    # Provide recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)

    issues_found = False

    # Check for time alignment issues
    if abs(video_start - mcap_start) > 1.0:
        logger.warning("⚠️  Videos and MCAP have different start times (>1s difference)")
        logger.warning("   Consider using MCAP timestamps to align video start times")
        issues_found = True

    # Check for dropped frames
    for stat in video_stats:
        if stat["std_interval"] / stat["mean_interval"] > 0.05:
            logger.warning(f"⚠️  {stat['camera']} has irregular frame timing")
            logger.warning("   Consider extracting actual timestamps from video metadata")
            issues_found = True

    if not issues_found:
        logger.info("✅ Synchronization looks good!")
        logger.info("   Videos and MCAP appear to be properly aligned.")
    else:
        logger.info("\n🔧 Action items:")
        logger.info("   1. Implement video timestamp extraction from metadata")
        logger.info("   2. Add timestamp alignment correction")
        logger.info("   3. Add validation tests for synchronization")


def main() -> None:
    """Run synchronization analysis."""
    data_dir = Path("/Users/donna/Downloads/run19")
    video_dir = data_dir / "videos"
    mcap_file = data_dir / "run19_0.mcap"

    # Analyze each video
    video_stats = []

    left_cam = video_dir / "left_arm.perception_interface.left_cam.state.mp4"
    if left_cam.exists():
        video_stats.append(analyze_video_timestamps(left_cam, "left_cam"))

    right_cam = video_dir / "right_arm.perception_interface.right_cam.state.mp4"
    if right_cam.exists():
        video_stats.append(analyze_video_timestamps(right_cam, "right_cam"))

    middle_cam = video_dir / "torso.perception_interface.middle_cam.state.mp4"
    if middle_cam.exists():
        video_stats.append(analyze_video_timestamps(middle_cam, "middle_cam"))

    # Analyze MCAP
    mcap_stats = analyze_mcap_timestamps(mcap_file)

    # Check synchronization
    check_synchronization(video_stats, mcap_stats)


if __name__ == "__main__":
    main()
