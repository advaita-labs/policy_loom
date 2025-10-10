"""Convert MP4 + MCAP recordings into a LeRobot dataset episode."""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import numpy as np

from mcap_ros2.reader import read_ros2_messages

from loom.io.lerobot import LeRobotDatasetWriter
from loom.preprocessing.utils import filter_samples_by_cameras
from loom.pipeline.merge import merge_streams
from loom.io.synchronized import SynchronizedVideoMCAPReader


def _load_joint_timeseries(mcap_path: Path, topics: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ROS2 JointState vectors for the requested topics."""
    data: dict[str, list[float]] = {topic: [] for topic in topics}
    values: dict[str, list[np.ndarray]] = {topic: [] for topic in topics}

    with mcap_path.open("rb") as stream:
        for message in read_ros2_messages(stream):
            topic = message.channel.topic
            if topic not in data:
                continue
            msg = message.ros_msg
            data[topic].append(message.publish_time_ns * 1e-9)
            values[topic].append(np.array(msg.position, dtype=np.float32))

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for topic in topics:
        times = np.array(data[topic], dtype=np.float64)
        series = np.stack(values[topic]) if values[topic] else np.zeros((0,), dtype=np.float32)
        result[topic] = (times, series)
    return result


def _nearest(times: np.ndarray, values: np.ndarray, timestamp: float) -> np.ndarray:
    if len(times) == 0:
        raise ValueError("Requested joint series is empty.")
    idx = bisect.bisect_left(times, timestamp)
    if idx == 0:
        return values[0]
    if idx >= len(times):
        return values[-1]
    before = times[idx - 1]
    after = times[idx]
    if (timestamp - before) <= (after - timestamp):
        return values[idx - 1]
    return values[idx]


def enrich_samples_with_joint_data(samples, joint_series, task: str):
    left_state_times, left_state_values = joint_series["left_arm/actor_interface/left_joints/state"]
    right_state_times, right_state_values = joint_series["right_arm/actor_interface/right_joints/state"]
    left_action_times, left_action_values = joint_series["left_arm/actor_interface/left_joints/action"]
    right_action_times, right_action_values = joint_series["right_arm/actor_interface/right_joints/action"]

    for sample in samples:
        timestamp = float(sample.timestamp)

        left_state = _nearest(left_state_times, left_state_values, timestamp)
        right_state = _nearest(right_state_times, right_state_values, timestamp)
        sample.proprio = np.concatenate([left_state, right_state]).astype(np.float32)

        left_action = _nearest(left_action_times, left_action_values, timestamp)
        right_action = _nearest(right_action_times, right_action_values, timestamp)
        action = np.concatenate([left_action, right_action]).astype(np.float32)
        if action.size < 32:
            padded = np.zeros(32, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        sample.action = action
        sample.metadata["task"] = task

    return samples


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    mcap_path = input_dir / "run19_0.mcap"

    video_map = {
        "left_cam": input_dir / "videos" / "left_arm.perception_interface.left_cam.state.mp4",
        "right_cam": input_dir / "videos" / "right_arm.perception_interface.right_cam.state.mp4",
        "middle_cam": input_dir / "videos" / "torso.perception_interface.middle_cam.state.mp4",
    }

    readers = [
        SynchronizedVideoMCAPReader(video_path=video_map["left_cam"], mcap_path=mcap_path, camera_topic="left_arm/perception_interface/left_cam/state", camera_name="left_cam"),
        SynchronizedVideoMCAPReader(video_path=video_map["right_cam"], mcap_path=mcap_path, camera_topic="right_arm/perception_interface/right_cam/state", camera_name="right_cam"),
        SynchronizedVideoMCAPReader(video_path=video_map["middle_cam"], mcap_path=mcap_path, camera_topic="torso/perception_interface/middle_cam/state", camera_name="middle_cam"),
    ]

    merged_samples = list(merge_streams(*readers, time_tolerance=args.time_tolerance))
    merged_samples = list(merge_streams(*readers, time_tolerance=args.time_tolerance))
    filtered_samples = filter_samples_by_cameras(
        merged_samples,
        required_cameras=["left_cam", "right_cam", "middle_cam"],
    )
    if not filtered_samples:
        raise ValueError("No samples contain all required cameras after filtering.")

    joint_series = _load_joint_timeseries(
        mcap_path,
        [
            "left_arm/actor_interface/left_joints/state",
            "right_arm/actor_interface/right_joints/state",
            "left_arm/actor_interface/left_joints/action",
            "right_arm/actor_interface/right_joints/action",
        ],
    )

    enriched_samples = enrich_samples_with_joint_data(filtered_samples, joint_series, args.task)

    first_sample = enriched_samples[0]
    camera_shapes = {cam.name: tuple(cam.image.shape) for cam in first_sample.cameras}

    writer = LeRobotDatasetWriter(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        camera_names=["left_cam", "right_cam", "middle_cam"],
        action_dim=32,
        proprio_dim=enriched_samples[0].proprio.size,
        root=output_dir,
        use_videos=args.use_videos,
        camera_shapes=camera_shapes,
    )

    writer.add_episode(enriched_samples, task=args.task)
    writer.consolidate(push_to_hub=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 + MCAP recordings into a LeRobot dataset")
    parser.add_argument("--input", required=True, help="Path to recording directory containing run*.mcap, videos/")
    parser.add_argument("--output", required=True, help="Output directory for the LeRobot dataset")
    parser.add_argument("--task", required=True, help="Task description to store with each sample")
    parser.add_argument("--repo-id", default="local/run19", help="Dataset identifier")
    parser.add_argument("--robot-type", default="dual_arm", help="Robot type metadata")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--use-videos", action="store_true", help="Store frames as videos (default: images)")
    parser.add_argument("--time-tolerance", type=float, default=0.033, help="Merge tolerance between streams")
    args = parser.parse_args()

    main(args)
