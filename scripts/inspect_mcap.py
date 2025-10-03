"""Inspect MCAP file to understand channel structure."""

import json
import logging
from pathlib import Path

from mcap.reader import make_reader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def inspect_mcap(mcap_path: Path) -> None:
    """Inspect MCAP file structure and sample messages."""
    logger.info(f"Inspecting: {mcap_path.name}\n")

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)

        # Get summary
        summary = reader.get_summary()
        if summary and summary.statistics:
            logger.info(f"Total messages: {summary.statistics.message_count}")
            logger.info(f"Channels: {len(summary.statistics.channel_message_counts)}\n")

        # Collect channel info
        channels_info = {}
        for schema, channel, message in reader.iter_messages():
            if channel.id not in channels_info:
                channels_info[channel.id] = {
                    "id": channel.id,
                    "topic": channel.topic,
                    "count": 0,
                    "first_timestamp": message.log_time / 1e9,
                    "last_timestamp": message.log_time / 1e9,
                    "sample_messages": [],
                }

            info = channels_info[channel.id]
            info["count"] += 1
            info["last_timestamp"] = message.log_time / 1e9

            # Store first 2 sample messages
            if len(info["sample_messages"]) < 2:
                try:
                    decoded = json.loads(message.data.decode("utf-8"))
                    info["sample_messages"].append({"timestamp": message.log_time / 1e9, "data": decoded})
                except (UnicodeDecodeError, json.JSONDecodeError):
                    info["sample_messages"].append(
                        {"timestamp": message.log_time / 1e9, "data": f"<binary: {len(message.data)} bytes>"}
                    )

    # Print channel information
    logger.info("=" * 100)
    logger.info("CHANNELS")
    logger.info("=" * 100)

    for channel_id in sorted(channels_info.keys()):
        info = channels_info[channel_id]
        duration = info["last_timestamp"] - info["first_timestamp"]
        rate = info["count"] / duration if duration > 0 else 0

        logger.info(f"\nChannel {channel_id}: {info['topic']}")
        logger.info(f"  Messages: {info['count']}")
        logger.info(f"  Duration: {duration:.3f}s")
        logger.info(f"  Rate: {rate:.2f} Hz")
        logger.info(f"  Time range: {info['first_timestamp']:.3f}s to {info['last_timestamp']:.3f}s")

        # Show sample messages
        logger.info("  Sample messages:")
        for i, sample in enumerate(info["sample_messages"], 1):
            logger.info(f"    Message {i} @ {sample['timestamp']:.3f}s:")
            if isinstance(sample["data"], dict):
                # Pretty print first level keys
                logger.info(f"      Keys: {list(sample['data'].keys())}")
                # Show first few key-value pairs
                for key in list(sample["data"].keys())[:5]:
                    value = sample["data"][key]
                    if isinstance(value, (list, dict)):
                        logger.info(f"      {key}: {type(value).__name__} (len={len(value)})")
                    else:
                        logger.info(f"      {key}: {value}")
            else:
                logger.info(f"      {sample['data']}")


def main() -> None:
    """Run MCAP inspection."""
    mcap_path = Path("/Users/donna/Downloads/run19/run19_0.mcap")
    inspect_mcap(mcap_path)


if __name__ == "__main__":
    main()
