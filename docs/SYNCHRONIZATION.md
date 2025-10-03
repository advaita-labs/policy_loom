# Video-MCAP Synchronization

## Problem

Videos use relative time (0.0s), MCAP uses absolute Unix time (1757503161s) → no temporal overlap.

## Solution

`SynchronizedVideoMCAPReader` extracts MCAP camera timestamps and applies them to video frames:

```python
from loom.io.synchronized import SynchronizedVideoMCAPReader

reader = SynchronizedVideoMCAPReader(
    video_path="left_cam.mp4",
    mcap_path="data.mcap",
    camera_topic="left_arm/perception_interface/left_cam/state",
    camera_name="left_cam"
)
```

Now video frames have correct absolute timestamps for merging with robot telemetry.

## Usage

```python
from loom.io.synchronized import SynchronizedVideoMCAPReader
from loom.pipeline import merge_streams

left_cam = SynchronizedVideoMCAPReader(...)
right_cam = SynchronizedVideoMCAPReader(...)

# All samples now share same time base
for sample in merge_streams(left_cam, right_cam, time_tolerance=0.033):
    # sample.cameras has synchronized multi-camera frames
    pass
```
