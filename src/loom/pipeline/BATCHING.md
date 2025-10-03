# Batching Strategy

## Overview

Batching is the process of grouping multiple samples for efficient processing. In `policy_loom`, batching happens **late** (at training time), not during preprocessing.

## Why Late Batching?

**Preprocessing**: Store samples individually
- Easier to filter/sample/shuffle
- No padding needed (variable-length episodes)
- Simpler validation and debugging

**Training**: Batch on-the-fly
- PyTorch `DataLoader` with `collate_fn`
- Dynamic batching based on memory constraints
- Mix episodes from different sources

## Preprocessing: No Batching

During preprocessing, samples flow **one at a time**:

```python
for sample in reader.read():
    sample = transform(sample)
    writer.write(sample)  # Write immediately
```

**Benefits**:
- Lower memory footprint
- Simpler error handling (no partial batches)
- Streaming-friendly (process infinite data)

**Trade-off**:
- Can't use batch operations (e.g., batch matrix multiply)
- For image transforms, this is fine (per-image resize is fast)

## Training: Dynamic Batching

At training time, use PyTorch `DataLoader`:

```python
from torch.utils.data import DataLoader, Dataset

class OpenPIDataset(Dataset):
    def __getitem__(self, idx):
        episode = load_episode(idx)
        return {
            "rgb": episode["observations"]["rgb"],
            "proprio": episode["observations"]["proprio"],
            "action": episode["actions"],
        }

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_variable_length,  # Handle variable episode lengths
    num_workers=4,
)

for batch in loader:
    # batch["rgb"].shape = (32, T, H, W, C) with padding
    loss = model(batch)
```

## Backpressure and Memory Limits

**Problem**: Reading faster than writing can cause memory bloat.

**Solution**: Use bounded queues (backpressure).

```python
from queue import Queue

sample_queue = Queue(maxsize=100)  # Buffer up to 100 samples

# Producer (reader + transforms)
for sample in reader.read():
    sample = transform(sample)
    sample_queue.put(sample)  # Blocks if queue is full

# Consumer (writer)
while not done:
    sample = sample_queue.get()
    writer.write(sample)
```

**Benefits**:
- Limits memory usage
- Smooth out read/write speed mismatches
- Graceful degradation under load

## Buffering in Writer

Writers may buffer samples before flushing to disk:

```python
class OpenPIWriter:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def write(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        # Write all buffered samples to parquet
        df = pd.DataFrame(self.buffer)
        df.to_parquet(self.parquet_path, append=True)
        self.buffer.clear()
```

**Trade-offs**:
- **Larger buffer**: Fewer I/O ops, higher memory
- **Smaller buffer**: More I/O ops, lower memory

**Recommendation**: 1000-10000 samples (adjust based on sample size).

## Parallelization with Batching

For multi-core preprocessing, batch samples before sending to workers:

```python
from multiprocessing import Pool

def process_batch(samples):
    return [transform(s) for s in samples]

with Pool(8) as pool:
    batches = chunk(reader.read(), batch_size=100)
    for processed_batch in pool.imap(process_batch, batches):
        for sample in processed_batch:
            writer.write(sample)
```

**When to use**:
- Heavy CPU transforms (e.g., large image resizing)
- Multi-core machines
- Batch transforms are faster than single-sample

**When to avoid**:
- Lightweight transforms (overhead not worth it)
- I/O bound (reading/writing is the bottleneck)

## Memory Budget

Estimate memory usage:

```python
# Per sample
rgb_memory = H * W * C * 4  # float32
proprio_memory = proprio_dims * 4
action_memory = action_dims * 4
total_per_sample = rgb_memory + proprio_memory + action_memory

# For buffer
buffer_memory = total_per_sample * buffer_size

# Example: 224x224x3 image, 6 proprio, 7 actions
# = (224*224*3 + 6 + 7) * 4 = 602,652 bytes = 0.6 MB per sample
# Buffer of 1000 = 600 MB
```

**Guideline**: Keep buffer under 10% of available RAM.

## Async I/O (Future)

For very large datasets, use async I/O:

```python
import asyncio

async def async_read():
    async for sample in reader.async_read():
        yield sample

async def async_write(sample):
    await writer.async_write(sample)

async def pipeline():
    async for sample in async_read():
        sample = transform(sample)
        await async_write(sample)
```

This decouples I/O from processing, maximizing throughput.

## Summary

| Stage | Batching | Why |
|-------|----------|-----|
| Preprocessing | **No** | Simpler, lower memory, streaming |
| Training | **Yes** | GPU efficiency, standard practice |
| Writer | **Buffered** | Reduce I/O ops, amortize overhead |
| Parallelization | **Optional** | Only for CPU-heavy transforms |
