# Critical Merge Pipeline Bug Fixes

## What Was Broken

Three bugs in `src/loom/pipeline/merge.py` would corrupt VLA training data:

1. **Proprio concatenation**: Concatenated multiple samples → wrong dimensions → training crash
2. **Action selection**: Picked first sample (arbitrary) → wrong action labels → bad policy
3. **Timestamp matching**: Used first match instead of closest → misaligned data

## The Fix

All three now use **nearest neighbor** temporal matching:

```python
# Select temporally closest sample when multiple candidates exist
closest = min(samples, key=lambda s: abs(s.timestamp - merged_timestamp))
```

**Why nearest neighbor?**
- Most accurate for discrete robot states
- Actions can't be interpolated
- Keeps code simple (one strategy for everything)

## Tests

Added 3 tests covering each bug, all passing (51 total, 78% coverage).

Verified on real run19 data: 258 merged samples, 217 multi-camera (84%) synchronized.
