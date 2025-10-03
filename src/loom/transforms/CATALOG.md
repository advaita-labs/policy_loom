# Transform Catalog

## Vision Transforms (Image-only)

| Short Name | Class | Module | Status |
|-----------|-------|--------|--------|
| `resize` | `Resize` | `loom.transforms.vision` | ✅ Planned |
| `crop` | `Crop` | `loom.transforms.vision` | ✅ Planned |
| `normalize` | `Normalize` | `loom.transforms.vision` | ✅ Planned |
| `color_convert` | `ColorConversion` | `loom.transforms.vision` | ✅ Planned |
| `rotate` | `Rotate` | `loom.transforms.vision` | ✅ Planned |
| `flip` | `Flip` | `loom.transforms.vision` | ✅ Planned |

## Time Transforms (Sequence-level)

| Short Name | Class | Module | Status |
|-----------|-------|--------|--------|
| `resample_fps` | `ResampleFPS` | `loom.transforms.time` | ✅ Planned |
| `window` | `Window` | `loom.transforms.time` | ✅ Planned |
| `align` | `Align` | `loom.transforms.time` | ✅ Planned |
| `subsample` | `Subsample` | `loom.transforms.time` | ✅ Planned |
| `deduplicate` | `Deduplicate` | `loom.transforms.time` | ✅ Planned |
| `temporal_shift` | `TemporalShift` | `loom.transforms.time` | ✅ Planned |

## Future Transforms

### Vision
- `brightness` - Adjust brightness
- `contrast` - Adjust contrast
- `blur` - Gaussian/motion blur
- `jpeg_compress` - Simulate compression artifacts
- `random_crop` - Data augmentation

### Time
- `interpolate` - Temporal interpolation
- `smooth` - Temporal smoothing (moving average)
- `detect_drops` - Flag dropped frames

### Action/Proprio
- `normalize_proprio` - Normalize joint values
- `clip_actions` - Clip action magnitudes
- `delta_to_absolute` - Convert delta actions to absolute

## Usage in Configs

Transforms are referenced by their short name in YAML configs:

```yaml
transforms:
  - type: resize
    height: 224
    width: 224
  - type: resample_fps
    target_fps: 10
  - type: normalize
    preset: imagenet
```

## Adding a New Transform

1. Subclass `Transform` ABC
2. Implement `__call__(self, sample: Sample) -> Sample`
3. Add docstring with usage example
4. Add tests in `tests/transforms/`
5. Update this catalog
6. Document in relevant README (vision/time)
