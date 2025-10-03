# Vision Transforms

## Purpose

Stateless transformations for the `rgb` field of `Sample` objects. All transforms:
- Take a `Sample` with `rgb` field
- Return a `Sample` with modified `rgb` field
- Preserve all other fields (`timestamp`, `proprio`, `action`, `metadata`)
- Are deterministic (same input → same output)

## Available Transforms

### `Resize`

**Purpose**: Resize images to a target resolution.

**Config**:
```yaml
- type: resize
  height: 224
  width: 224
  interpolation: bilinear  # bilinear, nearest, bicubic
```

**Behavior**:
- Uses OpenCV or PIL for resizing
- Preserves aspect ratio if `maintain_aspect_ratio: true`
- Updates `metadata['original_shape']` if not present

---

### `Crop`

**Purpose**: Crop images to a region of interest.

**Config**:
```yaml
- type: crop
  mode: center  # center, random, bbox
  height: 224
  width: 224
```

**Modes**:
- `center`: Crop from center
- `random`: Random crop (requires seed for determinism)
- `bbox`: Crop to bounding box (requires `bbox` in metadata)

---

### `ColorConversion`

**Purpose**: Convert color space (BGR ↔ RGB, grayscale, etc.).

**Config**:
```yaml
- type: color_convert
  source: bgr
  target: rgb
```

**Common conversions**:
- `bgr → rgb` (OpenCV uses BGR by default)
- `rgb → grayscale`
- `rgb → hsv`

---

### `Normalize`

**Purpose**: Normalize pixel values for neural network input.

**Config**:
```yaml
- type: normalize
  mean: [0.485, 0.456, 0.406]  # ImageNet mean
  std: [0.229, 0.224, 0.225]   # ImageNet std
  scale: 255.0  # divide by 255 first (if input is uint8)
```

**Behavior**:
- Converts to float32
- Applies: `(pixel / scale - mean) / std`
- Common presets: `imagenet`, `clip`, `zero_one`

---

### `Rotate`

**Purpose**: Rotate images by a fixed angle.

**Config**:
```yaml
- type: rotate
  angle: 90  # degrees, clockwise
  expand: false  # expand canvas to fit rotated image
```

---

### `Flip`

**Purpose**: Flip images horizontally or vertically.

**Config**:
```yaml
- type: flip
  direction: horizontal  # horizontal, vertical
```

**Note**: Also flips `action` if `flip_action: true` (for horizontal flips, mirrors lateral actions).

---

## Dependencies

- `numpy` (required)
- `opencv-python` (optional, for fast resize/rotate)
- `pillow` (optional, fallback for image ops)

Install with:
```bash
uv pip install opencv-python
```

## Usage Example

```python
from loom.transforms.vision import Resize, Normalize
from loom.core import Sample
import numpy as np

sample = Sample(
    timestamp=1.0,
    rgb=np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
)

# Apply transforms
resize = Resize(height=224, width=224)
normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

sample = resize(sample)
sample = normalize(sample)

print(sample.rgb.shape)  # (224, 224, 3)
print(sample.rgb.dtype)  # float32
```

## Design Notes

- **Stateless**: No internal state between calls
- **Composable**: Can chain transforms freely
- **Typed**: Input/output are `Sample` objects, not raw arrays
- **Fast**: Use vectorized numpy/OpenCV operations
- **Safe**: Validate inputs, raise `ValueError` for invalid samples
