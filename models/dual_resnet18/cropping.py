"""Square crops around a deer's bounding box, shared by training and inference.

Input:  a full-resolution uint8 image tensor (3, H, W) and a normalized [x, y, w, h] box
        (top-left origin, 0-1; same format as the split CSV bbox columns and MegaDetector)
Output: a uint8 tensor (3, out_size, out_size)

Resizing a 1600x1200 frame straight to 224x224 shrinks fawn spots and antler tips to a pixel or two.
Cropping a square around the deer first keeps them, and a square (not the box's own shape) means the
resize never stretches the deer.
"""

import math  # floor/ceil for pixel-exact crop windows
import random  # bbox jitter augmentation

import torch  # tensor canvas for padding
from torchvision.transforms.v2 import functional as tvf  # antialiased resize

IMAGENET_MEAN_UINT8 = (124, 116, 104)
CROP_METHODS = ('square', 'letterbox')

Box = tuple[float, float, float, float]


def square_window(img_w: int, img_h: int, bbox: Box, margin: float) -> tuple[int, int, int]:
    """Compute the square crop window for a box, shifted inward so it stays inside the image.

    The side is max(box w, box h) * (1 + margin), and at least 2 px larger than the box so pixel
    rounding can never cut the box. The window is centered on the box, then clamped: when the side
    fits in the image, the window shifts inward until it is fully inside; when it doesn't fit, the
    window is clamped so it covers that whole image dimension and the overflow is padded later.

    Args:
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        bbox: Normalized [x, y, w, h] box (top-left origin).
        margin: Extra context around the box as a fraction of its longer side.

    Returns:
        (x0, y0, side) in pixels. x0/y0 may be negative and x0 + side may exceed img_w when the
        side is larger than the image.
    """
    x, y, w, h = bbox
    bx0 = math.floor(max(x, 0.0) * img_w)
    by0 = math.floor(max(y, 0.0) * img_h)
    bx1 = math.ceil(min(x + w, 1.0) * img_w)
    by1 = math.ceil(min(y + h, 1.0) * img_h)
    longest = max(bx1 - bx0, by1 - by0, 1)
    side = max(math.ceil(longest * (1 + margin)), longest + 2)

    def clamp_start(center: float, size: int) -> int:
        lo, hi = min(0, size - side), max(0, size - side)
        return min(max(math.floor(center - side / 2), lo), hi)

    return clamp_start((bx0 + bx1) / 2, img_w), clamp_start((by0 + by1) / 2, img_h), side


def square_crop(image: torch.Tensor, bbox: Box, margin: float = 0.15, out_size: int = 224) -> torch.Tensor:
    """Crop a square around the box, pad any part outside the image with the ImageNet mean, resize.

    Args:
        image: uint8 tensor (3, H, W), full resolution.
        bbox: Normalized [x, y, w, h] box (top-left origin).
        margin: Extra context around the box as a fraction of its longer side.
        out_size: Output height and width in pixels.

    Returns:
        uint8 tensor (3, out_size, out_size).
    """
    _, img_h, img_w = image.shape
    x0, y0, side = square_window(img_w=img_w, img_h=img_h, bbox=bbox, margin=margin)
    canvas = torch.tensor(IMAGENET_MEAN_UINT8, dtype=image.dtype).view(3, 1, 1).expand(3, side, side).clone()
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, img_w), min(y0 + side, img_h)
    canvas[:, sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[:, sy0:sy1, sx0:sx1]
    return tvf.resize(canvas, size=[out_size, out_size], antialias=True)


def letterbox_crop(image: torch.Tensor, bbox: Box, margin: float = 0.15, out_size: int = 224) -> torch.Tensor:
    """Placeholder: crop the box itself and pad it to a square (for comparison with square_crop).

    Raises:
        NotImplementedError: Always; not built yet.
    """
    raise NotImplementedError('letterbox crop is a stub; use crop_method = "square"')


def crop(image: torch.Tensor, bbox: Box, method: str, margin: float, out_size: int) -> torch.Tensor:
    """Dispatch to the configured crop method.

    Args:
        image: uint8 tensor (3, H, W).
        bbox: Normalized [x, y, w, h] box.
        method: One of CROP_METHODS.
        margin: Context margin, see square_crop.
        out_size: Output side in pixels.

    Returns:
        uint8 tensor (3, out_size, out_size).

    Raises:
        ValueError: If method is unknown.
    """
    if method == 'square':
        return square_crop(image=image, bbox=bbox, margin=margin, out_size=out_size)
    if method == 'letterbox':
        return letterbox_crop(image=image, bbox=bbox, margin=margin, out_size=out_size)
    raise ValueError(f"crop_method must be one of {CROP_METHODS}, got '{method}'")


def jitter_bbox(bbox: Box, shift: float, scale: float, rng: random.Random) -> Box:
    """Randomly shift and rescale a box (training augmentation on ground-truth boxes).

    Imitates the slightly-off boxes a detector produces on unseen images.

    Args:
        bbox: Normalized [x, y, w, h] box.
        shift: Max center shift as a fraction of box width/height (e.g. 0.10 = +/-10%).
        scale: Max size change as a fraction (e.g. 0.15 = 85%-115%).
        rng: Random generator (pass a seeded one for reproducibility).

    Returns:
        Jittered normalized box, clipped to the image.
    """
    x, y, w, h = bbox
    cx = x + w / 2 + rng.uniform(-shift, shift) * w
    cy = y + h / 2 + rng.uniform(-shift, shift) * h
    w *= rng.uniform(1 - scale, 1 + scale)
    h *= rng.uniform(1 - scale, 1 + scale)
    x0, y0 = max(cx - w / 2, 0.0), max(cy - h / 2, 0.0)
    x1, y1 = min(cx + w / 2, 1.0), min(cy + h / 2, 1.0)
    return x0, y0, x1 - x0, y1 - y0
