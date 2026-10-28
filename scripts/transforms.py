"""Shared image transforms (augmentation + normalization) that any model's dataset.py can import.

Input:  an image (PIL image or uint8 tensor from torchvision.io.decode_image), 3-channel RGB.
        Night IR images are grayscale but load as 3 identical channels with PIL .convert('RGB')
        or decode_image(..., mode=ImageReadMode.RGB).
Output: a float32 tensor [3, image_size, image_size]

Every split first fits the whole image into an image_size square (no part of the frame is cut
off, so a deer at the edge is never lost). Train then adds random augmentation at a chosen
strength; val/test stop there and are deterministic, so every model is scored on exactly the
same pixels. Compare aug levels one at a time (one row each in logs/experiments.md); 'none' is
the baseline.

Usage (from the repo root, e.g. inside models/<model>/dataset.py):
    from scripts.transforms import build_transforms
    train_tf = build_transforms('train', image_size=224, aug_level='light')
    val_tf = build_transforms('val', image_size=224)
    val_tf = build_transforms('val', image_size=224, fit='pad')  # letterbox instead of squash
"""

import torch  # float32 dtype
from torchvision.transforms import v2  # transforms API the models already use

SPLITS = ('train', 'val', 'test')
FITS = ('squash', 'pad')

# ImageNet channel stats. Only needed when matching a pretrained network's expected input.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentation strength per level. Kept conservative because the labels live in small details:
# spots on a fawn's coat and antler tips can be cropped away or blurred out by heavy augmentation.
#   crop_scale: fraction of the image area RandomResizedCrop keeps (min, max)
#   rotation:   max degrees of tilt either way (trail cams sit slightly crooked, not sideways)
#   jitter:     max brightness/contrast change (+/- fraction)
AUG_LEVELS = {
    'none': None,  # train looks exactly like val/test: the baseline to compare against
    'light': {'crop_scale': (0.85, 1.0), 'rotation': 5, 'jitter': 0.1},
    'default': {'crop_scale': (0.7, 1.0), 'rotation': 10, 'jitter': 0.2},
    'strong': {'crop_scale': (0.5, 1.0), 'rotation': 15, 'jitter': 0.3},
}


def fit_to_square(image_size: int, fit: str = 'squash') -> list[v2.Transform]:
    """Transforms that fit the whole image into an image_size x image_size square.

    'squash' resizes straight to the square, stretching the image a little (a 16:9 frame gets
    narrower). Matches what simple_cnn does now. 'pad' (letterbox) shrinks the longer side to
    image_size, keeping the aspect ratio, and fills the rest with black bars. No distortion, but
    fewer pixels go to the deer.

    Args:
        image_size: Output height and width in pixels.
        fit: 'squash' or 'pad'.

    Returns:
        A list of transforms, to splice into a pipeline.
    """
    if fit == 'squash':
        return [v2.Resize((image_size, image_size), antialias=True)]
    return [
        v2.Resize(None, max_size=image_size, antialias=True),  # longer side -> image_size
        v2.CenterCrop(image_size),  # crop larger than the image = pad the short side with zeros
    ]


def build_transforms(
    split: str, image_size: int, aug_level: str = 'default', pretrained: bool = False, fit: str = 'squash'
) -> v2.Compose:
    """Build the transform pipeline for one split.

    All splits:
        - Fit the whole image into an image_size square, squashed or padded (see fit_to_square).
          Nothing is cropped away, unlike Resize + CenterCrop, which cuts ~22% off each side of
          a 16:9 frame.

    Train (aug_level != 'none'), applied to the square:
        - RandomResizedCrop: zooms in on a random region keeping crop_scale of the area, then
          resizes back to image_size. Teaches the model that the deer can be anywhere in the
          frame. Because the input is already square, every crop shape in the default aspect
          range (3/4 to 4/3) fits, so the scale ranges in AUG_LEVELS hold as documented. The
          minimum scale stays >= 0.5 so the deer (and its antlers) usually survive the crop.
        - RandomHorizontalFlip: a deer facing left is the same deer facing right. No vertical
          flip, since an upside-down deer never shows up on a trail cam.
        - RandomRotation: small tilt only. Corners fill with black.
        - ColorJitter (brightness + contrast only): simulates dusk, flash, and exposure changes.
          Saturation and hue are left out: they change nothing on grayscale night IR images and
          would give day images colors a real camera never produces. Brightness and contrast
          scale all channels equally, so gray night images stay gray.
    Val/test (and train with aug_level='none'):
        - The square fit only. No randomness, so scores are reproducible and comparable across
          runs and models.

    Every split ends with ToDtype (scale pixels to 0-1 floats), plus Normalize with ImageNet
    mean/std when pretrained=True. The team trains from scratch, so this defaults to False.

    Args:
        split: 'train', 'val', or 'test'.
        image_size: Output height and width in pixels.
        aug_level: Train augmentation strength: 'none', 'light', 'default', or 'strong'.
            Ignored for val/test.
        pretrained: Normalize with ImageNet stats, to match a network pretrained on ImageNet.
        fit: How the full frame becomes a square: 'squash' (stretch) or 'pad' (letterbox).
            Use the same fit for training and evaluating a model.

    Returns:
        A v2.Compose that maps one image to a float32 tensor [3, image_size, image_size].

    Raises:
        ValueError: If split, aug_level, or fit isn't one of the allowed values.
    """
    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}'. Expected one of: {', '.join(SPLITS)}.")
    if aug_level not in AUG_LEVELS:
        raise ValueError(f"Invalid aug_level '{aug_level}'. Expected one of: {', '.join(AUG_LEVELS)}.")
    if fit not in FITS:
        raise ValueError(f"Invalid fit '{fit}'. Expected one of: {', '.join(FITS)}.")

    aug = AUG_LEVELS[aug_level] if split == 'train' else None

    steps: list[v2.Transform] = [v2.ToImage()]  # PIL or tensor -> tensor image (uint8 stays uint8)
    steps += fit_to_square(image_size, fit)
    if aug is not None:
        steps += [
            v2.RandomResizedCrop(image_size, scale=aug['crop_scale'], antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(aug['rotation']),
            v2.ColorJitter(brightness=aug['jitter'], contrast=aug['jitter']),
        ]
    steps.append(v2.ToDtype(torch.float32, scale=True))  # 0-255 -> 0-1 (replaces deprecated ToTensor)
    if pretrained:
        steps.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return v2.Compose(steps)
