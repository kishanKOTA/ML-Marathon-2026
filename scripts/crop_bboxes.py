"""Crop each deer by its bounding box and sort the crops into class folders (optional crop experiment).

Input:  data/raw/images/ (from fetch_data.py) + data/processed/split.csv (from
        make_split.py; carries the bbox columns, labels, and split)
Output: data/processed/crops/<age|antler>/<train|val|test>/<CLASS>/*.jpg,
        loadable with torchvision.datasets.ImageFolder

Uses the same train/val/test assignment as everyone else, so crop results are
comparable with full-image results. Replaces crop_images.py (crop-images branch)
+ build_crop_datasets.py: cropping happens in memory and each crop is written
straight to its class folders.

Usage:
    uv run scripts/crop_bboxes.py
    uv run scripts/crop_bboxes.py --pad 40
"""

import argparse  # --pad flag
from pathlib import Path  # filesystem paths

import pandas as pd  # read the split csv
from PIL import Image  # open, crop, and save images (installed with torchvision)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images"
SPLIT_CSV = REPO_ROOT / "data" / "processed" / "split.csv"
CROPS_ROOT = REPO_ROOT / "data" / "processed" / "crops"

ATTRIBUTES = {
    'age': 'age_status',
    'antler': 'antler_status',
}
JPEG_QUALITY = 95
# Truncated in the Kaggle download: bottom half is gray and PIL raises OSError on load
CORRUPT_FILES = {'SSWI000000025575326A.jpg'}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments: pad (pixels of context added on each side of the box).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pad', type=int, default=20, help='padding in pixels on each side (default: 20)')
    return parser.parse_args()


def bbox_to_pixels(
    x: float, y: float, w: float, h: float, width: int, height: int, pad: int
) -> tuple[int, int, int, int]:
    """Convert a normalized (origin, size) bbox to a padded pixel box clamped to the image.

    Args:
        x: Left edge, as a fraction of image width.
        y: Top edge, as a fraction of image height.
        w: Box width, as a fraction of image width.
        h: Box height, as a fraction of image height.
        width: Image width in pixels.
        height: Image height in pixels.
        pad: Pixels added on each side before clamping.

    Returns:
        (left, top, right, bottom) in pixels, as expected by PIL's Image.crop.
    """
    x0 = max(0, int(x * width) - pad)
    y0 = max(0, int(y * height) - pad)
    x1 = min(width, int((x + w) * width) + pad)
    y1 = min(height, int((y + h) * height) + pad)
    # Guarantee at least 1px so a degenerate box still yields a valid crop
    x1 = max(x1, min(width, x0 + 1))
    y1 = max(y1, min(height, y0 + 1))
    return x0, y0, x1, y1


def main() -> None:
    """Crop every image in split.csv and save it under each attribute's class folder."""
    args = parse_args()
    df = pd.read_csv(SPLIT_CSV)

    counts = {attr: {'train': 0, 'val': 0, 'test': 0} for attr in ATTRIBUTES}
    missing = 0
    skipped = 0

    for row in df.itertuples():
        if row.filename in CORRUPT_FILES:
            skipped += 1
            continue
        src_path = IMAGES_DIR / row.filename
        if not src_path.exists():
            missing += 1
            continue

        with Image.open(src_path) as img:
            box = bbox_to_pixels(
                x=row.bbox_origin_x,
                y=row.bbox_origin_y,
                w=row.bbox_width,
                h=row.bbox_height,
                width=img.width,
                height=img.height,
                pad=args.pad,
            )
            crop = img.crop(box)

        for attr_name, label_col in ATTRIBUTES.items():
            out_dir = CROPS_ROOT / attr_name / row.split / getattr(row, label_col)
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / row.filename, quality=JPEG_QUALITY)
            counts[attr_name][row.split] += 1

    if skipped:
        print(f"skipped {skipped} known-corrupt image(s): {sorted(CORRUPT_FILES)}")
    if missing:
        print(f"warning: {missing} filenames from the csv were not found in {IMAGES_DIR}")
    for attr_name, split_counts in counts.items():
        print(f"{attr_name}: train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}")


if __name__ == '__main__':
    main()
