"""Tag each image with day/night (from color saturation) and deer visibility (from bbox extent).

Input:  data/raw/images/ + data/raw/Snapshot_WI-Oh_Deer_data-v2.csv (from fetch_data.py)
        data/processed/split.csv (from make_split.py)
Output: data/processed/tags.csv, one row per image:
        filename, mean_saturation, day_night, bbox_area, bbox_size, bbox_touches_edge
        data/processed/split_tagged.csv: split.csv with the tag columns appended (corrupt
        images are already dropped from split.csv, so they are absent here too)

Night shots use an IR flash and are saved as 3-channel JPEGs with B == G == R, so their HSV
saturation is exactly 0. On the v2 data, 2562 images have mean saturation 0 and the least
saturated color image (a snowy day) is ~9, so any threshold in between separates them.
'night' therefore means grayscale/IR, which can include dusk and dawn.

Visibility: bbox_area is the fraction of the frame covered by the box (bbox columns are
normalized to 0-1). bbox_size splits it into balanced tertiles (small/medium/large), and
bbox_touches_edge flags deer likely cut off by the frame. Raw numbers are kept next to the
labels so thresholds can change without re-reading images.

Regenerate (after make_split.py):
    uv run scripts/tag_images.py
    uv run scripts/tag_images.py --night-threshold 2
"""

import argparse  # --night-threshold flag
from pathlib import Path  # filesystem paths

import numpy as np  # saturation statistics
import pandas as pd  # read labels, write tags csv
from PIL import Image  # reduced-size JPEG loading + HSV (comes with torchvision)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images"
LABELS_CSV = REPO_ROOT / "data" / "raw" / "Snapshot_WI-Oh_Deer_data-v2.csv"
TAGS_CSV = REPO_ROOT / "data" / "processed" / "tags.csv"
SPLIT_CSV = REPO_ROOT / "data" / "processed" / "split.csv"
SPLIT_TAGGED_CSV = REPO_ROOT / "data" / "processed" / "split_tagged.csv"

# Truncated in the Kaggle download: bottom half is gray and PIL raises OSError on load
CORRUPT_FILES = {'SSWI000000025575326A.jpg'}
# Load JPEGs at 1/4 resolution: plenty for a whole-image saturation mean, and ~4x faster
DRAFT_SCALE = 4
# A box edge within 1% of the frame border counts as touching it
EDGE_MARGIN = 0.01
SIZE_LABELS = ['small', 'medium', 'large']


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments: night_threshold (mean saturation below which an image is night).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--night-threshold', type=float, default=1.0, help='mean saturation (0-255) below which = night (default: 1)'
    )
    return parser.parse_args()


def mean_saturation(path: Path) -> float:
    """Load an image at reduced size and return its mean HSV saturation.

    Args:
        path: Path to a JPEG image.

    Returns:
        Mean saturation on PIL's 0-255 scale; 0 for a grayscale (B == G == R) image.
    """
    with Image.open(path) as img:
        img.draft('RGB', (img.width // DRAFT_SCALE, img.height // DRAFT_SCALE))
        saturation = np.asarray(img.convert('HSV'))[:, :, 1]
    return float(saturation.mean())


def day_night(mean_s: float, threshold: float) -> str:
    """Label an image as day or night from its mean saturation.

    Args:
        mean_s: Mean HSV saturation (0-255), NaN if the image could not be read.
        threshold: Mean saturation below which the image is night.

    Returns:
        'night', 'day', or 'unknown' when mean_s is NaN.
    """
    if np.isnan(mean_s):
        return 'unknown'
    return 'night' if mean_s < threshold else 'day'


def bbox_tags(labels: pd.DataFrame) -> pd.DataFrame:
    """Compute bbox area, size tertile, and edge contact for each labeled box.

    Args:
        labels: Ground-truth table with normalized bbox_origin_x/y and bbox_width/height columns.

    Returns:
        DataFrame with filename, bbox_area, bbox_size, and bbox_touches_edge columns.
    """
    x, y = labels['bbox_origin_x'], labels['bbox_origin_y']
    w, h = labels['bbox_width'], labels['bbox_height']
    area = w * h
    touches_edge = (x <= EDGE_MARGIN) | (y <= EDGE_MARGIN) | (x + w >= 1 - EDGE_MARGIN) | (y + h >= 1 - EDGE_MARGIN)
    return pd.DataFrame(
        {
            'filename': labels['filename'],
            'bbox_area': area.round(4),
            'bbox_size': pd.qcut(area, q=len(SIZE_LABELS), labels=SIZE_LABELS),
            'bbox_touches_edge': touches_edge,
        }
    )


def main() -> None:
    """Tag every labeled image, write data/processed/tags.csv, and join it onto split.csv."""
    args = parse_args()
    labels = pd.read_csv(LABELS_CSV)
    # One box per image in v2; the merge below would duplicate rows if that changes
    assert labels['filename'].is_unique, 'expected one bbox row per image'

    saturations = []
    for i, filename in enumerate(labels['filename'], start=1):
        mean_s = np.nan if filename in CORRUPT_FILES else mean_saturation(path=IMAGES_DIR / filename)
        saturations.append({'filename': filename, 'mean_saturation': round(mean_s, 3)})
        if i % 500 == 0:
            print(f"{i}/{len(labels)} images read")

    tags = pd.DataFrame(saturations)
    tags['day_night'] = [day_night(mean_s=s, threshold=args.night_threshold) for s in tags['mean_saturation']]
    tags = tags.merge(bbox_tags(labels=labels), on='filename')

    TAGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tags.to_csv(TAGS_CSV, index=False)
    print(f"Wrote {len(tags)} rows to {TAGS_CSV}")
    print(tags['day_night'].value_counts().to_string())
    print(pd.crosstab(tags['bbox_size'], tags['bbox_touches_edge']).to_string())

    split = pd.read_csv(SPLIT_CSV)
    split_tagged = split.merge(tags, on='filename', how='left', validate='one_to_one')
    assert split_tagged['day_night'].notna().all(), 'some split images have no tags'
    split_tagged.to_csv(SPLIT_TAGGED_CSV, index=False)
    print(f"Wrote {len(split_tagged)} rows to {SPLIT_TAGGED_CSV}")


if __name__ == '__main__':
    main()
