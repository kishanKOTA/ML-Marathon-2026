"""Save image grids that show what build_transforms does to real trail-cam frames.

Input:  data/raw/images/ (from fetch_data.py) + data/processed/split.csv (from make_split.py)
Output: logs/runs/aug_preview/<aug_level>_<fit>_<image_size>.png, one per aug level and fit

Picks 8 images (4 day, 4 night IR; at least 2 whose bbox touches the frame edge, plus fawns and
antlered bucks so spots and antler tips can be checked). Each grid row is one image: the original
frame letterboxed with its bbox in red, then 6 train-transform samples, then the val/test output.
For night images it also prints how far the train samples drift from gray (max channel difference;
0-1 scale, so ~0 means they stay gray).

Usage:
    uv run scripts/preview_augmentations.py
    uv run scripts/preview_augmentations.py --image-size 320 --seed 1
"""

import argparse  # --image-size and --seed flags
import random  # seeded image selection
import sys  # put the repo root on the import path
from pathlib import Path  # filesystem paths

import pandas as pd  # read the split csv
import torch  # seeding + tensor math
from PIL import Image, ImageDraw  # load images, draw grids and labels
from torchvision.transforms import v2  # tensor -> PIL for the grid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.transforms import AUG_LEVELS, FITS, build_transforms  # noqa: E402

IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images"
SPLIT_CSV = REPO_ROOT / "data" / "processed" / "split.csv"
OUT_DIR = REPO_ROOT / "logs" / "runs" / "aug_preview"

# Truncated in the Kaggle download: bottom half is gray and PIL raises OSError on load
CORRUPT_FILES = {'SSWI000000025575326A.jpg'}
N_TRAIN_SAMPLES = 6
EDGE_TOL = 0.005  # bbox within 0.5% of the border counts as touching it
GRAY_TOL = 3.0  # mean |R-G|, |G-B| below this (0-255) = night IR frame
HEADER_HEIGHT = 20
LABEL_HEIGHT = 14
# Per light condition: edge deer, fawn (spots), antlered buck (tips), anything else
SLOTS = ('edge', 'young', 'antlered', 'any')


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments: image_size and seed.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image-size', type=int, default=224, help='transform output size (default: 224)')
    parser.add_argument('--seed', type=int, default=0, help='seed for image choice and augmentation (default: 0)')
    return parser.parse_args()


def touches_edge(row) -> bool:
    """Whether a normalized bbox reaches any border of the frame."""
    return (
        row.bbox_origin_x <= EDGE_TOL
        or row.bbox_origin_y <= EDGE_TOL
        or row.bbox_origin_x + row.bbox_width >= 1 - EDGE_TOL
        or row.bbox_origin_y + row.bbox_height >= 1 - EDGE_TOL
    )


def is_night(img: Image.Image) -> bool:
    """Whether an RGB image is a grayscale night IR frame (all channels about equal)."""
    t = v2.functional.pil_to_tensor(img.resize((64, 64))).float()
    return ((t[0] - t[1]).abs().mean() + (t[1] - t[2]).abs().mean()).item() / 2 < GRAY_TOL


def pick_images(df: pd.DataFrame, seed: int) -> list[tuple[str, object]]:
    """Pick one image per (light, slot) pair: 4 day + 4 night, 2 of them with edge bboxes.

    Args:
        df: split.csv rows.
        seed: Shuffle seed, so the same images come back each run.

    Returns:
        (light, row) pairs, day first, in SLOTS order.
    """
    rows = [r for r in df.itertuples() if r.filename not in CORRUPT_FILES]
    random.Random(seed).shuffle(rows)
    wants = {
        'edge': touches_edge,
        'young': lambda r: r.age_status == 'YOUNG',
        'antlered': lambda r: r.antler_status == 'ANTLERED',
        'any': lambda r: True,
    }
    picked: dict[tuple[str, str], object] = {}
    for row in rows:
        open_slots = [s for s in SLOTS if wants[s](row)]
        if not open_slots or not (IMAGES_DIR / row.filename).exists():
            continue
        with Image.open(IMAGES_DIR / row.filename) as img:
            light = 'night' if is_night(img.convert('RGB')) else 'day'
        slot = next((s for s in open_slots if (light, s) not in picked), None)
        if slot is not None:
            picked[(light, slot)] = row
        if len(picked) == 2 * len(SLOTS):
            break
    return [(light, picked[(light, s)]) for light in ('day', 'night') for s in SLOTS if (light, s) in picked]


def original_tile(img: Image.Image, row, size: int) -> Image.Image:
    """Letterbox the raw frame into a size x size tile and outline its bbox in red."""
    scale = size / max(img.size)
    w, h = round(img.width * scale), round(img.height * scale)
    tile = Image.new('RGB', (size, size))
    ox, oy = (size - w) // 2, (size - h) // 2
    tile.paste(img.resize((w, h), Image.Resampling.BILINEAR), (ox, oy))
    x0, y0 = ox + row.bbox_origin_x * w, oy + row.bbox_origin_y * h
    x1, y1 = x0 + row.bbox_width * w, y0 + row.bbox_height * h
    ImageDraw.Draw(tile).rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 0, 0), width=2)
    return tile


def make_grid(picks, aug_level: str, fit: str, size: int, seed: int) -> tuple[Image.Image, list[float]]:
    """Build one grid and measure how gray the night train samples stay.

    Args:
        picks: (light, row) pairs from pick_images.
        aug_level: Key of AUG_LEVELS.
        fit: 'squash' or 'pad'.
        size: Transform output size.
        seed: Base seed for the train samples (same per image across grids).

    Returns:
        The grid image, and the max channel difference (0-1) of each night image's train samples.
    """
    train_tf = build_transforms('train', size, aug_level=aug_level, fit=fit)
    val_tf = build_transforms('val', size, fit=fit)
    headers = ['original + bbox'] + [f'train {i + 1}' for i in range(N_TRAIN_SAMPLES)] + ['val/test']
    row_h = size + LABEL_HEIGHT
    grid = Image.new('RGB', (len(headers) * size, HEADER_HEIGHT + len(picks) * row_h), 'white')
    draw = ImageDraw.Draw(grid)
    for c, text in enumerate(headers):
        draw.text((c * size + 4, 4), text, fill='black')

    night_drift = []
    for r, (light, row) in enumerate(picks):
        with Image.open(IMAGES_DIR / row.filename) as raw:
            img = raw.convert('RGB')
        torch.manual_seed(seed + r)
        samples = [train_tf(img) for _ in range(N_TRAIN_SAMPLES)]
        if light == 'night':
            night_drift.append(max((s.max(0).values - s.min(0).values).max().item() for s in samples))
        tiles = [original_tile(img, row, size)] + [v2.functional.to_pil_image(t) for t in samples + [val_tf(img)]]
        y = HEADER_HEIGHT + r * row_h
        for c, tile in enumerate(tiles):
            grid.paste(tile, (c * size, y))
        label = f'{light} | {row.age_status} {row.antler_status} | edge={touches_edge(row)} | {row.filename}'
        draw.text((4, y + size + 1), label, fill='black')
    return grid, night_drift


def main() -> None:
    """Save a preview grid for every aug level and fit."""
    args = parse_args()
    picks = pick_images(pd.read_csv(SPLIT_CSV), args.seed)
    n_night = sum(light == 'night' for light, _ in picks)
    n_edge = sum(touches_edge(row) for _, row in picks)
    print(f"picked {len(picks)} images: {len(picks) - n_night} day, {n_night} night, {n_edge} with edge bboxes")
    for light, row in picks:
        print(f"  {light:5} {row.filename} {row.age_status} {row.antler_status} edge={touches_edge(row)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for aug_level in AUG_LEVELS:
        for fit in FITS:
            grid, night_drift = make_grid(picks, aug_level, fit, args.image_size, args.seed)
            out_path = OUT_DIR / f"{aug_level}_{fit}_{args.image_size}.png"
            grid.save(out_path)
            drift = max(night_drift, default=0.0)
            print(f"{out_path.relative_to(REPO_ROOT)}  night max channel diff: {drift:.4f}")


if __name__ == '__main__':
    main()
