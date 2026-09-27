"""Download the Snapshot WI - Oh Deer competition data and lay it out under data/raw/.

Input:  the Kaggle competition (needs your Kaggle credentials)
Output: data/raw/ with the two CSVs and a flat images/ folder. Next step: make_split.py.

Resulting layout:
    data/raw/Snapshot_WI-Oh_Deer_data-v2.csv
    data/raw/Snapshot_WI-Oh_Deer_locs-v2.csv
    data/raw/images/*.jpg   (flat, 4880 files)

kagglehub downloads into its own cache (~/.cache/kagglehub/) and this script
copies from there, so rerunning is cheap. Requires Kaggle credentials
(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY) and having accepted the
competition rules on the Kaggle website.

Usage:
    uv run scripts/fetch_data.py            # skip if already set up
    uv run scripts/fetch_data.py --force    # re-copy everything
"""

import argparse  # CLI flags (--force, --raw-dir)
import shutil  # copy files out of the kagglehub cache
from pathlib import Path  # filesystem paths

import kagglehub  # Kaggle competition download + local caching

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"

COMPETITION = 'snapshot-wi-oh-deer'
CSV_NAMES = ('Snapshot_WI-Oh_Deer_data-v2.csv', 'Snapshot_WI-Oh_Deer_locs-v2.csv')
PHOTOS_DIR_NAME = 'Snapshot_WI-Oh_Deer_Photos'
EXPECTED_IMAGES = 4880


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments: force (re-copy even if set up) and raw_dir (target folder).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true', help='re-copy even if data/raw/ looks complete')
    parser.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW_DIR, help='target folder (default: data/raw/)')
    return parser.parse_args()


def is_set_up(raw_dir: Path) -> bool:
    """Check whether raw_dir already has both CSVs and all images.

    Args:
        raw_dir: Target raw-data folder.

    Returns:
        True if nothing needs to be copied.
    """
    csvs_present = all((raw_dir / name).exists() for name in CSV_NAMES)
    n_images = len(list((raw_dir / 'images').glob('*.jpg')))
    return csvs_present and n_images == EXPECTED_IMAGES


def organize_raw(src_dir: Path, raw_dir: Path) -> None:
    """Copy the CSVs and a flattened image folder from a downloaded bundle into raw_dir.

    Args:
        src_dir: Root of the downloaded competition bundle (kagglehub cache path).
        raw_dir: Target raw-data folder.

    Raises:
        FileNotFoundError: If a CSV or the photos folder is missing from the bundle.
    """
    images_dir = raw_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    for name in CSV_NAMES:
        matches = list(src_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found under {src_dir}")
        shutil.copy2(matches[0], raw_dir / name)

    photo_dirs = [p for p in src_dir.rglob(PHOTOS_DIR_NAME) if p.is_dir()]
    if not photo_dirs:
        raise FileNotFoundError(f"{PHOTOS_DIR_NAME}/ not found under {src_dir}")

    n_copied = 0
    for img_path in photo_dirs[0].rglob('*.jpg'):
        shutil.copy2(img_path, images_dir / img_path.name)
        n_copied += 1
    print(f"copied {len(CSV_NAMES)} csvs and {n_copied} images into {raw_dir}")
    if n_copied != EXPECTED_IMAGES:
        print(f"warning: expected {EXPECTED_IMAGES} images, got {n_copied}")


def main() -> None:
    """Download (or reuse the cached) competition bundle and organize it into raw_dir."""
    args = parse_args()
    if is_set_up(raw_dir=args.raw_dir) and not args.force:
        print(f"{args.raw_dir} already set up, nothing to do (use --force to re-copy)")
        return

    src_dir = Path(kagglehub.competition_download(handle=COMPETITION))
    print(f"competition files at {src_dir}")
    organize_raw(src_dir=src_dir, raw_dir=args.raw_dir)


if __name__ == '__main__':
    main()
