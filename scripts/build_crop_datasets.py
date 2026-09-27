"""Sort cropped deer images into class folders using the shared split (optional crop experiment).

Input:  a flat folder of cropped images (scripts/draw_bboxes.py, crop-images
        branch) + data/processed/split.csv (from make_split.py)
Output: data/processed/crops/<age|antler>/<train|val|test>/<CLASS>/*.jpg,
        loadable with torchvision.datasets.ImageFolder

Uses the same train/val/test assignment as everyone else, so crop results are
comparable with full-image results.

Usage:
    uv run scripts/build_crop_datasets.py --input-dir data/processed/cropped
"""

import argparse  # CLI flag for the teammate's crop output directory
import shutil  # copy files without disturbing the source crop output
from pathlib import Path  # filesystem paths

import pandas as pd  # csv + dataframe grouping

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = REPO_ROOT / "data" / "processed" / "split.csv"
CROPS_ROOT = REPO_ROOT / "data" / "processed" / "crops"

ATTRIBUTES = {
    "age": "age_status",
    "antler": "antler_status",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments: input_dir (folder of cropped images, one file per
        row in the labels csv).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Sort the flat crop folder into labeled train/val/test folders."""
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"{args.input_dir} not found")

    df = pd.read_csv(SPLIT_CSV)

    counts = {attr: {"train": 0, "val": 0, "test": 0} for attr in ATTRIBUTES}
    missing = 0

    for row in df.itertuples():
        src_path = args.input_dir / row.filename
        if not src_path.exists():
            missing += 1
            continue

        for attr_name, label_col in ATTRIBUTES.items():
            label = getattr(row, label_col)
            out_dir = CROPS_ROOT / attr_name / row.split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, out_dir / row.filename)
            counts[attr_name][row.split] += 1

    if missing:
        print(f"warning: {missing} filenames from the csv were not found in {args.input_dir}")
    for attr_name, split_counts in counts.items():
        print(f"{attr_name}: train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}")


if __name__ == "__main__":
    main()
