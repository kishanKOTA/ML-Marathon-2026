"""Create the team's shared train/val/test split.

Input:  data/raw/Snapshot_WI-Oh_Deer_data-v2.csv (from fetch_data.py) and a
        split config (default: configs/split.toml, the team's official split)
Output: the config's output CSV (default: data/processed/split.csv), the labels
        CSV plus a 'split' column ('train' / 'val' / 'test'), one row per image.

Every member loads this same file, so everyone's F1 scores are comparable.
Deterministic: same seed + same locked sklearn version -> same split on every
machine. Compare the printed fingerprint with a teammate's to confirm.

Split is grouped by camera_location_seq_no (a camera's images all land in one
split, so backgrounds can't leak from train into val/test) and stratified on the
age+antler combination (each split keeps roughly the overall class balance).
Images in CORRUPT_IMAGES are dropped after splitting, so removing one doesn't
reshuffle any other image's split.

Usage:
    uv run scripts/make_split.py
    uv run scripts/make_split.py --config configs/split_seed7.toml
"""

import argparse  # --config flag for alternate splits
import hashlib  # short fingerprint so teammates can confirm identical splits
import tomllib  # read the split config
from pathlib import Path  # filesystem paths

import pandas as pd  # read/write the labels csv
from sklearn.model_selection import StratifiedGroupKFold  # grouped + stratified folds

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_CSV = REPO_ROOT / "data" / "raw" / "Snapshot_WI-Oh_Deer_data-v2.csv"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "split.toml"
GROUP_COL = 'camera_location_seq_no'
CORRUPT_IMAGES = {
    'SSWI000000025575326A.jpg',  # truncated JPEG; torchvision and PIL both fail to decode
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments: config (path to a split .toml).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='split config (default: configs/split.toml)')
    return parser.parse_args()


def assign_splits(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Assign each row to 'train', 'val', or 'test', grouped by camera and stratified by class.

    Splits into n_folds camera-disjoint folds: fold 0 -> test, fold 1 -> val,
    the rest -> train (~71/14/14 with 7 folds; exact sizes vary with camera sizes).

    Args:
        df: Labels dataframe with age_status, antler_status, and camera_location_seq_no.
        n_folds: Number of folds; val and test each get ~1/n_folds.
        seed: Random seed for the fold assignment.

    Returns:
        Series of split names, indexed like df.
    """
    strata = df['age_status'] + '_' + df['antler_status']
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = [held_out for _, held_out in folds.split(df, y=strata, groups=df[GROUP_COL])]

    split = pd.Series('train', index=df.index)
    split.iloc[fold_indices[0]] = 'test'
    split.iloc[fold_indices[1]] = 'val'
    return split


def check_no_camera_overlap(df: pd.DataFrame) -> None:
    """Raise if any camera appears in more than one split.

    Args:
        df: Dataframe with 'split' and camera_location_seq_no columns.

    Raises:
        ValueError: If a camera spans multiple splits.
    """
    splits_per_camera = df.groupby(GROUP_COL)['split'].nunique()
    leaked = splits_per_camera[splits_per_camera > 1]
    if not leaked.empty:
        raise ValueError(f"{len(leaked)} cameras appear in more than one split")


def fingerprint(df: pd.DataFrame) -> str:
    """Hash the filename -> split assignment into a short string.

    Args:
        df: Dataframe with 'filename' and 'split' columns.

    Returns:
        First 8 hex characters of an MD5 over the sorted assignments.
    """
    pairs = df.sort_values('filename')[['filename', 'split']].to_csv(index=False)
    return hashlib.md5(pairs.encode()).hexdigest()[:8]


def main() -> None:
    """Write the split CSV named in the config and print a per-split summary."""
    args = parse_args()
    with open(args.config, 'rb') as f:
        config = tomllib.load(f)
    split_csv = REPO_ROOT / config['output']

    df = pd.read_csv(LABELS_CSV)
    df['split'] = assign_splits(df=df, n_folds=config['n_folds'], seed=config['seed'])
    df = df[~df['filename'].isin(CORRUPT_IMAGES)]
    check_no_camera_overlap(df=df)

    split_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_csv, index=False)

    summary = df.groupby('split').agg(
        images=('filename', 'size'),
        cameras=(GROUP_COL, 'nunique'),
        young=('age_status', lambda s: (s == 'YOUNG').mean()),
        antlered=('antler_status', lambda s: (s == 'ANTLERED').mean()),
    )
    print(f"wrote {split_csv} ({args.config.name})")
    print(summary.round(3).to_string())
    print(f"fingerprint: {fingerprint(df=df)}")


if __name__ == '__main__':
    main()
