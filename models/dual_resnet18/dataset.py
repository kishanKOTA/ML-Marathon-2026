"""(image crop, month) pairs with both labels, for the dual-head model.

Input:  config (configs/*.toml), the shared split CSV (data/processed/split_tagged.csv, one row per
        image with normalized bbox, labels, month, camera, and tags), and a split name
Output: one dict per image:
            image         float32 (3, S, S)  square crop around the GT box, ImageNet-normalized
            month         float32 (2,)       [sin, cos] of the capture month, or [0, 0] if unknown/disabled
            age_label     int64 ()           0 adult, 1 young
            antler_label  int64 ()           0 antlerless, 1 antlered
            meta          dict               filename, crop_source, camera, day_night, month,
                                             bbox_touches_edge (proxy for "head may be cut off")

Usage (from the repo root):
    uv run python -m models.dual_resnet18.dataset            # sanity check on configs/default.toml
"""

import math  # sin/cos month encoding
import random  # bbox jitter
import tomllib  # read configs
from datetime import datetime  # parse EXIF timestamps
from pathlib import Path  # filesystem paths

import pandas as pd  # read split csv
import torch  # tensors
from PIL import Image  # EXIF month for images without a CSV row
from torch.utils.data import Dataset  # base class
from torchvision.io import ImageReadMode, decode_image  # fast JPEG -> uint8 tensor
from torchvision.transforms import v2  # augmentation + normalization

from models.dual_resnet18.constants import AGE_MAP, ANTLER_MAP  # label conventions
from models.dual_resnet18.cropping import crop, jitter_bbox  # shared square crop
from scripts.transforms import IMAGENET_MEAN, IMAGENET_STD  # team's normalization constants

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
EXIF_IFD = 0x8769
EXIF_DATETIME_ORIGINAL = 36867
SPLITS = ('train', 'val', 'test')


def load_config(name: str = "default.toml") -> dict:
    """Load a config from models/dual_resnet18/configs/ and name the run after the file.

    The run name is the config's file stem (default.toml -> 'default'), so every output can be traced
    back to the config that produced it, and copying a config can't silently overwrite another run.

    Args:
        name: Config file name.

    Returns:
        Parsed TOML as a dict, plus 'run_name' (file stem) and 'run_dir' (absolute runs folder).
    """
    with open(CONFIG_DIR / name, 'rb') as f:
        config = tomllib.load(f)
    config['run_name'] = Path(name).stem
    config['run_dir'] = ROOT / config['paths']['runs']
    return config


def checkpoint_path(config: dict) -> Path:
    """Where train.py saves (and inference.py loads) the best model: <run_dir>/<run_name>_best.pt.

    Args:
        config: Config from load_config.

    Returns:
        Absolute checkpoint path.
    """
    return config['run_dir'] / f"{config['run_name']}_best.pt"


def encode_month(month: int | None) -> torch.Tensor:
    """Encode a month as a point on the unit circle so December sits next to January.

    Args:
        month: 1-12, or None if unknown.

    Returns:
        float32 tensor [sin(2*pi*m/12), cos(2*pi*m/12)]; [0, 0] (the circle's center, matching no
        real month) when month is None.
    """
    if month is None:
        return torch.zeros(2, dtype=torch.float32)
    angle = 2 * math.pi * month / 12
    return torch.tensor([math.sin(angle), math.cos(angle)], dtype=torch.float32)


def month_from_exif(path: Path) -> int | None:
    """Read the capture month from EXIF DateTimeOriginal (used for images without a CSV row).

    Args:
        path: Path to a JPEG.

    Returns:
        Month 1-12, or None if the tag is missing.
    """
    with Image.open(path) as img:
        stamp = img.getexif().get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL)
    if not stamp:
        return None
    return datetime.strptime(stamp.strip('\x00'), '%Y:%m:%d %H:%M:%S').month


def build_crop_transform(train: bool, aug: dict) -> v2.Compose:
    """Augmentation (train only) + normalization, applied after cropping.

    No vertical flip (a deer is never upside down on a trail cam) and no hue/saturation jitter
    (night IR images are gray, and spot color matters). RandomGrayscale turns some day crops gray so
    the model learns spots/antlers from shape and texture, as it must on ~half the images (night IR).

    Args:
        train: Whether to add random augmentation.
        aug: The config's [augment] table.

    Returns:
        Transform from a uint8 (3, S, S) crop to a normalized float32 (3, S, S) tensor.
    """
    steps: list[v2.Transform] = []
    if train:
        steps += [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=0, translate=(aug['translate'], aug['translate']), scale=tuple(aug['scale'])),
            v2.ColorJitter(brightness=aug['jitter'], contrast=aug['jitter']),
            v2.RandomGrayscale(p=aug['grayscale_p']),
        ]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return v2.Compose(steps)


class DualDeerDataset(Dataset):
    """One split of the shared split CSV, cropped around each GT box, with month and both labels."""

    def __init__(self, split: str, config: dict, train: bool | None = None) -> None:
        """Load one split.

        Args:
            split: 'train', 'val', or 'test'.
            config: Parsed config (see load_config).
            train: Apply augmentation + bbox jitter. Defaults to split == 'train'.

        Raises:
            ValueError: If split is unknown or a label isn't in the mapping.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
        df = pd.read_csv(ROOT / config['paths']['split'])
        df = df[df['split'] == split].reset_index(drop=True)
        df['age_label'] = df['age_status'].map(AGE_MAP)
        if df['age_label'].isna().any():
            raise ValueError(f"unmapped age labels: {df.loc[df['age_label'].isna(), 'age_status'].unique()}")
        df['antler_label'] = df['antler_status'].map(ANTLER_MAP)
        if df['antler_label'].isna().any():
            raise ValueError(f"unmapped antler labels: {df.loc[df['antler_label'].isna(), 'antler_status'].unique()}")

        self.df = df
        self.data_cfg = config['data']
        self.n_antler = config['model']['n_antler']
        self.img_dir = ROOT / config['paths']['images']
        self.train = split == 'train' if train is None else train
        self.transform = build_crop_transform(train=self.train, aug=config['augment'])
        self.rng = random.Random(config['data']['seed'])

    def __len__(self) -> int:
        """Number of images in the split."""
        return len(self.df)

    def class_counts(self) -> dict[str, list[int]]:
        """Label counts per head, e.g. for inverse-frequency class weights.

        Returns:
            {'age': [n_adult, n_young], 'antler': [n_antlerless, n_antlered, ...]} up to n_antler.
        """
        return {
            'age': self.df['age_label'].value_counts().reindex(range(2), fill_value=0).tolist(),
            'antler': self.df['antler_label'].value_counts().reindex(range(self.n_antler), fill_value=0).tolist(),
        }

    def __getitem__(self, idx: int) -> dict:
        """Return the crop, month encoding, labels, and metadata for one image."""
        row = self.df.iloc[idx]
        image = decode_image(str(self.img_dir / row['filename']), mode=ImageReadMode.RGB)
        bbox = (row['bbox_origin_x'], row['bbox_origin_y'], row['bbox_width'], row['bbox_height'])
        if self.train and self.data_cfg['gt_jitter']:
            bbox = jitter_bbox(
                bbox=bbox, shift=self.data_cfg['jitter_shift'], scale=self.data_cfg['jitter_scale'], rng=self.rng
            )
        crop_img = crop(
            image=image,
            bbox=bbox,
            method=self.data_cfg['crop_method'],
            margin=self.data_cfg['margin'],
            out_size=self.data_cfg['image_size'],
        )
        month = int(row['month']) if pd.notna(row['month']) else None
        return {
            'image': self.transform(crop_img),
            'month': encode_month(month=month if self.data_cfg['use_month'] else None),
            'age_label': torch.tensor(int(row['age_label']), dtype=torch.int64),
            'antler_label': torch.tensor(int(row['antler_label']), dtype=torch.int64),
            'meta': {
                'filename': row['filename'],
                'crop_source': 'gt',
                'camera': int(row['camera_location_seq_no']),
                'day_night': row['day_night'],
                'month': month if month is not None else 0,
                'bbox_touches_edge': bool(row['bbox_touches_edge']),
            },
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader  # batching for the sanity check

    cfg = load_config()
    ds = DualDeerDataset(split='train', config=cfg)
    print('train size:', len(ds), '| missing months:', int(ds.df['month'].isna().sum()))
    print('class counts:', ds.class_counts())
    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    print('image', tuple(batch['image'].shape), batch['image'].dtype)  # (8, 3, 224, 224) float32
    print('month', tuple(batch['month'].shape), batch['month'].dtype)  # (8, 2) float32
    print('age', batch['age_label'].tolist(), '| antler', batch['antler_label'].tolist())
