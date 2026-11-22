"""Predict age + antler per crop, with the antlered-young combination ruled out.

Input:  a trained DualResNet18 checkpoint (state_dict) + config, and either a split of the shared CSV
        (GT boxes) or a folder of images (GT box if the filename is in the CSV, else the [detect] fallback)
Output: CSV, one row per crop: filename, bbox, detection_conf, crop_source, no_detection, month,
        label, prob, the three valid joint probabilities, raw per-head probabilities, antler_status,
        removed_mass, needs_review, age_pred, antler_pred

How the constraint works: the two heads' probabilities are multiplied into a joint table
[antler, age]; the antlered + young cell is always zeroed (it is biologically impossible) and the
remaining cells renormalized, i.e. the prediction is conditioned on the combination being valid.
If the antler head is unsure, only the age result is used.

Usage (from the repo root):
    uv run python -m models.dual_resnet18.inference --split val
    uv run python -m models.dual_resnet18.inference --images path/to/folder --config default.toml
"""

import argparse  # CLI flags
import time  # inference speed
from pathlib import Path  # filesystem paths

import pandas as pd  # split csv in, predictions csv out
import torch  # tensors, no_grad
from torch.utils.data import DataLoader, Dataset  # batched crops
from torchvision.io import ImageReadMode, decode_image  # JPEG -> uint8 tensor

from models.dual_resnet18.constants import (  # label conventions
    ADULT,
    ADULT_UNDETERMINED,
    ANTLER_NAMES,
    ANTLERLESS,
    COMBINED_NAMES,
    IMPOSSIBLE_CELL,
    YOUNG,
)
from models.dual_resnet18.cropping import crop  # same crop as training
from models.dual_resnet18.dataset import (  # config, month encoding, eval transform
    ROOT,
    build_crop_transform,
    checkpoint_path,
    encode_month,
    load_config,
    month_from_exif,
)
from models.dual_resnet18.detect import load_megadetector, resolve_boxes  # box source per image
from models.dual_resnet18.model import DualResNet18  # user-implemented model

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def constrained_prediction(
    p_age: torch.Tensor, p_ant: torch.Tensor, antler_conf_threshold: float, review_threshold: float
) -> dict:
    """Combine one crop's head probabilities into a valid (age, antler) prediction.

    Args:
        p_age: (2,) softmax of the age head [P(adult), P(young)].
        p_ant: (2,) softmax of the antler head [P(antlerless), P(antlered)].
        antler_conf_threshold: Below this, max(P(antlerless), P(antlered)) means
            "antler status undetermined" and only the age result is used.
        review_threshold: Removed impossible-cell mass above this sets needs_review. It only flags;
            the cell is zeroed regardless.

    Returns:
        Dict with label, prob, joint probabilities (p_adult_antlerless, p_adult_antlered,
        p_young_antlerless), raw head probabilities, antler_status, removed_mass, needs_review,
        and the final per-head predictions age_pred / antler_pred (for F1).
    """
    joint = torch.outer(p_ant, p_age)
    removed = joint[IMPOSSIBLE_CELL].item()
    joint[IMPOSSIBLE_CELL] = 0.0
    joint = joint / joint.sum()

    out = {f'p_{name}': joint[cell].item() for cell, name in COMBINED_NAMES.items()}
    out |= {'p_age_adult': p_age[ADULT].item(), 'p_age_young': p_age[YOUNG].item()}
    out |= {f'p_{name}': p_ant[i].item() for i, name in enumerate(ANTLER_NAMES)}
    out |= {'removed_mass': removed, 'needs_review': removed > review_threshold}

    if p_ant.max().item() < antler_conf_threshold:
        age = int(p_age.argmax().item())
        antler = ANTLERLESS if age == YOUNG else int(p_ant.argmax().item())
        label = COMBINED_NAMES[(ANTLERLESS, YOUNG)] if age == YOUNG else ADULT_UNDETERMINED
        return out | {
            'label': label,
            'prob': p_age[age].item(),
            'antler_status': 'undetermined',
            'age_pred': age,
            'antler_pred': antler,
        }

    flat = int(joint.argmax().item())
    antler, age = divmod(flat, joint.shape[1])
    return out | {
        'label': COMBINED_NAMES[(antler, age)],
        'prob': joint[antler, age].item(),
        'antler_status': 'determined',
        'age_pred': age,
        'antler_pred': antler,
    }


class CropJobs(Dataset):
    """Crops to classify: one per (image, box), decoded and cropped lazily."""

    def __init__(self, jobs: list[dict], data_cfg: dict) -> None:
        """Store the jobs.

        Args:
            jobs: Dicts with path, bbox, and month (None if unknown).
            data_cfg: The config's [data] table.
        """
        self.jobs = jobs
        self.data_cfg = data_cfg
        self.transform = build_crop_transform(train=False, aug={})

    def __len__(self) -> int:
        """Number of crops."""
        return len(self.jobs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized crop and month encoding for job idx."""
        job = self.jobs[idx]
        image = decode_image(str(job['path']), mode=ImageReadMode.RGB)
        crop_img = crop(
            image=image,
            bbox=job['bbox'],
            method=self.data_cfg['crop_method'],
            margin=self.data_cfg['margin'],
            out_size=self.data_cfg['image_size'],
        )
        month = job['month'] if self.data_cfg['use_month'] else None
        return self.transform(crop_img), encode_month(month=month)


def jobs_from_split(split_df: pd.DataFrame, img_dir: Path) -> list[dict]:
    """One job per CSV row, using its GT box and month.

    Args:
        split_df: Rows of the split CSV.
        img_dir: Folder holding the images.

    Returns:
        Job dicts (see CropJobs).
    """
    return [
        {
            'filename': r.filename,
            'path': img_dir / r.filename,
            'bbox': (r.bbox_origin_x, r.bbox_origin_y, r.bbox_width, r.bbox_height),
            'conf': r.bbox_confidence,
            'source': 'gt',
            'no_detection': False,
            'month': int(r.month) if pd.notna(r.month) else None,
        }
        for r in split_df.itertuples()
    ]


def jobs_from_folder(folder: Path, split_df: pd.DataFrame, detect_cfg: dict) -> list[dict]:
    """Jobs for every image in a folder: GT box + CSV month when known, else fallback box + EXIF month.

    Args:
        folder: Folder of images.
        split_df: Full split CSV (any split), used to look up GT boxes and months by filename.
        detect_cfg: The config's [detect] table.

    Returns:
        Job dicts (see CropJobs), possibly several per image.
    """
    known = split_df.set_index('filename')
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    needs_detector = detect_cfg['fallback'] == 'megadetector' and any(p.name not in known.index for p in paths)
    detector = load_megadetector(model_name=detect_cfg['md_model']) if needs_detector else None

    jobs = []
    for path in paths:
        if path.name in known.index:
            r = known.loc[path.name]
            gt_bbox = (r['bbox_origin_x'], r['bbox_origin_y'], r['bbox_width'], r['bbox_height'])
            month = int(r['month']) if pd.notna(r['month']) else None
        else:
            gt_bbox, month = None, month_from_exif(path=path)
        for det in resolve_boxes(path=path, gt_bbox=gt_bbox, detect_cfg=detect_cfg, detector=detector):
            jobs.append({'filename': path.name, 'path': path, 'month': month} | det)
    return jobs


def pick_device() -> torch.device:
    """Apple Silicon GPU, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(config: dict, device: torch.device) -> DualResNet18:
    """Build DualResNet18 from the config and load the checkpoint's state_dict.

    Args:
        config: Parsed config.
        device: Where to put the model.

    Returns:
        Model in eval mode.
    """
    model = DualResNet18(**(config['model'] | {'pretrained': False}))
    state = torch.load(checkpoint_path(config=config), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model: DualResNet18, jobs: list[dict], config: dict, device: torch.device) -> pd.DataFrame:
    """Run the model on every job and apply the constraint.

    Args:
        model: Trained model in eval mode.
        jobs: Job dicts (see CropJobs).
        config: Parsed config.
        device: Model device.

    Returns:
        One row per job (see module docstring).
    """
    inf = config['inference']
    loader = DataLoader(CropJobs(jobs=jobs, data_cfg=config['data']), batch_size=inf['batch_size'], shuffle=False)
    rows, model_seconds, i = [], 0.0, 0
    with torch.no_grad():
        for images, months in loader:
            start = time.perf_counter()
            age_logits, antler_logits = model(images.to(device), months.to(device))
            p_age_batch, p_ant_batch = age_logits.softmax(dim=1).cpu(), antler_logits.softmax(dim=1).cpu()
            model_seconds += time.perf_counter() - start
            for p_age, p_ant in zip(p_age_batch, p_ant_batch, strict=True):
                job = jobs[i]
                x, y, w, h = job['bbox']
                pred = constrained_prediction(
                    p_age=p_age,
                    p_ant=p_ant,
                    antler_conf_threshold=inf['antler_conf_threshold'],
                    review_threshold=inf['review_threshold'],
                )
                rows.append(
                    {
                        'filename': job['filename'],
                        'bbox_x': x,
                        'bbox_y': y,
                        'bbox_w': w,
                        'bbox_h': h,
                        'detection_conf': job['conf'],
                        'crop_source': job['source'],
                        'no_detection': job['no_detection'],
                        'month': job['month'],
                    }
                    | pred
                )
                i += 1
    print(f"{len(rows)} crops, model time {1000 * model_seconds / max(len(rows), 1):.2f} ms/crop on {device}")
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default="default.toml", help='file in models/dual_resnet18/configs/')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    source.add_argument('--images', type=Path, help='folder of images to predict instead of a split')
    parser.add_argument(
        '--out', type=Path, help='output CSV (default: <run_dir>/<run_name>_<split|folder>_predictions.csv)'
    )
    args = parser.parse_args()

    config = load_config(name=args.config)
    split_df = pd.read_csv(ROOT / config['paths']['split'])
    if args.images is not None:
        jobs = jobs_from_folder(folder=args.images, split_df=split_df, detect_cfg=config['detect'])
        tag = args.images.name
    else:
        subset = split_df[split_df['split'] == args.split]
        jobs = jobs_from_split(split_df=subset, img_dir=ROOT / config['paths']['images'])
        tag = args.split

    device = pick_device()
    preds = predict(model=load_model(config=config, device=device), jobs=jobs, config=config, device=device)
    out = args.out or config['run_dir'] / f"{config['run_name']}_{tag}_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out, index=False)
    print(f"Wrote {len(preds)} rows to {out}")
    print(preds['label'].value_counts().to_string())


if __name__ == '__main__':
    main()
