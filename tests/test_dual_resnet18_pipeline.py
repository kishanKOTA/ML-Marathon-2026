"""Tests for the dual_resnet18 plumbing: cropping, month encoding, the inference constraint, data loading.

These don't need a working model. Dataset tests are skipped when the data isn't downloaded.

Run: uv run pytest tests/test_dual_resnet18_pipeline.py
"""

import math  # expected sin/cos values
import random  # seeded jitter
from pathlib import Path  # tmp_path type hint

import numpy as np  # calibration toy data
import pandas as pd  # empty labels table
import pytest  # parametrize, skip
import torch  # tensors
from PIL import Image  # synthetic image without EXIF

from models.dual_resnet18.constants import ANTLERED, ANTLERLESS, YOUNG  # label conventions
from models.dual_resnet18.cropping import IMAGENET_MEAN_UINT8, jitter_bbox, square_crop, square_window  # under test
from models.dual_resnet18.dataset import ROOT, DualDeerDataset, encode_month, load_config  # under test
from models.dual_resnet18.evaluate import expected_calibration_error  # under test
from models.dual_resnet18.inference import constrained_prediction, jobs_from_folder  # under test

W, H = 1600, 1200

CROP_CASES = {
    'centered': (0.4, 0.4, 0.2, 0.2),
    'left_edge': (0.0, 0.4, 0.2, 0.2),
    'right_edge': (0.8, 0.4, 0.2, 0.2),
    'top_edge': (0.4, 0.0, 0.2, 0.2),
    'bottom_edge': (0.4, 0.8, 0.2, 0.2),
    'corner': (0.85, 0.85, 0.15, 0.15),
    'taller_than_image': (0.3, 0.0, 0.3, 1.0),
    'wider_than_tall_image': (0.0, 0.2, 1.0, 0.5),
    'tiny': (0.5, 0.5, 0.003, 0.004),
}


@pytest.mark.parametrize('bbox', CROP_CASES.values(), ids=CROP_CASES.keys())
def test_square_window_contains_box_and_stays_inside_when_it_fits(bbox: tuple) -> None:
    """Full box lies inside the crop; if the side fits the image, the crop is fully inside the image."""
    x0, y0, side = square_window(img_w=W, img_h=H, bbox=bbox, margin=0.15)
    bx0, by0 = math.floor(bbox[0] * W), math.floor(bbox[1] * H)
    bx1, by1 = math.ceil((bbox[0] + bbox[2]) * W), math.ceil((bbox[1] + bbox[3]) * H)
    assert x0 <= bx0 and x0 + side >= bx1
    assert y0 <= by0 and y0 + side >= by1
    if side <= W:
        assert 0 <= x0 and x0 + side <= W
    if side <= H:
        assert 0 <= y0 and y0 + side <= H


@pytest.mark.parametrize('bbox', CROP_CASES.values(), ids=CROP_CASES.keys())
def test_square_crop_output_shape(bbox: tuple) -> None:
    """Every case resizes to 224x224 uint8, including the upsampled tiny box."""
    image = torch.randint(low=0, high=256, size=(3, H, W), dtype=torch.uint8)
    out = square_crop(image=image, bbox=bbox, margin=0.15, out_size=224)
    assert out.shape == (3, 224, 224) and out.dtype == torch.uint8


def test_edge_box_is_shifted_not_padded() -> None:
    """A box touching the left edge shifts the window to x0 = 0 instead of padding."""
    x0, _, _ = square_window(img_w=W, img_h=H, bbox=CROP_CASES['left_edge'], margin=0.15)
    assert x0 == 0


def test_oversized_side_pads_with_imagenet_mean() -> None:
    """A full-height box needs a side > image height: the overflow rows are the ImageNet mean color."""
    image = torch.zeros(3, H, W, dtype=torch.uint8)
    out = square_crop(image=image, bbox=CROP_CASES['taller_than_image'], margin=0.15, out_size=224)
    assert out[:, 0, 112].tolist() == list(IMAGENET_MEAN_UINT8)
    assert out[:, 112, 112].tolist() == [0, 0, 0]


def test_jitter_bbox_stays_in_image() -> None:
    """Jittered boxes are clipped to [0, 1] and keep positive size."""
    rng = random.Random(0)
    for _ in range(500):
        x, y, w, h = jitter_bbox(bbox=(0.9, 0.0, 0.1, 0.5), shift=0.10, scale=0.15, rng=rng)
        assert 0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9


def test_encode_month_wraps_and_handles_missing() -> None:
    """December is as close to January as February is; unknown month is the circle's center."""
    dec, jan, feb = (encode_month(month=m) for m in (12, 1, 2))
    assert torch.isclose((dec - jan).norm(), (feb - jan).norm())
    assert torch.isclose(jan.norm(), torch.tensor(1.0))
    assert encode_month(month=None).tolist() == [0.0, 0.0]


def test_constrained_prediction_worked_example() -> None:
    """Handoff worked example, in team label order (antler: [antlerless, antlered])."""
    out = constrained_prediction(
        p_age=torch.tensor([0.85, 0.15]),
        p_ant=torch.tensor([0.30, 0.70]),
        antler_conf_threshold=0.6,
        review_threshold=0.5,
    )
    assert out['label'] == 'adult_antlered'
    assert out['prob'] == pytest.approx(0.665, abs=1e-3)
    assert out['p_adult_antlerless'] == pytest.approx(0.285, abs=1e-3)
    assert out['p_young_antlerless'] == pytest.approx(0.050, abs=1e-3)
    assert out['removed_mass'] == pytest.approx(0.105, abs=1e-6)
    assert out['needs_review'] is False
    assert out['antler_status'] == 'determined'


def test_strong_disagreement_flags_review() -> None:
    """Both heads confident in the impossible combination: flagged, and still never predicted."""
    out = constrained_prediction(
        p_age=torch.tensor([0.2, 0.8]),
        p_ant=torch.tensor([0.1, 0.9]),
        antler_conf_threshold=0.6,
        review_threshold=0.5,
    )
    assert out['removed_mass'] == pytest.approx(0.72)
    assert out['needs_review'] is True
    assert out['label'] == 'adult_antlered'
    assert out['prob'] == pytest.approx(0.18 / 0.28, abs=1e-6)


@pytest.mark.parametrize(
    ('p_age', 'label', 'antler_pred'),
    [([0.3, 0.7], 'young_antlerless', ANTLERLESS), ([0.7, 0.3], 'adult_undetermined', ANTLERED)],
)
def test_unsure_antler_head_falls_back_to_age(p_age: list, label: str, antler_pred: int) -> None:
    """Antler confidence 0.55 < 0.6: young implies antlerless; adult is undetermined (antler_pred = argmax)."""
    out = constrained_prediction(
        p_age=torch.tensor(p_age), p_ant=torch.tensor([0.45, 0.55]), antler_conf_threshold=0.6, review_threshold=0.5
    )
    assert out['label'] == label and out['antler_status'] == 'undetermined' and out['antler_pred'] == antler_pred


def test_impossible_combination_never_predicted() -> None:
    """Across random head outputs, the final prediction is never antlered + young."""
    torch.manual_seed(0)
    for _ in range(2000):
        out = constrained_prediction(
            p_age=torch.rand(2).softmax(dim=0) if torch.rand(1) < 0.5 else torch.randn(2).mul(3).softmax(dim=0),
            p_ant=torch.randn(2).mul(3).softmax(dim=0),
            antler_conf_threshold=0.6,
            review_threshold=0.5,
        )
        assert not (out['age_pred'] == YOUNG and out['antler_pred'] == ANTLERED)
        valid = out['p_adult_antlerless'] + out['p_adult_antlered'] + out['p_young_antlerless']
        assert valid == pytest.approx(1.0, abs=1e-5)


def test_expected_calibration_error_on_calibrated_toy() -> None:
    """Confidence 0.8 with exactly 80% correct is perfectly calibrated (ECE 0); always-wrong is ECE 0.8."""
    confidence = np.full(100, 0.8)
    correct = np.array([True] * 80 + [False] * 20)
    assert expected_calibration_error(confidence=confidence, correct=correct)[0] == pytest.approx(0.0)
    assert expected_calibration_error(confidence=confidence, correct=~np.ones(100, bool))[0] == pytest.approx(0.8)


def test_unseen_image_full_frame_fallback(tmp_path: Path) -> None:
    """An image not in the CSV, with no EXIF and fallback 'full_frame': one full-frame job, no month."""
    Image.new('RGB', (64, 48), color=(10, 20, 30)).save(tmp_path / 'unseen.jpg')
    empty = pd.DataFrame(columns=['filename', 'bbox_origin_x', 'bbox_origin_y', 'bbox_width', 'bbox_height', 'month'])
    jobs = jobs_from_folder(folder=tmp_path, split_df=empty, detect_cfg={'fallback': 'full_frame'})
    assert len(jobs) == 1
    assert jobs[0]['bbox'] == (0.0, 0.0, 1.0, 1.0) and jobs[0]['no_detection'] and jobs[0]['month'] is None


CONFIG = load_config()
needs_data = pytest.mark.skipif(not (ROOT / CONFIG['paths']['split']).exists(), reason='split CSV not downloaded')


@needs_data
def test_dataset_item_matches_spec() -> None:
    """Each item has the documented fields, shapes, dtypes, and label ranges."""
    ds = DualDeerDataset(split='val', config=CONFIG)
    item = ds[0]
    assert item['image'].shape == (3, 224, 224) and item['image'].dtype == torch.float32
    assert item['month'].shape == (2,) and item['month'].dtype == torch.float32
    assert torch.isclose(item['month'].norm(), torch.tensor(1.0))
    assert item['age_label'].dtype == torch.int64 and item['age_label'].item() in (0, 1)
    assert item['antler_label'].item() in (0, 1)
    assert item['meta']['crop_source'] == 'gt'


@needs_data
def test_month_ablation_feeds_zeros() -> None:
    """use_month = false gives [0, 0] for every image."""
    cfg = CONFIG | {'data': CONFIG['data'] | {'use_month': False}}
    assert DualDeerDataset(split='val', config=cfg)[0]['month'].tolist() == [0.0, 0.0]


@needs_data
def test_no_antlered_young_in_labels() -> None:
    """The hard constraint assumes antlered + young never occurs in the ground truth."""
    df = DualDeerDataset(split='train', config=CONFIG).df
    assert not ((df['age_label'] == YOUNG) & (df['antler_label'] == ANTLERED)).any()
