"""Where each crop's box comes from: the GT box when the image has one, else a fallback.

Input:  an image path, its GT box (or None), and the config's [detect] table
Output: a list of detections, one per deer to classify:
            {'bbox': normalized [x, y, w, h], 'conf': float, 'source': 'gt' | 'megadetector' | 'full_frame',
             'no_detection': bool}

Fallback for unseen images (no GT row):
    'megadetector': run MegaDetector, keep 'animal' boxes above md_threshold (one crop each); if none,
                    use the full frame with no_detection = True.
    'full_frame':   always the full frame (no detector needed).

MegaDetector is optional and NOT a project dependency (the team ruled out off-the-shelf detectors for
the main pipeline). To use it: `uv add megadetector` (ask the team first; it is a shared-file change).
The GT boxes in the CSV have bbox_confidence 0.95-0.99, so they very likely already come from
MegaDetector; for the labeled data nothing changes.
"""

from pathlib import Path  # image paths

from PIL import Image  # MegaDetector takes a PIL image

from models.dual_resnet18.cropping import Box  # normalized box type

FULL_FRAME: Box = (0.0, 0.0, 1.0, 1.0)
MD_ANIMAL_CATEGORY = '1'


def load_megadetector(model_name: str) -> object:
    """Load a MegaDetector model (downloads weights on first use).

    Args:
        model_name: e.g. 'MDV5A'.

    Returns:
        A detector exposing generate_detections_one_image().
    """
    from megadetector.detection.run_detector import load_detector  # optional dependency, see module docstring

    return load_detector(model_name)


def megadetector_boxes(detector: object, path: Path, threshold: float) -> list[dict]:
    """Run MegaDetector on one image and keep animal boxes above the threshold.

    Args:
        detector: From load_megadetector.
        path: Image path.
        threshold: Minimum detection confidence.

    Returns:
        Detections (see module docstring), highest confidence first.
    """
    with Image.open(path) as img:
        result = detector.generate_detections_one_image(img.convert('RGB'), path.name, detection_threshold=threshold)
    animals = [d for d in result['detections'] if d['category'] == MD_ANIMAL_CATEGORY and d['conf'] >= threshold]
    animals.sort(key=lambda d: d['conf'], reverse=True)
    return [
        {'bbox': tuple(d['bbox']), 'conf': d['conf'], 'source': 'megadetector', 'no_detection': False} for d in animals
    ]


def resolve_boxes(path: Path, gt_bbox: Box | None, detect_cfg: dict, detector: object | None) -> list[dict]:
    """Pick the boxes to classify for one image.

    Args:
        path: Image path.
        gt_bbox: The image's GT box, or None if it has no CSV row.
        detect_cfg: The config's [detect] table.
        detector: Loaded MegaDetector, or None when fallback is 'full_frame'.

    Returns:
        One or more detections (see module docstring).
    """
    if gt_bbox is not None:
        return [{'bbox': gt_bbox, 'conf': float('nan'), 'source': 'gt', 'no_detection': False}]
    if detect_cfg['fallback'] == 'megadetector':
        boxes = megadetector_boxes(detector=detector, path=path, threshold=detect_cfg['md_threshold'])
        if boxes:
            return boxes
    return [{'bbox': FULL_FRAME, 'conf': float('nan'), 'source': 'full_frame', 'no_detection': True}]
