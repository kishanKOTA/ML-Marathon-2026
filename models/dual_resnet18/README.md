# Dual-head ResNet-18

**Owner**: Lucas

Shared step-by-step guide, label rule, and resources: `models/README.md`. Design rationale: `Handoff Dual-Head ResNet18 for Deer Age & Antler Classification.md`.

**Architecture**: one ResNet-18 backbone (ImageNet-pretrained) plus a small month MLP, with two heads: age (adult/young) and antler (antlerless/antlered). Both labels come from one forward pass, trained with one summed loss. At inference, the impossible antlered + young combination is removed from the joint probability table.

**What I'm exploring**: whether sharing one backbone across both tasks, adding the capture month, and enforcing the antlered-young constraint beat the two separate ResNet-18 classifiers.

## Files

| File | What | Who |
|---|---|---|
| `constants.py` | Fixed class order (team rule: age 0 adult / 1 young; antler 0 antlerless / 1 antlered) | Claude |
| `cropping.py` | Square crop around a box, shifted inward at edges, padded with the ImageNet mean when larger than the image | Claude |
| `dataset.py` | `DualDeerDataset`: GT-box crop + `[sin, cos]` month + both labels + metadata; `encode_month`, `month_from_exif` | Claude |
| `detect.py` | Box per image: GT box if known, else MegaDetector (optional) or full frame | Claude |
| `model.py` | `DualResNet18`: **skeleton with TODOs** | **Lucas** |
| `losses.py`, `train.py` | Dual loss (`L = L_age + lam * L_antler`) and training loop | **Lucas** (team rule: members write their own training loop and loss/optimizer) |
| `inference.py` | Joint table + hard constraint + review flag -> predictions CSV | Claude |
| `evaluate.py` | Scores a predictions CSV: per-head F1, confusion, breakdowns, calibration | Claude |
| `configs/default.toml` | All settings, including `[train]` values for your `train.py` | Claude |

Tests: `tests/test_dual_resnet18.py` (model interface; fails until `model.py` is done) and `tests/test_dual_resnet18_pipeline.py` (cropping, month, constraint, data).

## Interface the rest of the code relies on

- `DualResNet18(n_age=2, n_antler=2, month_dim=2, month_hidden=16, pretrained=True)` (built from the config's `[model]` table)
- `forward(image (B, 3, 224, 224), month (B, 2)) -> (age_logits (B, 2), antler_logits (B, n_antler))`, raw logits
- Run naming: `load_config(name="default.toml")` sets `config['run_name'] = 'default'` (the file stem) and `config['run_dir']`. Save the best model with `torch.save(model.state_dict(), checkpoint_path(config=config))` (-> `logs/runs/dual_resnet18/default_best.pt`); inference loads from the same path
- A dataset batch is a dict: `batch['image']`, `batch['month']`, `batch['age_label']`, `batch['antler_label']`, `batch['meta']`
- For class weights: `DualDeerDataset.class_counts()` gives per-head counts

## How to run (from the repo root)

```bash
uv run python -m models.dual_resnet18.dataset                       # sanity check: shapes, counts
uv run pytest tests/test_dual_resnet18.py                            # after writing model.py
uv run python -m models.dual_resnet18.train                          # your training loop
uv run python -m models.dual_resnet18.inference --split val          # -> logs/runs/dual_resnet18/default_val_predictions.csv
uv run python -m models.dual_resnet18.evaluate logs/runs/dual_resnet18/default_val_predictions.csv
uv run python -m models.dual_resnet18.inference --images some/folder # unseen images: GT box if known, else [detect] fallback
```

Ablations are just separate configs: copy `default.toml` to e.g. `no_month.toml`, set `use_month = false`, train, and evaluate both. The copy's outputs are automatically named `no_month_*`, so nothing is overwritten. Compare against the separate models by passing their predictions (`filename, age_pred` / `filename, antler_pred`, team label order) as `--baseline`.

## Cropping: refinement of the `resnet18` bbox crop

Amanda's `models/resnet18` (`crop = true`) introduced training on the GT-box crop instead of the full frame. This model keeps that idea and changes how the crop is shaped, so the deer is never stretched and the head/antlers aren't randomly cut away:

| | `resnet18` (`run4_cropaug`) | `dual_resnet18` (`cropping.py`) |
|---|---|---|
| Window | Box + 10% of box w/h on each side; keeps the box's aspect ratio | Square, side = longer box side x 1.15, centered on the box |
| At image edges | Clipped (window shrinks) | Shifted inward to keep the full square; padded with the ImageNet mean only if the square is larger than the image |
| To 224x224 | `Resize((224, 224))`: squashes non-square crops (a wide deer gets narrower) | Square already, so the resize keeps proportions |
| Train-time randomness | `RandomResizedCrop(scale 0.7-1.0)` after cropping: can drop up to 30% of the crop, incl. head/antlers | Box jitter before cropping (center +/-10%, size +/-15%), then a mild `RandomAffine` (shift 5%, scale 0.95-1.05) |
| Color augmentation | `ColorJitter` brightness/contrast 0.3, saturation 0.2; `RandomGrayscale(p=0.3)` | `ColorJitter` brightness/contrast 0.2, no saturation (does nothing on gray night images, unrealistic on day ones); `RandomGrayscale(p=0.3)` kept from `resnet18` |

Comparisons against the separate `resnet18` models therefore measure the whole pipeline (crop + augmentation + shared heads + month + constraint), not the dual-head architecture alone.

## Assumptions and deviations from the handoff

- **Crops come from the GT boxes.** The team ruled out off-the-shelf detectors, and the CSV boxes have `bbox_confidence` 0.95-0.99, so they very likely already are MegaDetector output. MegaDetector is only a fallback for images without a GT row. It is not a dependency (`uv add megadetector`, ask the team first), its API call in `detect.py` is untested, and `md_threshold` is untuned.
- **Antler head is 2-class.** No `not_visible` labels exist, so the handoff's third class was dropped. Headless images are handled only at inference: a low-confidence antler head gives `adult_undetermined`.
- **No masked (-1) antler labels.** Every image is ANTLERED or ANTLERLESS, so the handoff's `ignore_index` masking was removed; the antler loss is plain cross-entropy over all images.
- **No head-visible label.** Breakdowns use `bbox_touches_edge` as a weak proxy (52.6% of images).
- **Month** comes from the CSV `month` column (0 missing). Unseen images use EXIF `DateTimeOriginal`, or `[0, 0]` if absent. In val, months 1-4 contain no fawns or antlered deer, so month is a strong signal; the `use_month = false` ablation shows how much the model leans on it.
- **Config is TOML** (repo convention), not YAML.

## What I've tried / what I learned

_(Short notes as you go.)_

---
AI use disclaimer: Claude Code was used to plan and generate the data, cropping, inference, and evaluation plumbing and tests in this folder. The model (`model.py`), loss, and training loop are written by the owner. All code was reviewed.
