# Handoff: Dual-Head ResNet18 for Deer Age & Antler Classification

Nov 22, 2026 · @Lucas

## Executive summary

We are replacing two separate ResNet18 classifiers (antler status and age) with one multi-task model that returns both labels from a single forward pass. The new model lives in `./models/dual_resnet18`.

**Why not a cascade (antlers first, then age)?** It would rule out antlered young deer by design, but many camera trap images crop out the head. A cascade would force a guess on antlers first and pass that error on to the age decision. Young deer are often identified by body spots, which do not need the head at all.

**Why not one flat 3-class classifier?** It also removes the impossible class, but it cannot say "antler status unknown" when the head is not visible. It also cannot use headless images whose age is known but antler status is not.

**What we chose:**

- **Detect and crop first.** MegaDetector finds the deer, and we take a square crop around it (with edge handling) from the original image. Resizing a full frame to 224×224 would erase spots and small antlers. We first measure MegaDetector against our ground truth boxes before deciding whether fine-tuning is needed.
- **Shared backbone, two inputs.** The image goes through a ResNet18 backbone (512 features). The month, encoded as sin/cos so December sits next to January, goes through a small MLP (16 features). The two vectors are concatenated.
- **Two heads.** One head predicts age (adult/young), the other antler status (antlered/antlerless/not visible).
- **Combined loss.** Each head has its own cross-entropy loss. The losses are summed (weighted) and backpropagated once. Unknown antler labels are masked so those images still train the age head.
- **Hard constraint at inference.** Head probabilities are multiplied into a joint table. The antlered + young cell is always set to zero and the rest renormalized. Large removed mass flags the image for review.

Success means the combined model matches or beats the two current models, with results reported separately for images with the head visible and with the head cropped.

## Instructions for Claude Code

Read the existing repository before writing any code. Reuse what is already there; do not duplicate it.

**Before implementing, inspect and report back on:**

1. The existing ResNet18 implementation(s): file locations, how the backbone is built (torchvision weights, custom layers), and how the two current models are trained.
2. The data layout: where images, labels and ground truth bounding boxes live, and their formats (CSV, COCO JSON, etc.).
3. Where the capture date/month comes from (EXIF, filename, metadata CSV) and how many images lack it.
4. Existing utilities to reuse: datasets, transforms, training loops, metrics, config handling, logging.
5. The framework and versions in use. This spec assumes PyTorch; flag it if the repo differs.

Summarize findings and any conflicts with this spec before writing code.

**Scope:**

- **Generate:** everything except the model class. That covers the MegaDetector evaluation script, square cropping, the dataset with month encoding, the loss function, the training loop, inference with the hard constraint, evaluation and configs.
- **Do not implement:** the model class in `./models/dual_resnet18/model.py`. Create that file with the exact skeleton in Step 4 and leave the TODOs for the user. The user is writing it to learn how to build a dual-input, dual-output network.
- **Code against the interface, not the implementation.** All other code must rely only on the interface contract in Step 4, so it works once the user fills in the model.
- **Write a test** (`tests/test_dual_resnet18.py`) that checks the interface contract (output shapes, dtypes, gradient flow to both branches). It will fail until the user finishes the model, which is expected.

**Suggested layout** (adapt to repo conventions):

```
./models/dual_resnet18/
    model.py          # SKELETON ONLY: user implements
    dataset.py        # crops + month encoding + labels
    cropping.py       # square crop with edge handling
    losses.py         # combined masked loss
    train.py
    inference.py      # joint probabilities + hard constraint
    evaluate.py
    config.yaml
./scripts/eval_megadetector.py
./tests/test_dual_resnet18.py
```

## Label conventions

The class ordering below is fixed. Training labels, model outputs and the inference constraint all depend on it. Define it once in a shared constants module and import it everywhere.

| Head | Index 0 | Index 1 | Index 2 | Masked value |
| --- | --- | --- | --- | --- |
| Age | adult | young |  |  |
| Antler | antlered | antlerless | not\_visible | -1 (`ignore_index`) |

- **not\_visible:** the head is cropped out or hidden, so antler status cannot be judged. This is a real class the model learns to predict.
- **-1:** antler label is missing or uncertain for another reason. These samples are excluded from the antler loss only.
- The impossible combination antlered + young is joint table cell `[0, 1]` (antler row 0, age column 1).
- If existing labels use a different scheme, write a mapping in `dataset.py`. Do not change the order above.

Open question for the user: do current annotations distinguish "head not visible" from "antlerless"? If not, a relabeling pass of headless images is needed.

## Step 1: Evaluate MegaDetector against ground truth boxes

Measure MegaDetector on our labeled data before building on it. Fine-tune only if this evaluation shows a real gap. Fine-tuning adds work and can reduce robustness at new camera sites.

**Script:** `scripts/eval_megadetector.py`

1. Run MegaDetector (latest stable release; check whether the repo already pins one) on every image that has a ground truth box. Keep only the `animal` category.
2. Match each ground truth box to the highest-IoU detection.
3. Report, overall and broken down by camera site, day/night and (if available) head visible vs. cropped:
   - Recall at IoU ≥ 0.5, across confidence thresholds 0.1 to 0.8 in steps of 0.1.
   - Mean IoU of matched boxes.
   - Rate of images with no detection above threshold.
   - Rate of images where the top detection is not the labeled deer (wrong animal or object).
4. Save a CSV of failures and a folder of 50 sample failure images with both boxes drawn, for manual review.
5. Recommend a confidence threshold that balances recall against false detections.

**Decision rule:** if recall at IoU ≥ 0.5 is at least 97% at the chosen threshold, use MegaDetector as is. Below that, report which conditions fail and pause for a fine-tuning decision by the user. Do not fine-tune automatically.

**Detection policy for training and inference:**

- One crop per detection above threshold. Multiple deer in a frame each get classified separately.
- No detection above threshold: fall back to the full frame and set a `no_detection` flag in outputs.
- Train the classifier mainly on MegaDetector crops, so training matches deployment. Ground truth box crops with random jitter (shift ±10%, scale ±15%) may be added as augmentation.

## Step 2: Square crop with edge handling

Every detection becomes a square crop taken from the original full-resolution image, then resized to 224×224. Never stretch a non-square box. The same function must be used in training and inference.

**Module:** `models/dual_resnet18/cropping.py`, function `square_crop(image, bbox, margin=0.15, out_size=224)`.

1. Convert the box to pixel coordinates (MegaDetector boxes are normalized `[x, y, w, h]`; check the ground truth format in the repo).
2. Side length = `max(w, h) × (1 + margin)`, rounded to an integer.
3. Center the square on the box center.
4. **Edge handling:** if the square extends past an image border, shift it inward until it fits. The deer will no longer be centered, which is fine.
5. If the side is larger than the image's width or height, shifting cannot fit it. Clip to the image and pad the missing part with the ImageNet mean color.
6. Resize to `out_size × out_size` with bilinear interpolation (antialiasing on).

```python
side = int(max(w, h) * (1 + margin))
x0 = min(max(cx - side // 2, 0), img_w - side)   # shift inward
y0 = min(max(cy - side // 2, 0), img_h - side)
# if side > img_w or side > img_h: clip and pad the remainder
```

**Unit tests:** a centered box, a box touching each of the four edges, a box in a corner, a box taller than the image, and a tiny box (upsampled). Each test asserts the full original box lies inside the crop and the output is 224×224.

**Config:** expose `margin` and `crop_method` in `config.yaml`. Implement `square` now; leave a stub for `letterbox` so it can be compared later.

**Training augmentation** (after cropping): random horizontal flip, small random shift/scale, color jitter. No vertical flips. Use ImageNet normalization.

## Step 3: Input specification

Each sample is an (image, month) pair plus two labels. The dataset in `models/dual_resnet18/dataset.py` returns:

| Field | Shape | Dtype | Content |
| --- | --- | --- | --- |
| `image` | (3, 224, 224) | float32 | Square crop from Step 2, ImageNet-normalized |
| `month` | (2,) | float32 | `[sin(2π·m/12), cos(2π·m/12)]`, m = 1 to 12 |
| `age_label` | () | int64 | 0 adult, 1 young |
| `antler_label` | () | int64 | 0 antlered, 1 antlerless, 2 not\_visible, or -1 masked |
| `meta` | dict | mixed | image id, crop source (MegaDetector / GT / full frame), site, head-visible flag |

A batch therefore has `image` of shape (B, 3, 224, 224) and `month` of shape (B, 2).

**Why two numbers for the month:** a single value like m/12 puts December (1.0) far from January (0.08). The sin/cos pair places months on a circle, so they wrap correctly.

**Month source:** use whatever the repo inspection finds (EXIF `DateTimeOriginal`, filename or metadata). Write a helper `encode_month(m)` used by both training and inference.

**Missing dates:** encode as `[0, 0]` (the circle's center, which matches no real month). Report how many images this affects.

**Splits:** split by camera site (or by site and season), not by random image. Burst images from the same trigger must never land in different splits. If the repo already has a split, reuse it so results are comparable with the current models.

## Step 4: Model architecture (skeleton only; the user implements)

Claude Code creates `models/dual_resnet18/model.py` with exactly the skeleton below and does **not** fill in the TODOs. All other code depends only on this interface contract:

- **Constructor:** `DualResNet18(n_age=2, n_antler=3, month_dim=2, month_hidden=16, pretrained=True)`
- **Forward:** `forward(image, month) -> (age_logits, antler_logits)`
- **Shapes:** image (B, 3, 224, 224), month (B, 2) → age\_logits (B, 2), antler\_logits (B, 3)
- **Outputs are raw logits.** No softmax inside the model; the loss and inference code apply it.

**Data flow:** image → ResNet18 backbone → 512 features. In parallel, month → small MLP → 16 features. Concatenate → 528 features. That vector feeds the age head (528 → 2) and the antler head (528 → 3).

```python
"""
DualResNet18: one shared model, two inputs (image, month), two outputs (age, antler).

THIS FILE IS FOR YOU TO IMPLEMENT. Everything else in models/dual_resnet18/
codes against the interface below, so keep the names, arguments and return types.
Run tests/test_dual_resnet18.py when you are done; it should pass.

Before starting, look at the existing ResNet18 code in this repo. You can reuse
how it loads the backbone.
"""
import torch
import torch.nn as nn
import torchvision


class DualResNet18(nn.Module):
    def __init__(self, n_age=2, n_antler=3, month_dim=2, month_hidden=16, pretrained=True):
        super().__init__()

        # ---------------------------------------------------------------
        # TODO 1: IMAGE BRANCH (the backbone)
        # ---------------------------------------------------------------
        # Goal: turn a (B, 3, 224, 224) image into a (B, 512) feature vector.
        #
        # - Load torchvision.models.resnet18, with ImageNet weights if
        #   `pretrained` is True (weights="IMAGENET1K_V1"), or reuse the
        #   repo's existing loader.
        # - A standard ResNet18 ends in `fc`, a Linear(512, 1000) layer that
        #   classifies ImageNet. You do NOT want that layer. You want the
        #   512 features that go INTO it.
        # - Two common ways to remove it:
        #     a) nn.Sequential(*list(resnet.children())[:-1])
        #        (keeps everything up to and including the avgpool;
        #         output is (B, 512, 1, 1), so you flatten it in forward)
        #     b) replace it: resnet.fc = nn.Identity()
        #        (output is already (B, 512))
        # - Store the result as self.backbone. Note which option you chose,
        #   because it changes whether forward() needs a flatten.
        # - Hint: resnet.fc.in_features gives you 512 without hard-coding it.
        #   Save it as self.img_dim.

        # ---------------------------------------------------------------
        # TODO 2: MONTH BRANCH (small MLP)
        # ---------------------------------------------------------------
        # Goal: turn the (B, 2) sin/cos month into a (B, month_hidden) vector.
        #
        # - nn.Sequential with a Linear(month_dim, month_hidden) followed by
        #   a ReLU is enough. Store it as self.month_mlp.
        # - Why a learned layer instead of concatenating the raw 2 numbers?
        #   It lets the model build its own month features (e.g. "early
        #   summer") before they meet the image features.

        # ---------------------------------------------------------------
        # TODO 3: THE TWO HEADS
        # ---------------------------------------------------------------
        # Goal: two separate classifiers that read the SAME combined vector.
        #
        # - Combined size = self.img_dim + month_hidden (512 + 16 = 528).
        # - self.age_head    = Linear(combined, n_age)     -> adult / young
        # - self.antler_head = Linear(combined, n_antler)  -> antlered /
        #                                                     antlerless /
        #                                                     not_visible
        # - Each head has its own weights. Positions 513-528 of the input
        #   always hold month features, so the weights at those positions
        #   learn how much the month should matter. Nothing labels them as
        #   "month"; consistent position is what makes it work.
        # - Optional: add nn.Dropout(p=0.2) before each head to reduce
        #   overfitting.

        raise NotImplementedError("TODO: build the layers above")

    def forward(self, image, month):
        # ---------------------------------------------------------------
        # TODO 4: FORWARD PASS
        # ---------------------------------------------------------------
        # image: (B, 3, 224, 224)   month: (B, 2)
        #
        # Step A: img_feat = image through self.backbone.
        #         If you used option (a) in TODO 1, flatten it:
        #         img_feat = img_feat.flatten(1)   -> (B, 512)
        # Step B: month_feat = month through self.month_mlp -> (B, 16)
        #         (the two branches are independent; order does not matter)
        # Step C: combine with torch.cat([img_feat, month_feat], dim=1)
        #         -> (B, 528). dim=1 joins along features, not the batch.
        #         The image features MUST come first, every time.
        # Step D: age_logits    = self.age_head(combined)     -> (B, 2)
        #         antler_logits = self.antler_head(combined)  -> (B, 3)
        # Step E: return age_logits, antler_logits   (a tuple, raw logits,
        #         NO softmax; cross_entropy applies it internally)
        #
        # How gradients flow back (you don't write this; autograd does):
        # the loss gradient for the 528-vector is split by torch.cat,
        # the first 512 values go to the backbone, the last 16 to month_mlp.
        raise NotImplementedError("TODO: implement forward")


# ---------------------------------------------------------------------
# OPTIONAL TODO 5: freezing helpers for staged training
# ---------------------------------------------------------------------
# def freeze_backbone(self): set requires_grad=False on self.backbone params
# def unfreeze_backbone(self): set it back to True
# train.py calls these if they exist (warm up the heads first, then
# fine-tune everything with a lower learning rate).
```

**Interface test** (`tests/test_dual_resnet18.py`, generated by Claude Code) checks:

1. Output is a tuple of two tensors with shapes (B, 2) and (B, 3) for B = 4.
2. Outputs are not probabilities (they need not sum to 1).
3. After a backward pass on the summed loss, both a backbone parameter and a month MLP parameter have non-zero gradients.
4. Changing only the month input changes the outputs (the month branch is actually wired in).

## Step 5: Combined loss and training

Each head gets its own cross-entropy loss. The two are summed with a weight and backpropagated in one call, so the shared backbone and month MLP learn from both tasks.

```latex
L = L_{age} + \lambda \, L_{antler}
```

**Module:** `models/dual_resnet18/losses.py`

```python
def dual_loss(age_logits, antler_logits, age_y, antler_y,
              lam=1.0, age_weight=None, antler_weight=None):
    loss_age = F.cross_entropy(age_logits, age_y, weight=age_weight)
    valid = antler_y != -1
    if valid.any():                     # all-masked batch -> NaN otherwise
        loss_antler = F.cross_entropy(antler_logits, antler_y,
                                      weight=antler_weight, ignore_index=-1)
    else:
        loss_antler = antler_logits.sum() * 0.0   # keeps the graph, adds nothing
    total = loss_age + lam * loss_antler
    return total, loss_age.detach(), loss_antler.detach()
```

- **Masking:** samples with antler label -1 add nothing to the antler loss but still train the age head and the shared layers.
- **Class weights:** compute inverse-frequency weights from the training split for each head (young is likely rare). Make them a config option, on by default.
- **λ:** start at 1.0. Log both losses separately. If one task lags, try 0.5 or 2.0.

**Training loop** (`train.py`; reuse the repo's existing loop where possible):

1. Unpack the batch: `image, month, age_y, antler_y`.
2. `age_logits, antler_logits = model(image, month)`
3. `loss, l_age, l_ant = dual_loss(...)`
4. `optimizer.zero_grad(); loss.backward(); optimizer.step()`, a single backward pass for both heads.
5. Log total loss, each head's loss, and per-head validation accuracy and macro-F1 every epoch.

**Defaults** (all in `config.yaml`): AdamW, learning rate 1e-3 for heads and month MLP, 1e-4 for the backbone, weight decay 1e-4, cosine schedule, batch size 64, up to 30 epochs. Early stop on mean validation macro-F1 across the two heads. If the model has `freeze_backbone`, train only heads and month MLP for the first 3 epochs.

**Month ablation:** support `use_month: false` in config, which feeds zeros for the month input. Train both variants so we can measure what the month adds.

## Step 6: Inference with the hard constraint

The two heads' probabilities are multiplied into a 2×2 joint table. Cell `[0, 1]` (antlered + young) is always set to zero, whatever its value, and the remaining cells are renormalized. There is no threshold on that cell; it is removed because it is biologically impossible. In probability terms, this conditions the prediction on the combination being valid.

**Module:** `models/dual_resnet18/inference.py`, applied per crop.

1. Detect and crop (Steps 1 and 2), encode the month (Step 3).
2. `age_logits, antler_logits = model(image, month)`, then softmax each head separately.
3. **Check antler visibility first.** Let `p_ant = [P(antlered), P(antlerless), P(not_visible)]`. If `not_visible` is the top class, or `max(P(antlered), P(antlerless)) / (P(antlered) + P(antlerless))` is below `antler_conf_threshold` (default 0.6), return the age result only:
   - young → "young, antlerless (implied)"
   - adult → "adult, antler status undetermined"
4. **Otherwise build the joint table.** Renormalize antlered/antlerless to sum to 1, then:

```python
p_age = softmax(age_logits)                  # [P(adult), P(young)]
p_ant2 = p_ant[:2] / p_ant[:2].sum()         # [P(antlered), P(antlerless)]
joint = torch.outer(p_ant2, p_age)           # rows: antler, cols: age
removed = joint[0, 1].item()                 # mass on the impossible cell
joint[0, 1] = 0                              # always, no threshold
joint = joint / joint.sum()                  # renormalize the 3 valid cells
```

5. **Disagreement flag.** If `removed > review_threshold` (default 0.5), the heads strongly contradict each other. Set `needs_review = True`. This threshold only flags images; it never decides whether to zero the cell.
6. Return the top valid combination with its probability.

**Worked example** (use as a unit test): P(adult) = 0.85, P(young) = 0.15, P(antlered) = 0.70, P(antlerless) = 0.30.

| Combination | Joint | After constraint |
| --- | --- | --- |
| antlered, adult | 0.595 | 0.665 |
| antlerless, adult | 0.255 | 0.285 |
| antlered, young | 0.105 | 0 (removed) |
| antlerless, young | 0.045 | 0.050 |

Prediction: antlered adult at 0.665. Removed mass 0.105, so no review flag.

**Output per crop** (CSV/JSON): image id, bbox, detection confidence, `no_detection`, predicted label, its probability, all three valid joint probabilities, raw per-head probabilities, `antler_status` (determined / undetermined), `removed_mass`, `needs_review`.

## Evaluation plan and deliverables

The new model is judged against the two current separate ResNet18 models on the same test split, using the final combined label.

**Script:** `models/dual_resnet18/evaluate.py`. Report:

- Confusion matrix over the three valid combined classes, plus the "adult, undetermined" outcome.
- Per-head accuracy and macro-F1, for the new model and both current models.
- All metrics split by **head visible vs. head cropped**. This is the comparison that tests the core motivation.
- Breakdown by month and by camera site, to catch seasonal errors (late fawns losing spots, bucks after antler drop).
- Rate of `needs_review` flags and `no_detection` fallbacks.
- Calibration (reliability plot or expected calibration error) for each head, since the joint table assumes meaningful probabilities.
- Ablations: with vs. without month; MegaDetector crops vs. full frames.

**Deliverables checklist:**

- [ ] Repo inspection summary (Instructions section) shared before coding
- [ ] `scripts/eval_megadetector.py` with recall/IoU report and threshold recommendation
- [ ] `cropping.py` with square crop, edge handling and unit tests
- [ ] Shared label constants and `dataset.py` returning (image, month, age, antler, meta)
- [ ] `model.py` skeleton with TODOs, left unimplemented
- [ ] `tests/test_dual_resnet18.py` interface test
- [ ] `losses.py` with masking and the all-masked guard
- [ ] `train.py` and `config.yaml` with month ablation switch
- [ ] `inference.py` with hard constraint, worked-example unit test and review flag
- [ ] `evaluate.py` with the comparison report above
- [ ] Short README in `./models/dual_resnet18` explaining how to train, run inference and evaluate
