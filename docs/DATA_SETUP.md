# Data setup

Getting the competition data onto your machine. Everything lands in the gitignored `data/`. Steps 1-2 are all the default setup needs; steps 3-4 are only for experiments that train on bounding-box crops.

## 1. Download

After Kaggle credential setup (`README.md`):

```bash
uv run scripts/fetch_data.py    # skips if already set up; --force re-copies
```

```
data/raw/
  Snapshot_WI-Oh_Deer_data-v2.csv   # per image: filename, bbox, age_status, antler_status, camera_location_seq_no, ...
  Snapshot_WI-Oh_Deer_locs-v2.csv   # county-level camera locations
  images/*.jpg                      # 4880 images, flat
```

Works the same on Windows, macOS, and Linux/WSL. The kagglehub cache is `~/.cache/kagglehub/` (Windows: `C:\Users\<you>\.cache\kagglehub\`). Images are copied, not symlinked, because symlinks need admin rights on Windows.

- **WSL**: clone the repo inside the Linux filesystem (e.g. `~/projects/`), not `/mnt/c/`. Reading images through `/mnt/c/` every epoch is much slower.
- **Windows**: don't clone into a OneDrive-synced folder, or it will try to upload 4880 images.

## 2. Make the shared split

```bash
uv run scripts/make_split.py
```

Writes `data/processed/split.csv`: the labels CSV plus a `split` column (`train` ~71%, `val` ~14%, `test` ~14%). Everyone loads this same file, so scores are comparable. It's deterministic; the printed fingerprint (`673e0097` as of 9/27) should match across machines.

- **Grouped by camera**: shots from one camera share background, lighting, and often the same deer. A random split puts near-duplicates in train and val/test, so scores reward memorizing backgrounds. Each camera lands entirely in one split.
- **Stratified by class**: each split keeps about the overall balance (20% YOUNG, 30% ANTLERED).
- **Settings** live in `configs/split.toml` (seed, number of folds, output path). Report results on this official split. To check that an improvement isn't just luck, copy the file with a new `seed` and `output` and run `uv run scripts/make_split.py --config configs/<copy>.toml`. Scores from different splits aren't comparable with each other.

## 3. (Optional) Crop each deer by its bounding box

**Pending**: lives on `origin/crop-images` as `scripts/draw_bboxes.py` (20px pad). Before merging: rename to `crop_bboxes.py`, read from `data/raw/`, and write to `data/processed/cropped/`. Its `preprocess_for_yolo11n.py` is no longer needed.

## 4. (Optional) Sort crops into class folders

```bash
uv run scripts/build_crop_datasets.py --input-dir data/processed/cropped
```

Uses `split.csv`, so crops get the same split as everyone else:

```
data/processed/crops/<age|antler>/<train|val|test>/<CLASS>/*.jpg
```
