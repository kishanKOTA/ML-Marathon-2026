"""Loads the images and labels your model learns from.

Input:  your config's images folder + split file, and a split name ('train', 'val', or 'test')
Output: (image, label) pairs for that split, ready for your model

Purpose: pair each photo in one split with its age or antler label and turn it
into something PyTorch can feed to your model. The split is shared by the whole
team so scores are comparable: data/processed/split.csv has one row per image
(filename, labels, bounding box, camera, date) and a 'split' column. Read it
with pandas and keep the rows for the split you want.

What to include: reading an image, resizing/normalizing it (plus any
preprocessing you want to try), and returning it with its label.

Worth thinking about:
- What size should every image become?
- How should night (grayscale) images be handled?
"""

import os
import pandas as pd
import tomllib
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image

print(os.getcwd())

with open("./configs/default.toml", "rb") as f:
    data = tomllib.load(f)

age_mapping = {
    "YOUNG": 0,
    "ADULT": 1
}

root = Path("../..").resolve()
split_path = root / data['paths']['split']

df = pd.read_csv(split_path)
df_train = df[df['split'] == 'train']
df_labels = df_train[['filename', 'age_status']]
df_labels['age_status'] = df_labels['age_status'].map(age_mapping)

img_dir = data['paths']['images']

# Custom class needed to prepare data for model. Copied reference from torch docs
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, df_labels):
        self.img_labels = df_labels
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self):
        img_path = os.path.join(self.img_dir, self.img_labels['filename'])
        image = decode_image(img_path)
        label = self.img_labels['age_status']
        return image, label

        # Returns tuple (SSWI000000027884533B.jp, 0)


'''
Here's what's still missing for it to work as a dataset, roughly in the order I'd build it:

1. Row lookup by index. __getitem__(self, idx) should pull one row with .iloc[idx], then read that row's filename and label.
2. A split argument. Filter to 'train', 'val' or 'test' inside __init__, so one class can build all three datasets. Right now the filter runs at import time and is fixed to 'train'.
3. Config and paths inside the class, built from the repo root. Take the config (or its path) as an argument and build both the imagefile__), so it works when run from the repo root.
4. Label choice from the config. Use attribute to pick age_status or antler_status, and add a mapping for antlers (ANTLERLESS/ANTLERED).
5. Image preprocessing:
   - force 3 channels (ImageReadMode decode as 1 channel
   - resize to image_size
   - convert uint8 0–255 to float 0–
   - a transform hook where your preprocessing experiments can plug in later
6. Return types. Return the image as as an int or tensor, so theDataLoader can batch them.
7. Docstrings and type hints once th

'''