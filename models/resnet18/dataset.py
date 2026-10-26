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
import pandas as pd
from pathlib import Path
import tomllib
import torchvision.transforms.v2

age_mapping = {"ADULT": 0, "YOUNG": 1}
antler_mapping = {"ANTLERLESS": 0, "ANTLERED": 1}
splits = ["train", "val", "test"]

with open("../configs/default.toml", "rb") as f:
    data = tomllib.load(f)
root = Path("../..").resolve()
split_path = root / data['paths']['split']
df = pd.read_csv(split_path)



class Dataset:

    def __init__(self, split:str):
        """Load the split file and keep only the rows for the split you want"""
        if split not in splits:
            raise ValueError(f"Invalid split '{split}'. Expected one of: train, val, test.")

        self.split = split
        self.split_df = df[df['split'] == split]

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        """return the image and label for the row at index idx in self.split"""
        row = self.split.iloc[idx]
        image_name = row['filename']
        image_path = root / data['paths']['images'] / image_name
        label = row[data['train']['attribute']]
        return image_path, label