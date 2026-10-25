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
from torch.utils.data import Dataset    # Dataset stores the samples and their corresponding labels
from torchvision.io import decode_image # Converts image to tensor
from torchvision.transforms import v2   # Contains different transform functions
from torchvision.transforms.functional import InterpolationMode
from torch import float32

# Defines paths that work from wherever the file is called
root = Path(__file__).resolve().parents[2]
config_path = root / Path("workspaces/lucas/configs/default.toml")

# Load configuration file
with open(config_path, "rb") as f:
    data = tomllib.load(f)

split_path = root / data['paths']['split']
img_dir = root / data['paths']['images']

age_mapping = {"ADULT": 0, "YOUNG": 1}
antler_mapping = {"ANTLERLESS": 0, "ANTLERED": 1}

# Default transformation applied to every image loaded 
default_transforms = v2.Compose([
            v2.ToImage(),   # ?
            v2.ToDtype(float32, scale=True), # Converts pixel values to floats from 0-1
            v2.Resize(size=(data['train']['image_size'], data['train']['image_size']), interpolation=InterpolationMode.BILINEAR, antialias=True)
        ])

# Object initialization needed to prepare data for model. Copied reference from torch docs
class DeerDataset(Dataset):
    def __init__(self, img_dir, attribute, split_path, split_name, transforms=default_transforms):

        df = pd.read_csv(split_path)
        df_split = df[df['split'] == split_name]

        if attribute == "age":
            attribute_col = "age_status"
            attribute_map = age_mapping
        if attribute == "antlers":
            attribute_col = "antler_status"
            attribute_map = antler_mapping

        df_labels = df_split[['filename', attribute_col]]
        df_labels[attribute_col] = df_labels[attribute_col].map(attribute_map)

        self.attribute_col = attribute_col
        self.img_labels = df_labels
        self.img_dir = img_dir
        self.transforms = transforms


    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels['filename'].iloc[idx])
        image = decode_image(img_path, mode="RGB") # converts image to a tensor

        if self.transforms is not None:
            image = self.transforms(image)

        label = self.img_labels[self.attribute_col].iloc[idx]
        return image, label
    
        # Returns (tensor, label) tuple pair