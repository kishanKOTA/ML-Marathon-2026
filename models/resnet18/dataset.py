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
import torch
import sys
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

# find the files
THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]

cfg_name = sys.argv[1] if len(sys.argv) > 1 else "default.toml"
with open(THIS_DIR / "configs" / cfg_name, "rb") as f:
    config = tomllib.load(f)
print("config:", cfg_name)

SPLIT_PATH = ROOT / config["paths"]["split"]
IMG_DIR = ROOT / config["paths"]["images"]

AGE_MAP = {"ADULT": 0, "YOUNG": 1}
ANTLER_MAP = {"ANTLERLESS": 0, "ANTLERED": 1}
# splits = ["train", "val", "test"]
# df = pd.read_csv(SPLIT_PATH)



class DeerDataset(Dataset):

    def __init__(self, attribute, split_name, transform=None, crop=False):
        if split_name not in ["train", "val", "test"]:
            raise ValueError(f"Bad split '{split_name}'")

        if attribute == "age":
            self.label_col = "age_status"
            label_map = AGE_MAP
        elif attribute == "antlers":
            self.label_col = "antler_status"
            label_map = ANTLER_MAP
        else:
            raise ValueError(f"attribute must be 'age' or 'antlers', got '{attribute}'")

        df = pd.read_csv(SPLIT_PATH)
        df = df[df["split"] == split_name].copy()

        df[self.label_col] = df[self.label_col].map(label_map) # swaps every text label for its number 0/1
        if df[self.label_col].isna().any():
            raise ValueError("some labels were not in the mapping")

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """return the image and label for the row at index idx in self.split"""
        row = self.df.iloc[idx]
        img_path = IMG_DIR / row['filename']
        img = decode_image(str(img_path), mode="RGB")

        if self.crop:
            _,H,W = img.shape
            pad = 0.1
            x0 = max(0, int((row.bbox_origin_x - pad * row.bbox_width) * W))
            y0 = max(0, int((row.bbox_origin_y - pad * row.bbox_height) * H))
            x1 = min(W, int((row.bbox_origin_x + (1 + pad) * row.bbox_width) * W))
            y1 = min(H, int((row.bbox_origin_y + (1 + pad) * row.bbox_height) * H))
            img = img[:, y0:y1, x0:x1]

        if self.transform is not None:
            img = self.transform(img)

        label = int(row[self.label_col])
        return img, label

def make_transform(img_size):
    return v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

def make_train_transform(img_size):
    return v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomGrayscale(p=0.3),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# test the code above
if __name__ == "__main__":
    ds = DeerDataset("age", "train", transform=make_transform(224))
    print("number of samples:", len(ds))

    image, label = ds[0]
    print("shape:", image.shape)     # expect torch.Size([3, 224, 224])
    print("dtype:", image.dtype)     # expect torch.float32
    print("label:", label)           # expect 0 or 1

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    images, labels = next(iter(loader))
    print("batch:", images.shape, labels)   # expect [8, 3, 224, 224]