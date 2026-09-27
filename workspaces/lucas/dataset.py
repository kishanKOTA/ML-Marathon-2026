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
