"""
Script to crop bounding boxes from CSV data and save them.
"""

import csv
import os
from pathlib import Path
from PIL import Image
import pandas as pd

# Configuration
DATA_CSV = "snapshot-wi-oh-deer/Snapshot_WI-Oh_Deer_data-v2.csv"
PHOTOS_DIR = "snapshot-wi-oh-deer/Snapshot_WI-Oh_Deer_Photos"
OUTPUT_DIR = "Snapshot_WI-Oh_Deer_Photos_Cropped"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the CSV data
df = pd.read_csv(DATA_CSV)

print(f"Loading {len(df)} image records from {DATA_CSV}")

# Process each row
for idx, row in df.iterrows():
    filename = row['filename']
    
    # Extract subdirectory from filename pattern or use a default
    # Assuming structure like SSWI000000027884533B.jpg might have a subdirectory
    # Based on workspace, there's a Deer/ subdirectory
    image_path = os.path.join(PHOTOS_DIR, "Deer", filename)
    
    # If not found in Deer/, try root
    if not os.path.exists(image_path):
        image_path = os.path.join(PHOTOS_DIR, filename)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        continue
    
    try:
        # Open image
        img = Image.open(image_path)
        width, height = img.size
        
        # Get bounding box coordinates (normalized 0-1)
        bbox_x = float(row['bbox_origin_x'])
        bbox_y = float(row['bbox_origin_y'])
        bbox_w = float(row['bbox_width'])
        bbox_h = float(row['bbox_height'])
        
        # Convert to pixel coordinates
        # x0 = int(bbox_x * width)
        # y0 = int(bbox_y * height)
        # x1 = int((bbox_x + bbox_w) * width)
        # y1 = int((bbox_y + bbox_h) * height)
        
        # Add 20px padding on all sides
        padding = 20
        x0 = max(0, int(bbox_x * width) - padding)
        y0 = max(0, int(bbox_y * height) - padding)
        x1 = min(width, int((bbox_x + bbox_w) * width) + padding)
        y1 = min(height, int((bbox_y + bbox_h) * height) + padding)
        
        # If the crop still collapses, clamp to a 1px minimum so it stays within bounds
        if x1 <= x0:
            x1 = min(width, x0 + 1)
        if y1 <= y0:
            y1 = min(height, y0 + 1)
        
        # Crop the image to the bounding box with padding
        cropped_img = img.crop((x0, y0, x1, y1))
        
        # Save to output directory
        output_path = os.path.join(OUTPUT_DIR, filename)
        cropped_img.save(output_path)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} images")
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"\nComplete! Saved {len(df)} cropped images to {OUTPUT_DIR}")
