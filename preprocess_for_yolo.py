"""
Preprocess cropped images for YOLO11n by padding to consistent square size.
Preserves aspect ratio with black padding.
"""

import os
from PIL import Image
import pandas as pd

# Configuration
INPUT_DIR = "Snapshot_WI-Oh_Deer_Photos_Cropped"
OUTPUT_DIR = "Snapshot_WI-Oh_Deer_Photos_YOLO"
TARGET_SIZE = 640  # YOLO standard input size

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all images
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Processing {len(image_files)} images...")

# Find max dimensions
max_width = 0
max_height = 0

for filename in image_files:
    try:
        img = Image.open(os.path.join(INPUT_DIR, filename))
        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)
    except Exception as e:
        print(f"Error reading {filename}: {e}")

print(f"Max dimensions found: {max_width}x{max_height}")

# Process each image
for idx, filename in enumerate(image_files, 1):
    try:
        img = Image.open(os.path.join(INPUT_DIR, filename))
        
        # Calculate square size (max of width/height)
        square_size = max(img.width, img.height)
        
        # Create square canvas with black background
        square_img = Image.new('RGB', (square_size, square_size), (0, 0, 0))
        
        # Calculate position to center the image
        x_offset = (square_size - img.width) // 2
        y_offset = (square_size - img.height) // 2
        
        # Paste image onto square canvas
        square_img.paste(img, (x_offset, y_offset))
        
        # Resize to target YOLO size
        square_img = square_img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, filename)
        square_img.save(output_path)
        
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_files)} images")
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"\nComplete! Saved {len(image_files)} preprocessed images to {OUTPUT_DIR}")
print(f"All images are now {TARGET_SIZE}x{TARGET_SIZE} with aspect ratio preserved")
