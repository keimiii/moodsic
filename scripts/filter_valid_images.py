#!/usr/bin/env python3
"""
Filter CSV files to only include valid image files.
This script removes entries where the image file is corrupted or doesn't exist.
"""

import pandas as pd
from PIL import Image
from pathlib import Path
import os

def is_valid_image(image_path: str) -> bool:
    """Check if an image file is valid and can be opened."""
    try:
        # Convert relative path to absolute from project root
        if image_path.startswith('../'):
            image_path = image_path[3:]  # Remove '../'
        
        full_path = Path('/Users/desmondchoy/Projects/emo-rec') / image_path
        
        if not full_path.exists():
            return False
            
        # Check file size (skip very small files that are likely corrupted)
        if full_path.stat().st_size < 1024:  # Less than 1KB
            return False
            
        # Try to open with PIL and actually load the data
        with Image.open(full_path) as img:
            img.convert('RGB')  # This loads the data and converts
            # Check minimum size
            if img.size[0] < 32 or img.size[1] < 32:
                return False
            return True
    except Exception as e:
        return False

def filter_csv(input_path: str, output_path: str):
    """Filter a CSV file to only include valid images."""
    df = pd.read_csv(input_path)
    print(f"Original {input_path}: {len(df)} entries")
    
    # Filter valid images with progress tracking
    valid_images = []
    for i, image_path in enumerate(df['image_path']):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(df)} images...")
        valid_images.append(is_valid_image(image_path))
    
    df_filtered = df[valid_images].reset_index(drop=True)
    
    print(f"Filtered {output_path}: {len(df_filtered)} entries ({len(df) - len(df_filtered)} removed)")
    
    # Save filtered CSV
    df_filtered.to_csv(output_path, index=False)
    
    return df_filtered

if __name__ == "__main__":
    # Filter all CSV files
    base_path = "/Users/desmondchoy/Projects/emo-rec/data"
    
    filter_csv(f"{base_path}/train.csv", f"{base_path}/train_clean.csv")
    filter_csv(f"{base_path}/valid.csv", f"{base_path}/valid_clean.csv") 
    filter_csv(f"{base_path}/test.csv", f"{base_path}/test_clean.csv")
