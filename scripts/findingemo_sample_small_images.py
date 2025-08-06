#!/usr/bin/env python3

import os
import random
from pathlib import Path
import json

def get_small_images(threshold_kb=10):
    """Get all images smaller than the threshold"""
    data_path = Path("data")
    small_images = []
    
    print(f"Finding images smaller than {threshold_kb}KB...")
    
    for run_dir in ["Run_1", "Run_2"]:
        run_path = data_path / run_dir
        if not run_path.exists():
            continue
            
        for img_file in run_path.rglob("*.jpg"):
            try:
                file_size_kb = img_file.stat().st_size / 1024
                if file_size_kb <= threshold_kb:
                    small_images.append({
                        'path': str(img_file),
                        'size_bytes': img_file.stat().st_size,
                        'size_kb': file_size_kb
                    })
            except OSError as e:
                print(f"Error accessing {img_file}: {e}")
    
    return small_images

def sample_images(images, sample_size=15):
    """Randomly sample images from the list"""
    if len(images) <= sample_size:
        return images
    
    return random.sample(images, sample_size)

def display_sample(sample_images):
    """Display the sampled images for manual verification"""
    print(f"\n{'='*80}")
    print(f"RANDOM SAMPLE OF {len(sample_images)} SMALL IMAGES FOR MANUAL VERIFICATION")
    print(f"{'='*80}")
    
    for i, img in enumerate(sample_images, 1):
        print(f"\n{i:2d}. {img['path']}")
        print(f"    Size: {img['size_bytes']:,} bytes ({img['size_kb']:.2f} KB)")
        
        # Try to get some basic info about the file
        path = Path(img['path'])
        if path.exists():
            try:
                stat = path.stat()
                print(f"    Modified: {stat.st_mtime}")
                
                # Check if file is readable
                with open(path, 'rb') as f:
                    first_bytes = f.read(10)
                    if first_bytes:
                        hex_bytes = ' '.join(f'{b:02x}' for b in first_bytes)
                        print(f"    First bytes: {hex_bytes}")
                        
                        # Check for JPEG header
                        if first_bytes[:2] == b'\xff\xd8':
                            print(f"    ✓ Has JPEG header")
                        else:
                            print(f"    ✗ Missing JPEG header")
                    else:
                        print(f"    ✗ File is empty")
                        
            except Exception as e:
                print(f"    ✗ Error reading file: {e}")
        else:
            print(f"    ✗ File does not exist")
    
    print(f"\n{'='*80}")
    print("MANUAL VERIFICATION INSTRUCTIONS:")
    print("1. Try opening each file in an image viewer")
    print("2. Note which files fail to open or display corrupted content")
    print("3. Files with 0 bytes or missing JPEG headers are likely corrupted")
    print("4. Very small files (<1KB) are almost certainly corrupted")
    print(f"{'='*80}")

def categorize_sample(sample_images):
    """Categorize the sample by size and potential corruption indicators"""
    categories = {
        'zero_bytes': [],
        'very_small': [],  # <1KB
        'small': [],       # 1-5KB
        'medium_small': [] # 5-10KB
    }
    
    for img in sample_images:
        if img['size_bytes'] == 0:
            categories['zero_bytes'].append(img)
        elif img['size_kb'] < 1:
            categories['very_small'].append(img)
        elif img['size_kb'] < 5:
            categories['small'].append(img)
        else:
            categories['medium_small'].append(img)
    
    print(f"\n--- SAMPLE CATEGORIZATION ---")
    for category, images in categories.items():
        if images:
            print(f"{category.replace('_', ' ').title()}: {len(images)} files")
            for img in images:
                print(f"  {img['size_bytes']:4d} bytes - {img['path']}")

def main():
    # Set random seed for reproducible results
    random.seed(42)
    
    print("Sampling small images for manual verification...")
    
    # Get all images ≤10KB
    small_images = get_small_images(threshold_kb=10)
    
    print(f"Found {len(small_images)} images ≤10KB")
    
    if not small_images:
        print("No small images found!")
        return
    
    # Sample 15 random images
    sample = sample_images(small_images, sample_size=15)
    
    # Sort sample by size for easier review
    sample.sort(key=lambda x: x['size_bytes'])
    
    # Display the sample
    display_sample(sample)
    
    # Categorize the sample
    categorize_sample(sample)
    
    # Save sample for reference
    with open("data/manual_verification_sample.json", "w") as f:
        json.dump(sample, f, indent=2)
    
    print(f"\nSample saved to: data/manual_verification_sample.json")

if __name__ == "__main__":
    main()
