#!/usr/bin/env python3
"""
Download clean FindingEmo dataset from Google Drive.

Downloads Run_1_clean, Run_2_clean folders and processed_annotations.csv
to the data/ directory in the project root.
"""

import os
import subprocess
import sys
from pathlib import Path

# Google Drive folder URL
GDRIVE_URL = "https://drive.google.com/drive/folders/1dM6hg8gTIEPXL3B6EtvA16kpMkcljPld"

# Project root directory (parent of scripts folder)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Items to download
DOWNLOAD_ITEMS = [
    "Run_1_clean",
    "Run_2_clean", 
    "processed_annotations.csv"
]

def check_gdown_installed():
    """Check if gdown is installed, install if not."""
    try:
        import gdown
        print("✓ gdown is already installed")
        return True
    except ImportError:
        print("Installing gdown...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], 
                         check=True, capture_output=True)
            print("✓ gdown installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install gdown: {e}")
            return False

def create_data_directory():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    print(f"✓ Data directory ready: {DATA_DIR}")

def download_folder():
    """Download the entire Google Drive folder."""
    print(f"Downloading from: {GDRIVE_URL}")
    print(f"Saving to: {DATA_DIR}")
    
    try:
        subprocess.run([
            "gdown", 
            GDRIVE_URL,
            "-O", str(DATA_DIR),
            "--folder"
        ], check=True)
        print("✓ Download completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the Google Drive folder is shared with 'Anyone with the link'")
        print("2. Try downloading manually from the browser and extract to data/ folder")
        print("3. If the folder is too large, try downloading individual items:")
        print(f"   gdown --folder https://drive.google.com/drive/folders/SUBFOLDER_ID -O {DATA_DIR}")
        print("\nAlternative: Download manually and place in data/ folder:")
        for item in DOWNLOAD_ITEMS:
            print(f"   - {item}")
        return False

def cleanup_unwanted_files():
    """Remove Original Files folder if it was downloaded."""
    original_files_path = DATA_DIR / "Original Files"
    if original_files_path.exists():
        import shutil
        shutil.rmtree(original_files_path)
        print("✓ Removed 'Original Files' folder")

def verify_downloads():
    """Verify that expected items were downloaded."""
    print("\nVerifying downloads:")
    all_good = True
    
    for item in DOWNLOAD_ITEMS:
        item_path = DATA_DIR / item
        if item_path.exists():
            if item_path.is_dir():
                file_count = len(list(item_path.glob("**/*")))
                print(f"✓ {item} (folder with {file_count} files)")
            else:
                size = item_path.stat().st_size
                print(f"✓ {item} ({size} bytes)")
        else:
            print(f"✗ {item} - NOT FOUND")
            all_good = False
    
    return all_good

def main():
    """Main download function."""
    print("FindingEmo Clean Dataset Downloader")
    print("=" * 40)
    
    # Check dependencies
    if not check_gdown_installed():
        return 1
    
    # Setup
    create_data_directory()
    
    # Download
    if not download_folder():
        return 1
    
    # Cleanup
    cleanup_unwanted_files()
    
    # Verify
    if verify_downloads():
        print("\n✓ All files downloaded successfully!")
        return 0
    else:
        print("\n⚠ Some files may be missing. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
