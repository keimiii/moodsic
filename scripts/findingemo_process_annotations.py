#!/usr/bin/env python3

import pandas as pd
from findingemo_light.data.read_annotations import read_annotations
from pathlib import Path
import json

def load_annotations():
    """Load and parse the FindingEmo annotations"""
    raw_data = read_annotations()
    
    # Parse as CSV
    from io import StringIO
    df = pd.read_csv(StringIO(raw_data))
    
    return df

def filter_annotations_for_downloaded_data(df, data_dir="data"):
    """Filter annotations to only include images we have downloaded"""
    data_path = Path(data_dir)
    existing_images = []
    
    # Find all downloaded images
    for run_dir in ["Run_1", "Run_2"]:
        run_path = data_path / run_dir
        if run_path.exists():
            for img_file in run_path.rglob("*.jpg"):
                # Convert to relative path format used in annotations
                rel_path = "/" + str(img_file.relative_to(data_path))
                existing_images.append(rel_path)
    
    print(f"Found {len(existing_images)} downloaded images")
    
    # Filter dataframe to only include existing images
    filtered_df = df[df['image_path'].isin(existing_images)]
    
    print(f"Filtered annotations: {len(filtered_df)} out of {len(df)} total annotations")
    
    return filtered_df

def analyze_annotations(df):
    """Analyze the emotion annotations"""
    print("\n=== ANNOTATION ANALYSIS ===")
    print(f"Total annotations: {len(df)}")
    print(f"Unique images: {df['image_path'].nunique()}")
    print(f"Unique users: {df['user'].nunique()}")
    
    print("\n--- Emotion Distribution ---")
    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)
    
    print("\n--- Valence Distribution ---")
    valence_counts = df['valence'].value_counts()
    print(valence_counts)
    
    print("\n--- Arousal Distribution ---")
    arousal_counts = df['arousal'].value_counts()
    print(arousal_counts)
    
    print("\n--- Age Group Distribution ---")
    age_counts = df['age'].value_counts()
    print(age_counts)
    
    return {
        'emotion_counts': emotion_counts.to_dict(),
        'valence_counts': valence_counts.to_dict(), 
        'arousal_counts': arousal_counts.to_dict(),
        'age_counts': age_counts.to_dict()
    }

def save_processed_annotations(df, output_file="data/processed_annotations.csv"):
    """Save the filtered and processed annotations"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed annotations to: {output_path}")

def main():
    print("Loading FindingEmo annotations...")
    df = load_annotations()
    
    print(f"Loaded {len(df)} total annotations")
    print(f"Columns: {list(df.columns)}")
    
    # Filter for downloaded data
    filtered_df = filter_annotations_for_downloaded_data(df)
    
    # Analyze the data
    stats = analyze_annotations(filtered_df)
    
    # Save processed annotations
    save_processed_annotations(filtered_df)
    
    # Save statistics
    with open("data/annotation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n=== Sample annotations ===")
    print(filtered_df.head())

if __name__ == "__main__":
    main()
