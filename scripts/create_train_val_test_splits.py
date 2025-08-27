#!/usr/bin/env python3
"""
Create train/validation/test splits from processed FindingEmo annotations.
Generates the CSV files needed for the DINOv3 regression notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_splits():
    # Read the processed annotations
    data_root = Path("/Users/desmondchoy/Projects/emo-rec/data")
    annotations_file = data_root / "processed_annotations.csv"
    
    print(f"Reading annotations from: {annotations_file}")
    df = pd.read_csv(annotations_file)
    
    # Convert image_path to relative paths from project root
    df['image_path'] = df['image_path'].apply(lambda x: f"data{x}")
    
    # Keep only the columns needed: image_path, valence, arousal
    df_clean = df[['image_path', 'valence', 'arousal']].copy()
    
    # Remove any rows with missing values
    df_clean = df_clean.dropna()
    
    print(f"Total samples: {len(df_clean)}")
    print(f"Valence range: {df_clean['valence'].min():.2f} to {df_clean['valence'].max():.2f}")
    print(f"Arousal range: {df_clean['arousal'].min():.2f} to {df_clean['arousal'].max():.2f}")
    
    # Create stratified splits based on valence/arousal bins for better distribution
    # Create bins for stratification
    v_bins = pd.cut(df_clean['valence'], bins=5, labels=False)
    a_bins = pd.cut(df_clean['arousal'], bins=5, labels=False)
    strat_labels = v_bins * 5 + a_bins  # Combine into single stratification label
    
    # 70% train, 15% validation, 15% test
    X = df_clean[['image_path']]
    y = df_clean[['valence', 'arousal']]
    
    X_train, X_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        X, y, strat_labels, test_size=0.3, random_state=42, stratify=strat_labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=strat_temp
    )
    
    # Combine X and y back into dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)  
    test_df = pd.concat([X_test, y_test], axis=1)
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Save the splits
    train_csv = data_root / "train.csv"
    val_csv = data_root / "valid.csv"
    test_csv = data_root / "test.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\nSaved splits to:")
    print(f"Train: {train_csv}")
    print(f"Validation: {val_csv}")
    print(f"Test: {test_csv}")
    
    # Show sample from each split
    print(f"\nTrain sample:")
    print(train_df.head(2))
    print(f"\nValidation sample:")
    print(val_df.head(2))
    print(f"\nTest sample:")
    print(test_df.head(2))
    
    return train_csv, val_csv, test_csv

if __name__ == "__main__":
    create_splits()
