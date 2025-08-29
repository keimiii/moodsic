#!/usr/bin/env python3
"""
Create train/validation/test splits from processed FindingEmo annotations.
Generates the CSV files needed for the DINOv3 regression notebook.

Splitting strategy
------------------
- 70/15/15 split with label-aware stratification using 2D bins over
  FindingEmo valence/arousal (V∈[-3,3], A∈[0,6]).
- Uses a fallback mechanism that decreases the number of bins if any
  resulting stratum has fewer than 2 samples (required by sklearn's
  stratify= argument).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# Configuration
BINS_PER_AXIS = 10  # starting number of bins per axis; auto-decreases if sparse
RANDOM_STATE = 42


def _make_strata(df: pd.DataFrame, bins_per_axis: int) -> tuple[pd.Series, int]:
    """Compute 2D-binned strata labels for V/A with a sparsity fallback.

    Returns a tuple of (strata_labels, used_bins_per_axis). The function will
    decrease the number of bins until each stratum in `df` has at least 2
    samples, or fall back to a single bucket.
    """
    v = df["valence"].clip(-3, 3)
    a = df["arousal"].clip(0, 6)

    for b in range(bins_per_axis, 1, -1):
        v_bins = pd.cut(
            v, bins=np.linspace(-3, 3, b + 1), labels=False, include_lowest=True
        )
        a_bins = pd.cut(
            a, bins=np.linspace(0, 6, b + 1), labels=False, include_lowest=True
        )
        strata = (v_bins.astype(int) * b + a_bins.astype(int)).astype(int)
        counts = strata.value_counts()
        if (counts >= 2).all():
            return strata, b

    # Fallback: single bucket for all samples
    return pd.Series(0, index=df.index, dtype=int), 1


def _bin_va(df: pd.DataFrame, bins_per_axis: int) -> tuple[pd.Series, pd.Series]:
    """Return (v_bins, a_bins) in [0, bins_per_axis-1] for FE-unit V/A."""
    v = df["valence"].clip(-3, 3)
    a = df["arousal"].clip(0, 6)
    v_bins = pd.cut(
        v, bins=np.linspace(-3, 3, bins_per_axis + 1), labels=False, include_lowest=True
    ).astype(int)
    a_bins = pd.cut(
        a, bins=np.linspace(0, 6, bins_per_axis + 1), labels=False, include_lowest=True
    ).astype(int)
    return v_bins, a_bins


def _print_strata_report(df: pd.DataFrame, bins_per_axis: int, title: str):
    """Print a compact distribution report for visual inspection in logs."""
    v_bins, a_bins = _bin_va(df, bins_per_axis)
    strata = (v_bins * bins_per_axis + a_bins).astype(int)
    counts = strata.value_counts().reindex(range(bins_per_axis * bins_per_axis), fill_value=0)

    print(f"\n[{title}] bins_per_axis={bins_per_axis}")
    print(f"  Total samples: {len(df)} | Non-empty strata: {(counts>0).sum()} / {bins_per_axis*bins_per_axis}")
    print(
        "  Per-stratum counts => min/median/mean/max: "
        f"{counts.min()} / {int(counts.median())} / {counts.mean():.2f} / {counts.max()}"
    )
    # 2D crosstab for quick visual scan
    ct = pd.crosstab(v_bins, a_bins, rownames=["v_bin"], colnames=["a_bin"], dropna=False)
    print("  Crosstab (rows=v_bin [-3..3], cols=a_bin [0..6]):")
    print(ct.to_string())

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
    
    # Label-aware stratification with fallback binning
    # 70% train, 15% val, 15% test
    X = df_clean[["image_path"]]
    y = df_clean[["valence", "arousal"]]

    strat_labels, used_bins = _make_strata(df_clean, BINS_PER_AXIS)
    print(f"Stratification bins per axis used (stage 1): {used_bins}")
    _print_strata_report(df_clean, used_bins, title="ALL")

    X_train, X_temp, y_train, y_temp, strat_train, strat_temp = train_test_split(
        X, y, strat_labels, test_size=0.3, random_state=RANDOM_STATE, stratify=strat_labels
    )

    # Recompute strata on the temporary set (may need fewer bins due to smaller size)
    temp_df = pd.concat([X_temp.reset_index(drop=True), y_temp.reset_index(drop=True)], axis=1)
    strat_temp_labels, used_bins_temp = _make_strata(temp_df[["valence", "arousal"]], used_bins)
    print(f"Stratification bins per axis used (stage 2): {used_bins_temp}")

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=strat_temp_labels
    )
    
    # Combine X and y back into dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)  
    test_df = pd.concat([X_test, y_test], axis=1)

    # Print per-split distribution reports
    _print_strata_report(train_df, used_bins, title="TRAIN")
    _print_strata_report(val_df, used_bins_temp, title="VAL")
    _print_strata_report(test_df, used_bins_temp, title="TEST")
    
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
