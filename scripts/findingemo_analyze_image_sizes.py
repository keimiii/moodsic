#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict


def get_image_file_sizes(data_dir="data"):
    """Get file sizes for all images in the data directory"""
    data_path = Path(data_dir)
    image_sizes = []
    corrupted_candidates = []

    print("Scanning images...")

    for run_dir in ["Run_1", "Run_2"]:
        run_path = data_path / run_dir
        if not run_path.exists():
            print(f"Warning: {run_path} does not exist")
            continue

        for img_file in run_path.rglob("*.jpg"):
            try:
                file_size = img_file.stat().st_size
                image_sizes.append(
                    {
                        "path": str(img_file),
                        "size_bytes": file_size,
                        "size_kb": file_size / 1024,
                        "size_mb": file_size / (1024 * 1024),
                    }
                )

                # Flag potentially corrupted files (very small)
                if file_size < 1000:  # Less than 1KB
                    corrupted_candidates.append(
                        {"path": str(img_file), "size_bytes": file_size}
                    )

            except OSError as e:
                print(f"Error accessing {img_file}: {e}")

    return image_sizes, corrupted_candidates


def compute_statistics(image_sizes):
    """Compute detailed statistics on image file sizes"""
    if not image_sizes:
        return None

    sizes_bytes = [img["size_bytes"] for img in image_sizes]
    sizes_kb = [img["size_kb"] for img in image_sizes]
    sizes_mb = [img["size_mb"] for img in image_sizes]

    stats = {
        "count": len(sizes_bytes),
        "bytes": {
            "mean": np.mean(sizes_bytes),
            "median": np.median(sizes_bytes),
            "std": np.std(sizes_bytes),
            "min": np.min(sizes_bytes),
            "max": np.max(sizes_bytes),
            "q25": np.percentile(sizes_bytes, 25),
            "q75": np.percentile(sizes_bytes, 75),
            "q1": np.percentile(sizes_bytes, 1),
            "q5": np.percentile(sizes_bytes, 5),
            "q95": np.percentile(sizes_bytes, 95),
            "q99": np.percentile(sizes_bytes, 99),
        },
        "kb": {
            "mean": np.mean(sizes_kb),
            "median": np.median(sizes_kb),
            "std": np.std(sizes_kb),
            "min": np.min(sizes_kb),
            "max": np.max(sizes_kb),
            "q25": np.percentile(sizes_kb, 25),
            "q75": np.percentile(sizes_kb, 75),
        },
        "mb": {
            "mean": np.mean(sizes_mb),
            "median": np.median(sizes_mb),
            "std": np.std(sizes_mb),
            "min": np.min(sizes_mb),
            "max": np.max(sizes_mb),
        },
    }

    return stats


def analyze_size_distribution(image_sizes):
    """Analyze the distribution of file sizes to identify outliers"""
    sizes_kb = [img["size_kb"] for img in image_sizes]

    # Calculate various thresholds
    q1 = np.percentile(sizes_kb, 1)
    q5 = np.percentile(sizes_kb, 5)
    q25 = np.percentile(sizes_kb, 25)
    q75 = np.percentile(sizes_kb, 75)
    iqr = q75 - q25

    # Standard outlier detection methods
    outlier_thresholds = {
        "iqr_1.5": q25 - 1.5 * iqr,  # Standard IQR method
        "iqr_3.0": q25 - 3.0 * iqr,  # Conservative IQR method
        "q1_percentile": q1,
        "q5_percentile": q5,
        "absolute_1kb": 1.0,  # Less than 1KB
        "absolute_5kb": 5.0,  # Less than 5KB
        "absolute_10kb": 10.0,  # Less than 10KB
    }

    # Count images below each threshold
    threshold_counts = {}
    for name, threshold in outlier_thresholds.items():
        count = sum(1 for size in sizes_kb if size < threshold)
        threshold_counts[name] = {
            "threshold_kb": threshold,
            "count_below": count,
            "percentage": (count / len(sizes_kb)) * 100,
        }

    return threshold_counts


def create_visualizations(image_sizes, output_dir="data"):
    """Create visualizations of the file size distribution"""
    sizes_kb = [img["size_kb"] for img in image_sizes]

    # Create histogram
    plt.figure(figsize=(12, 8))

    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(sizes_kb, bins=1000, alpha=0.7, edgecolor="black")
    plt.xlabel("File Size (KB)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Image File Sizes")
    plt.xlim(0, 1000)
    plt.xticks(range(0, 1001, 100))
    plt.grid(True, alpha=0.3)

    # Log scale histogram
    plt.subplot(2, 2, 2)
    plt.hist(sizes_kb, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("File Size (KB)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Image File Sizes (Log Scale)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    # Box plot
    plt.subplot(2, 2, 3)
    plt.boxplot(sizes_kb)
    plt.ylabel("File Size (KB)")
    plt.title("Box Plot of Image File Sizes")
    plt.grid(True, alpha=0.3)

    # Focus on small files
    plt.subplot(2, 2, 4)
    small_sizes = [size for size in sizes_kb if size < 50]  # Files under 50KB
    if small_sizes:
        plt.hist(small_sizes, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("File Size (KB)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Small Files (<50KB)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/image_size_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_analysis_report(stats, threshold_counts, corrupted_candidates):
    """Print a comprehensive analysis report"""
    print("\n" + "=" * 60)
    print("IMAGE FILE SIZE ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nTotal images analyzed: {stats['count']:,}")

    print(f"\n--- SIZE STATISTICS (KB) ---")
    print(f"Mean:     {stats['kb']['mean']:.2f} KB")
    print(f"Median:   {stats['kb']['median']:.2f} KB")
    print(f"Std Dev:  {stats['kb']['std']:.2f} KB")
    print(f"Min:      {stats['kb']['min']:.2f} KB")
    print(f"Max:      {stats['kb']['max']:.2f} KB")
    print(f"Q25:      {stats['kb']['q25']:.2f} KB")
    print(f"Q75:      {stats['kb']['q75']:.2f} KB")

    print(f"\n--- PERCENTILE ANALYSIS ---")
    print(
        f"1st percentile:  {stats['bytes']['q1']:.0f} bytes ({stats['bytes']['q1'] / 1024:.2f} KB)"
    )
    print(
        f"5th percentile:  {stats['bytes']['q5']:.0f} bytes ({stats['bytes']['q5'] / 1024:.2f} KB)"
    )
    print(
        f"95th percentile: {stats['bytes']['q95']:.0f} bytes ({stats['bytes']['q95'] / 1024:.2f} KB)"
    )
    print(
        f"99th percentile: {stats['bytes']['q99']:.0f} bytes ({stats['bytes']['q99'] / 1024:.2f} KB)"
    )

    print(f"\n--- CORRUPTION DETECTION THRESHOLDS ---")
    for name, data in threshold_counts.items():
        print(
            f"{name:15s}: <{data['threshold_kb']:6.1f} KB -> {data['count_below']:4d} images ({data['percentage']:5.2f}%)"
        )

    if corrupted_candidates:
        print(f"\n--- POTENTIALLY CORRUPTED FILES (<1KB) ---")
        print(f"Found {len(corrupted_candidates)} files smaller than 1KB:")
        for candidate in corrupted_candidates[:10]:  # Show first 10
            print(f"  {candidate['size_bytes']:4d} bytes: {candidate['path']}")
        if len(corrupted_candidates) > 10:
            print(f"  ... and {len(corrupted_candidates) - 10} more")

    print(f"\n--- RECOMMENDED CORRUPTION THRESHOLD ---")
    # Suggest threshold based on analysis
    q1_threshold = stats["bytes"]["q1"] / 1024
    if q1_threshold < 10:
        recommended = 10.0
        reason = "1st percentile is very low, using conservative 10KB threshold"
    elif q1_threshold < 20:
        recommended = q1_threshold
        reason = f"Using 1st percentile ({q1_threshold:.1f} KB) as threshold"
    else:
        recommended = 20.0
        reason = "1st percentile seems reasonable, using conservative 20KB threshold"

    print(f"Recommended threshold: {recommended:.1f} KB")
    print(f"Reason: {reason}")

    threshold_key = f"absolute_{int(recommended)}kb"
    if threshold_key in threshold_counts:
        affected_count = threshold_counts[threshold_key]["count_below"]
        print(f"This would flag {affected_count} images as potentially corrupted")
    else:
        print("Threshold analysis not available for recommended value")


def main():
    print("Analyzing image file sizes in the data directory...")

    # Get image file sizes
    image_sizes, corrupted_candidates = get_image_file_sizes()

    if not image_sizes:
        print("No images found!")
        return

    # Compute statistics
    stats = compute_statistics(image_sizes)

    # Analyze distribution and thresholds
    threshold_counts = analyze_size_distribution(image_sizes)

    # Print comprehensive report
    print_analysis_report(stats, threshold_counts, corrupted_candidates)

    # Save results
    results = {
        "statistics": stats,
        "threshold_analysis": threshold_counts,
        "corrupted_candidates": corrupted_candidates,
        "total_images": len(image_sizes),
    }

    with open("data/image_size_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: data/image_size_analysis.json")

    # Create visualizations
    try:
        create_visualizations(image_sizes)
        print("Visualizations saved to: data/image_size_analysis.png")
    except ImportError:
        print("matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    main()
