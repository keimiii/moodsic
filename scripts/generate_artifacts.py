#!/usr/bin/env python3
"""
Generate artifacts from DEAM clustering notebook for the Flask backend
"""

import os
import sys
import numpy as np
import polars as pl
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_deam_data():
    """Load and process DEAM data similar to the notebook"""
    
    # Load annotations
    anno_1 = pl.read_csv("data/DEAM/static_annotations_averaged_songs_1_2000.csv")
    anno_1 = anno_1.rename({col: col.strip() for col in anno_1.columns})

    anno_2 = pl.read_csv("data/DEAM/static_annotations_averaged_songs_2000_2058.csv")
    anno_2 = anno_2.rename({col: col.strip() for col in anno_2.columns})

    all_annotations = pl.concat(
        [
            anno_1,
            anno_2.select(
                "song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"
            ),
        ],
        how="vertical",
    )

    # Load metadata
    metadata_2013 = pl.read_csv("data/DEAM/metadata_2013.csv")
    metadata_2014 = pl.read_csv("data/DEAM/metadata_2014.csv", truncate_ragged_lines=True)
    metadata_2015 = pl.read_csv("data/DEAM/metadata_2015.csv", truncate_ragged_lines=True)

    # Clean metadata
    metadata_2013 = (
        metadata_2013.select("song_id", "Artist", "Song title", "Genre")
        .rename({"Artist": "artist", "Song title": "song_title", "Genre": "genre"})
        .with_columns(
            pl.col(pl.Utf8)
            .str.replace_all("\u00a0", " ")
            .str.strip_chars()
            .str.strip_chars('"')
        )
    )

    metadata_2014 = (
        metadata_2014.select("Id", "Artist", "Track", "Genre")
        .rename(
            {
                "Id": "song_id",
                "Artist": "artist",
                "Track": "song_title",
                "Genre": "genre",
            }
        )
        .with_columns(
            pl.col(pl.Utf8)
            .str.replace_all("\u00a0", " ")
            .str.strip_chars()
            .str.strip_chars('"')
        )
    )

    metadata_2015 = (
        metadata_2015.select("id", "artist", "title", "genre")
        .rename(
            {
                "id": "song_id",
                "title": "song_title",
            }
        )
        .with_columns(
            pl.col(pl.Utf8)
            .str.replace_all("\u00a0", " ")
            .str.strip_chars()
            .str.strip_chars('"')
        )
    )

    all_metadata = pl.concat(
        [metadata_2013, metadata_2014, metadata_2015],
        how="vertical",
    )

    # Join annotations and metadata
    annot_meta = all_annotations.join(all_metadata, on="song_id", how="left").with_columns(
        pl.col("genre").str.to_lowercase().alias("genre")
    )

    # Simple genre classification (simplified from notebook)
    def classify_genre(raw):
        if not raw or raw.lower() in ['n/a', 'experimental', 'instrumental']:
            return "other"
        
        raw_lower = raw.lower()
        if any(word in raw_lower for word in ['rock', 'metal', 'punk']):
            return "rock"
        elif any(word in raw_lower for word in ['pop', 'dance']):
            return "pop"
        elif any(word in raw_lower for word in ['hip', 'rap']):
            return "hip-hop"
        elif any(word in raw_lower for word in ['electronic', 'techno', 'house']):
            return "electronic"
        elif any(word in raw_lower for word in ['jazz', 'blues', 'soul']):
            return "jazz"
        elif any(word in raw_lower for word in ['classical', 'orchestral']):
            return "classical"
        elif any(word in raw_lower for word in ['folk', 'country', 'acoustic']):
            return "folk/country"
        else:
            return "other"

    # Apply genre classification
    merged_genres = annot_meta.with_columns(
        pl.col("genre").map_elements(classify_genre, return_dtype=pl.String).alias("genre")
    )

    # Scale alignment (convert from 1-9 scale to -1 to 1)
    merged_genres = merged_genres.with_columns(
        ((pl.col("valence_mean") - 5.0) / 4.0).clip(-1.0, 1.0).alias("valence_ref"),
        ((pl.col("arousal_mean") - 5.0) / 4.0).clip(-1.0, 1.0).alias("arousal_ref"),
    )

    return merged_genres

def train_gmm_model(data):
    """Train GMM model on the data"""
    
    # Prepare features
    df_pl = data.select(["valence_ref", "arousal_ref", "genre"]).drop_nulls()
    X = df_pl.select(["valence_ref", "arousal_ref"]).to_numpy()
    
    # Scale features
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # Train GMM (using same parameters as notebook)
    gmm = GaussianMixture(
        n_components=5, 
        covariance_type="diag", 
        n_init=10, 
        random_state=2025
    ).fit(X_scaled)
    
    # Get cluster assignments
    clusters = gmm.predict(X_scaled)
    cluster_conf = gmm.predict_proba(X_scaled).max(axis=1)
    
    # Add cluster info to data
    data_with_clusters = data.select(
        "song_id", "valence_ref", "arousal_ref", "artist", "song_title", "genre"
    ).with_columns(
        [
            pl.Series("cluster", clusters),
            pl.Series("cluster_conf", cluster_conf),
        ]
    )
    
    return scaler, gmm, data_with_clusters

def main():
    """Main function to generate all artifacts"""
    
    print("Loading DEAM data...")
    data = load_deam_data()
    print(f"Loaded {data.height} songs")
    
    print("Training GMM model...")
    scaler, gmm, songs_with_clusters = train_gmm_model(data)
    print(f"Trained GMM with {gmm.n_components} components")
    
    # Create artifacts directory
    artifacts_dir = Path("notebooks/deam/artifacts/deam_gmm")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save artifacts
    print("Saving artifacts...")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(gmm, artifacts_dir / "gmm.pkl")
    songs_with_clusters.write_parquet(artifacts_dir / "deam_songs.parquet")
    
    print(f"Artifacts saved to {artifacts_dir}")
    print(f"Scaler: {artifacts_dir / 'scaler.pkl'}")
    print(f"GMM: {artifacts_dir / 'gmm.pkl'}")
    print(f"Songs: {artifacts_dir / 'deam_songs.parquet'}")
    
    # Print cluster summary
    print("\nCluster summary:")
    for cluster_id in range(gmm.n_components):
        cluster_data = songs_with_clusters.filter(pl.col("cluster") == cluster_id)
        if cluster_data.height > 0:
            center_valence = cluster_data["valence_ref"].mean()
            center_arousal = cluster_data["arousal_ref"].mean()
            print(f"Cluster {cluster_id}: {cluster_data.height} songs, center=({center_valence:.2f}, {center_arousal:.2f})")

if __name__ == "__main__":
    main()
