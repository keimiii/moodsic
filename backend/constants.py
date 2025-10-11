from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLUSTERS_CSV_PATH = BASE_DIR / "notebooks/Dataset - DEAM/artifacts/deam_gmm/deam_with_clusters.csv"
CLUSTER_METADATA = {
    0: {
        "name": "Cluster 0 - High Valence, High Arousal",
        "mood": "Party-starting joy, confetti energy.",
        "traits": ["fast tempo", "big drops", "bright majors", "loud and punchy"],
    },
    1: {
        "name": "Cluster 1 - Low Valence, Low Arousal",
        "mood": "Rainy-window melancholy and gentle sighs.",
        "traits": ["slow tempo", "minor keys", "sparse textures", "soft dynamics"],
    },
    2: {
        "name": "Cluster 2 - Moderately High Valence, Moderate Arousal",
        "mood": "Feel-good groove, smiles without the sweat.",
        "traits": ["steady beat", "warm chords", "catchy hooks", "relaxed lift"],
    },
    3: {
        "name": "Cluster 3 - Slightly Negative Valence, Neutral Arousal",
        "mood": "Moody focus with a thoughtful edge.",
        "traits": ["modal or minor", "mid tempo", "restrained energy", "atmospheric layers"],
    },
    4: {
        "name": "Cluster 4 - Slightly Positive Valence, Low Arousal",
        "mood": "Sunny chill and hammock vibes.",
        "traits": ["slow-mid tempo", "soft drums", "warm harmonies", "relaxed feel"],
    },
}