from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
VIDEO_ASSETS_DIR = DATA_DIR / "veatic" / "shortlisted_videos"
DEAM_AUDIO_DIR = DATA_DIR / "deam" / "MEMD_audio"

RESULTS_DIR = DATA_DIR / "results"
INFERENCE_RESULTS_DIR = RESULTS_DIR / "inference"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
DEAM_GMM_ARTIFACTS_DIR = ARTIFACTS_DIR / "deam_gmm"

CLUSTERS_CSV_PATH = DEAM_GMM_ARTIFACTS_DIR / "deam_with_clusters.csv"
PIPELINE_RESULTS_PATH = INFERENCE_RESULTS_DIR / "pipeline_results_20251006_144126.parquet"
PIPELINE_RESULTS_ENRICHED_PATH = INFERENCE_RESULTS_DIR / "pipeline_results_20251006_144126_enriched.parquet"
VEATIC_PER_VIDEO_PATH = EVALUATION_RESULTS_DIR / "veatic_per_video_20251006_144126.csv"
DEAM_CLUSTERED_CATALOG_PATH = DEAM_GMM_ARTIFACTS_DIR / "deam_gmm_clusters.csv"

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
