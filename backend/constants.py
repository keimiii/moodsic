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


AVAILABLE_VIDEOS = [
    {"id": "0", "name": "Video 0", "filename": "0.mp4"},
    {"id": "4", "name": "Video 4", "filename": "4.mp4"},
    {"id": "12", "name": "Video 12", "filename": "12.mp4"},
    {"id": "34", "name": "Video 34", "filename": "34.mp4"},
    {"id": "40", "name": "Video 40", "filename": "40.mp4"},
    {"id": "42", "name": "Video 42", "filename": "42.mp4"},
    {"id": "44", "name": "Video 44", "filename": "44.mp4"},
    {"id": "49", "name": "Video 49", "filename": "49.mp4"},
    {"id": "53", "name": "Video 53", "filename": "53.mp4"},
    {"id": "55", "name": "Video 55", "filename": "55.mp4"},
    {"id": "60", "name": "Video 60", "filename": "60.mp4"},
    {"id": "79", "name": "Video 79", "filename": "79.mp4"},
    {"id": "81", "name": "Video 81", "filename": "81.mp4"},
    {"id": "86", "name": "Video 86", "filename": "86.mp4"},
    {"id": "95", "name": "Video 95", "filename": "95.mp4"},
    {"id": "96", "name": "Video 96", "filename": "96.mp4"},
    {"id": "102", "name": "Video 102", "filename": "102.mp4"},
    {"id": "114", "name": "Video 114", "filename": "114.mp4"},
    {"id": "120", "name": "Video 120", "filename": "120.mp4"},
]

VIDEO_ID_TO_COMMENTS = {
    "0": "Scene: Two women chatting on the boat, presumably melancholic. Slow acoustic rock music appropriate at times, but not when characters are speaking.",
    "4": "Scene: Man trapped in a cave. Hard Rock music very appropriate to capture tension.",
    "12": "Scene: Intense scene of Cowboys fighting - somewhat violent. Whimsical music not very appropriate.",
    "34": "Scene: A heartfelt scene with a man talking to his dog that's lying on the vet table. Slow acoustic music very appropriate.",
    "40": "Scene: Daughter having a chat with her parents. The electronic tune that gives off a synth-heavy moody, atmospheric vibe doesn't seem appropriate.",
    "42": "Scene: Teacher caught student breaking a rule. The mysterious, bouncy tone works well up to the mood drop when the student approaches the front. The later return to upbeat music, however, disrupts the emotional flow.",
    "44": "Scene: A man dozes off in a chair and wakes up to find the patient bed beside him empty. Alarmed, he gets up and starts searching. Western Cowboy Ballad music about saying goodbye but it does not match the tension of the moment",
    "49": "Scene: uncomfortable tension as the man and woman lie stiffly on the same bed. The music’s bouncy, mellow vibe and romantic lyrics create a “falling in love” feeling, which doesn’t suit the tense atmosphere.",
    "53": "Scene: a man in a dirty suit walking on a dessert. The mysterious music gradually shifts to a tone of enlightenment as he looks around and takes a drink.",
    "55": "Scene: A man and a woman talk in a swimming pool, their conversation charged with romantic tension. The pop/electronic music complements the mood perfectly, matching the subtle chemistry between them.",
    "60": "No comments available.",
    "79": "No comments available.",
    "81": "No comments available.",
    "86": "No comments available.",
    "95": "Scene: Diver is anxious and scared while looking down at the pool he's supposed to dive into, almost jumps in but backs out multiple times due to fear. The jittery pop song suits the general vibe, but is slightly too upbeat.",
    "96": "Scene: A lady is giving a eulogy at someone's funeral, and there scenes of many people's mixed reactions. The rock song suits the chaotic energy of the video.",
    "102": "Scene: A couple is boarding a train, and the couple seems bittersweet about their having to part. The jazzy music suits this video and captures the tense mood.",
    "114": "Scene: A girl is complaining to another girl about something in an angry manner, and the other girl is trying to help her figure things out. This folksy music captures the negative emotions and slight mood arousal well.",
    "120": "Scene: A girl is visiting a guy at his place and there is romantic tension in the air. This electronic song is not appropriate.",
}

VIDEO_ID_TO_MATCH_RESULTS = {
    "0": "Partially",
    "4": "Yes",
    "12": "No",
    "34": "Yes",
    "40": "No",
    "42": "Partially",
    "44": "No",
    "49": "No",
    "53": "Yes",
    "55": "Yes",
    "60": "-",
    "79": "-",
    "81": "-",
    "86": "-",
    "95": "Partially",
    "96": "Yes",
    "102": "Yes",
    "114": "Yes",
    "120": "No",
}