import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import tempfile
from pathlib import Path
import polars as pl

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# For now, we'll use static data instead of loading models
# In a real implementation, you would load the trained models here
scaler, gmm, songs_df = None, None, None

BASE_DIR = Path(__file__).resolve().parent.parent
CLUSTERS_CSV_PATH = BASE_DIR / "notebooks/deam/artifacts/deam_gmm/deam_with_clusters.csv"
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

def process_video_for_emotion(video_id):
    """
    Process video to extract emotion features
    This is a placeholder - in a real implementation, you would:
    1. Load video from data/VEATIC/videos/{video_id}.mp4
    2. Extract frames from video
    3. Run face detection and emotion recognition
    4. Run scene analysis
    5. Fuse the results
    """
    # For now, return different static values based on video_id
    # In a real implementation, this would use your trained models
    video_scenes = {
        "4": {"valence": 0.37, "arousal": 0.34, "cluster_id": 0},
        "44": {"valence": -0.33, "arousal": -0.44, "cluster_id": 1},
        "60": {"valence": 0.14, "arousal": 0.16, "cluster_id": 2}
    }
    
    # Return the scene for the specific video
    return video_scenes.get(video_id, {"valence": 0.0, "arousal": 0.0, "cluster_id": 0})

def recommend_song(valence, arousal, cluster_id):
    """
    Recommend a song based on valence, arousal, and cluster
    """
    # Use static mock data for now
    mock_songs = [
        {"song_id": "10", "title": "Sunlit Avenues", "artist": "Ivory Coast", "genre": "electronic"},
        {"song_id": "1000", "title": "Gravity Rush", "artist": "Midnight Circuit", "genre": "rock"},
        {"song_id": "1001", "title": "Midnight Drizzle", "artist": "Slow Parade", "genre": "folk/country"},
        {"song_id": "1002", "title": "Caps & Gowns", "artist": "Riverfolk", "genre": "pop"},
        {"song_id": "1003", "title": "Full Court Press", "artist": "Baseline", "genre": "hip-hop"},
        {"song_id": "1004", "title": "Couch Screams", "artist": "Neon Noir", "genre": "electronic"}
    ]
    return mock_songs[cluster_id % len(mock_songs)]

@app.route('/api/videos')
def get_videos():
    """
    Get list of available videos
    """
    videos = [
        {"id": "4", "name": "Video 4", "filename": "4.mp4"},
        {"id": "44", "name": "Video 44", "filename": "44.mp4"},
        {"id": "60", "name": "Video 60", "filename": "60.mp4"}
    ]
    return jsonify(videos)

@app.route('/api/video/<video_id>')
def get_video(video_id):
    """
    Serve video file for a given video ID
    """
    try:
        video_path = Path(f"../data/VEATIC/videos/{video_id}.mp4")
        if video_path.exists():
            return send_file(str(video_path), as_attachment=False)
        else:
            return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error serving video: {str(e)}'}), 500

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """
    Process selected video and return emotion analysis + song recommendation
    """
    try:
        data = request.get_json()
        if not data or 'video_id' not in data:
            return jsonify({'error': 'No video ID provided'}), 400
        
        video_id = data['video_id']
        
        # Validate video ID
        valid_videos = ["4", "44", "60"]
        if video_id not in valid_videos:
            return jsonify({'error': 'Invalid video ID'}), 400
        
        # Process video for emotion (placeholder implementation)
        emotion_result = process_video_for_emotion(video_id)
        
        # Get song recommendation
        song_recommendation = recommend_song(
            emotion_result["valence"], 
            emotion_result["arousal"], 
            emotion_result["cluster_id"]
        )
        
        # Return results
        result = {
            "video_id": video_id,
            "valence": emotion_result["valence"],
            "arousal": emotion_result["arousal"],
            "cluster_id": emotion_result["cluster_id"],
            "song": song_recommendation
        }
        
        return jsonify(result)
            
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/song/<song_id>')
def get_song(song_id):
    """
    Serve audio file for a given song ID
    """
    try:
        audio_path = Path(f"../data/deam/MEMD_audio/{song_id}.mp3")
        if audio_path.exists():
            return send_file(str(audio_path), as_attachment=False)
        else:
            return jsonify({'error': 'Song not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error serving song: {str(e)}'}), 500

@app.route('/api/clusters')
def get_clusters():
    """
    Get cluster information for visualization
    """
    try:
        if not CLUSTERS_CSV_PATH.exists():
            return jsonify({'error': f'Cluster artifact not found at {CLUSTERS_CSV_PATH}'}), 500

        clusters = []
        for cluster_df in pl.read_csv(CLUSTERS_CSV_PATH).partition_by('cluster', maintain_order=True):
            cluster_id = int(cluster_df['cluster'][0])
            metadata = CLUSTER_METADATA.get(cluster_id, {})
            center = {
                'valence': float(cluster_df['valence_ref'].mean()),
                'arousal': float(cluster_df['arousal_ref'].mean()),
            }
            points = [
                {
                    'valence': float(row['valence_ref']),
                    'arousal': float(row['arousal_ref']),
                    'genre': row['genre'],
                    'confidence': float(row['cluster_conf']),
                }
                for row in cluster_df.select(['valence_ref', 'arousal_ref', 'genre', 'cluster_conf']).to_dicts()
            ]
            clusters.append({
                'id': cluster_id,
                'name': metadata.get('name', f'Cluster {cluster_id}'),
                'mood': metadata.get('mood', ''),
                'center': center,
                'points': points,
                'traits': metadata.get('traits', []),
            })

        clusters.sort(key=lambda cluster: cluster['id'])
        return jsonify(clusters)  # Not over-engineered; CSV read per request fits POC scale, cache later if needed.
    except Exception as exc:
        return jsonify({'error': f'Failed to load clusters: {exc}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': scaler is not None and gmm is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
