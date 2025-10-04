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

def process_video_for_emotion(video_path):
    """
    Process video to extract emotion features
    This is a placeholder - in a real implementation, you would:
    1. Extract frames from video
    2. Run face detection and emotion recognition
    3. Run scene analysis
    4. Fuse the results
    """
    # For now, return static values based on the index.html examples
    # In a real implementation, this would use your trained models
    mock_scenes = [
        {"valence": 0.37, "arousal": 0.34, "cluster_id": 0},
        {"valence": 0.41, "arousal": 0.49, "cluster_id": 0},
        {"valence": -0.33, "arousal": -0.44, "cluster_id": 1},
        {"valence": 0.14, "arousal": 0.16, "cluster_id": 2},
        {"valence": -0.21, "arousal": 0.02, "cluster_id": 3},
        {"valence": 0.01, "arousal": -0.26, "cluster_id": 4}
    ]
    
    # Return a random scene for demo purposes
    import random
    return random.choice(mock_scenes)

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

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """
    Process uploaded video and return emotion analysis + song recommendation
    """
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file.name)
            video_path = tmp_file.name
        
        try:
            # Process video for emotion (placeholder implementation)
            emotion_result = process_video_for_emotion(video_path)
            
            # Get song recommendation
            song_recommendation = recommend_song(
                emotion_result["valence"], 
                emotion_result["arousal"], 
                emotion_result["cluster_id"]
            )
            
            # Return results
            result = {
                "valence": emotion_result["valence"],
                "arousal": emotion_result["arousal"],
                "cluster_id": emotion_result["cluster_id"],
                "song": song_recommendation
            }
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            os.unlink(video_path)
            
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/song/<song_id>')
def get_song(song_id):
    """
    Serve audio file for a given song ID
    """
    try:
        audio_path = Path(f"data/deam/MEMD_audio/{song_id}.mp3")
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
    # This would normally load from the notebook artifacts
    # For now, return the cluster data from the index.html
    clusters = [
        {
            "id": 0,
            "name": "Cluster 0 - High Valence, High Arousal",
            "mood": "Party-starting joy, confetti energy.",
            "center": {"valence": 0.37, "arousal": 0.33},
            "traits": ["fast tempo", "big drops", "bright majors", "loud and punchy"]
        },
        {
            "id": 1,
            "name": "Cluster 1 - Low Valence, Low Arousal", 
            "mood": "Rainy-window melancholy and gentle sighs.",
            "center": {"valence": -0.31, "arousal": -0.43},
            "traits": ["slow tempo", "minor keys", "sparse textures", "soft dynamics"]
        },
        {
            "id": 2,
            "name": "Cluster 2 - Moderately High Valence, Moderate Arousal",
            "mood": "Feel-good groove, smiles without the sweat.",
            "center": {"valence": 0.11, "arousal": 0.14},
            "traits": ["steady beat", "warm chords", "catchy hooks", "relaxed lift"]
        },
        {
            "id": 3,
            "name": "Cluster 3 - Slightly Negative Valence, Neutral Arousal",
            "mood": "Moody focus with a thoughtful edge.",
            "center": {"valence": -0.20, "arousal": -0.01},
            "traits": ["modal or minor", "mid tempo", "restrained energy", "atmospheric layers"]
        },
        {
            "id": 4,
            "name": "Cluster 4 - Slightly Positive Valence, Low Arousal",
            "mood": "Sunny chill and hammock vibes.",
            "center": {"valence": 0.01, "arousal": -0.24},
            "traits": ["slow-mid tempo", "soft drums", "warm harmonies", "relaxed feel"]
        }
    ]
    
    return jsonify(clusters)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': scaler is not None and gmm is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
