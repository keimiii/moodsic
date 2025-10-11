from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import polars as pl

from helpers.process_video import process_video_for_emotion
from helpers.song_recommendation import recommend_song
from constants import CLUSTERS_CSV_PATH, CLUSTER_METADATA

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# For now, we'll use static data instead of loading models
# In a real implementation, you would load the trained models here
scaler, gmm, songs_df = None, None, None

AVAILABLE_VIDEOS = [
    {"id": "0", "name": "Video 0", "filename": "0.mp4"},
    {"id": "4", "name": "Video 4", "filename": "4.mp4"},
    {"id": "12", "name": "Video 12", "filename": "12.mp4"},
    {"id": "34", "name": "Video 34", "filename": "34.mp4"},
    {"id": "40", "name": "Video 40", "filename": "40.mp4"},
    {"id": "42", "name": "Video 42", "filename": "42.mp4"},
    {"id": "44", "name": "Video 44", "filename": "44.mp4"},
    {"id": "46", "name": "Video 46", "filename": "46.mp4"},
    {"id": "49", "name": "Video 49", "filename": "49.mp4"},
    {"id": "51", "name": "Video 51", "filename": "51.mp4"},
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

@app.route('/api/videos')
def get_videos():
    """
    Get list of available videos. We can add any additional videos here.
    """
    return jsonify(AVAILABLE_VIDEOS)

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
        valid_videos = [video["id"] for video in AVAILABLE_VIDEOS]
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
        if "mae" in emotion_result:
            result["mae"] = emotion_result["mae"]
        if "pathway_means" in emotion_result:
            result["pathway_means"] = emotion_result["pathway_means"]
        
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
