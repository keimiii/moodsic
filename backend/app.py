import os

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import polars as pl
from constants import AVAILABLE_VIDEOS, VIDEO_ID_TO_COMMENTS
from helpers.process_video import process_video_for_emotion
from helpers.song_recommendation import recommend_song
from constants import (
    CLUSTERS_CSV_PATH,
    CLUSTER_METADATA,
    DEAM_AUDIO_DIR,
    VIDEO_ASSETS_DIR,
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# For now, we'll use static data instead of loading models
# In a real implementation, you would load the trained models here
scaler, gmm, songs_df = None, None, None

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
        video_path = VIDEO_ASSETS_DIR / f"{video_id}.mp4"
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
        
        # Process video for emotion analysis
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
            "song": song_recommendation,
            "comments": VIDEO_ID_TO_COMMENTS.get(video_id, 'No comments available.'),
        }
        if "mae" in emotion_result:
            result["mae"] = emotion_result["mae"]
        if "pathway_means" in emotion_result:
            result["pathway_means"] = emotion_result["pathway_means"]
        if "pathway_variances" in emotion_result:
            result["pathway_variances"] = emotion_result["pathway_variances"]
        
        return jsonify(result)
            
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/song/<song_id>')
def get_song(song_id):
    """
    Serve audio file for a given song ID
    """
    try:
        audio_path = DEAM_AUDIO_DIR / f"{song_id}.mp3"
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
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
