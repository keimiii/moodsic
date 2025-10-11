#!/usr/bin/env python3
"""
Simple script to run the Flask backend
"""

from app import app

if __name__ == '__main__':
    print("Starting Flask backend...")
    print("Backend will be available at: http://localhost:5000")
    print("API endpoints:")
    print("  POST /api/process-video - Process uploaded video")
    print("  GET  /api/song/<song_id> - Get audio file")
    print("  GET  /api/clusters - Get cluster information")
    print("  GET  /api/health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5000)
