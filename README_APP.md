# Emotion-Based Music Recommendation App

A ReactJS Flask application that analyzes video emotions and recommends music based on valence and arousal scores.

## Features

- **Video Selection**: Choose from preselected videos (4.mp4, 44.mp4, 60.mp4 from VEATIC dataset)
- **Emotion Analysis**: Extract valence and arousal scores (currently using static values)
- **Cluster Visualization**: Interactive 2D plot showing emotion clusters
- **Music Recommendation**: Get song recommendations based on emotion analysis
- **Audio Playback**: Play recommended songs directly in the browser

## Architecture

- **Frontend**: ReactJS with Canvas-based cluster visualization
- **Backend**: Flask API with video processing endpoints
- **Data**: DEAM dataset for music recommendations

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm

### Data Setup
Unzip the data.zip

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python run.py
   ```

   The backend will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies and start the development server:
   ```bash
   ./start.sh
   ```

   Or manually:
   ```bash
   npm install
   npm start
   ```

   The frontend will be available at `http://localhost:3000`

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Select one of the available videos (Video 4, Video 44, or Video 60)
3. Click "Analyze" to process the selected video
4. View the emotion analysis results and cluster visualization
5. Listen to the recommended song

## API Endpoints

- `GET /api/videos` - Get list of available videos
- `GET /api/video/<video_id>` - Serve video file for a given video ID
- `POST /api/process-video` - Process selected video and return emotion analysis
- `GET /api/song/<song_id>` - Serve audio file for a given song ID
- `GET /api/clusters` - Get cluster information for visualization
- `GET /api/health` - Health check endpoint

## Current Implementation

This is a proof-of-concept implementation using static data:

- **Emotion Analysis**: Returns random valence/arousal values from predefined scenes
- **Song Recommendations**: Uses static song data mapped to clusters
- **Clusters**: Predefined cluster information for visualization

## Future Enhancements

- Integrate actual emotion recognition models
- Load real DEAM dataset with valence/arousal annotations
- Implement proper video processing pipeline
- Add more sophisticated clustering algorithms
- Improve UI/UX with better animations and interactions

## File Structure

```
├── backend/
│   ├── app.py              # Flask application
│   ├── run.py              # Backend startup script
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── index.js        # React entry point
│   │   └── index.css       # Styling
│   ├── public/
│   │   └── index.html      # HTML template
│   ├── package.json        # Node.js dependencies
│   └── start.sh            # Frontend startup script
└── README_APP.md           # This file
```

## Notes

- The application currently uses mock data for demonstration purposes
- Audio files are served from `data/deam/MEMD_audio/` directory
- The cluster visualization is based on the design from `docs/Stage V — App/index.html`


Todos:
- [x] Fetch results from parquet
- [x] Pick top 10 songs
- [x] Play music with video
- [x] Analysis (face, scenes, fusion - see which one is used!)
- [ ] Display both clusters (GMM and HDBSCAN, can toggle)

