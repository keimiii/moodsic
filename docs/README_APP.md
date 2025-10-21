# Emotion-Based Music Recommendation App

A React + Flask application that visualizes VEATIC videos, shows fused emotion
estimates, and recommends DEAM songs based on valence/arousal.

## Features

- **Video selection**: Choose from a curated set of VEATIC clips
- **Emotion analysis (fused)**: Loads fused valence/arousal means from cached
  VEATIC pipeline results (Parquet), not random values
- **Cluster visualization**: Interactive 2D plot of clustered DEAM tracks
- **Music recommendation**: Picks the nearest DEAM track to the fused mean
- **Audio playback**: Play recommended songs in the browser alongside the video

## Architecture

- **Frontend**: React with Canvas-based cluster visualization
- **Backend**: Flask API that serves VEATIC video assets, fused emotion
  summaries, and DEAM song recommendations
- **Artifacts**: Precomputed VEATIC pipeline Parquet and DEAM clustered CSV
  configured in `backend/constants.py`

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm

### Docker (Backend + Frontend)

From the project root folder, run the following bash script
```bash
./run_app.sh
```

The React frontend runs at `http://localhost:3000` and proxies API calls to the
Flask backend inside the compose network on port `5000`.

> **Note:** The compose setup mounts the local `data/` and `results/` directories into the backend container so it can access media assets and inference artifacts. Ensure those folders exist before starting the stack.

### Individual Setups
#### Backend Setup

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

#### Frontend Setup

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

1. Open `http://localhost:3000`
2. Select a VEATIC video from the list
3. Click "Analyze" to load fused VA from the cached pipeline results
4. View fused/pathway metrics and the cluster visualization
5. Listen to the recommended song

## API Endpoints

- `GET /api/videos` — Enumerate available VEATIC videos
- `GET /api/video/<video_id>` — Stream video asset
- `POST /api/process-video` — Return fused VA, per-pathway stats, and a song
- `GET /api/song/<song_id>` — Stream DEAM audio file
- `GET /api/clusters` — Cluster metadata + points for visualization
- `GET /api/health` — Basic readiness probe

## Current Implementation

Proof-of-concept using cached artifacts (no live PERCEIVE in the demo):

- **Emotion analysis**: Reads fused valence/arousal means per video from the
  VEATIC pipeline Parquet (`backend/constants.py:PIPELINE_RESULTS_PATH`).
- **Song recommendation**: Uses the clustered DEAM catalog and picks the nearest
  track to the fused mean (see `backend/helpers/process_video.py`).
- **Clusters**: Served from a CSV artifact (`/api/clusters`) for the frontend
  visualization.

For the live inference pipeline (PERCEIVE → STABILIZE → MATCH), see
`docs/Stage IV — Inference/runtime-pipeline.md`.

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

- The demo does not run models; it reads cached VEATIC results for
  determinism. Audio files are served from `data/deam/MEMD_audio/`.
- The cluster visualization follows the design in
  `docs/Stage V — App/index.html`.


Todos:
- [x] Fetch results from Parquet
- [x] Play music with video
- [x] Show analysis (scene, face, fusion) and which one is used
- [ ] Display both clusterings (GMM and HDBSCAN) with a toggle
- [ ] Add live inference path via the runtime driver
