# EmoRec

A facial emotion recognition application with Streamlit web interface and video processing capabilities.

## Installation and Running

### 1. Setup Virtual Environment
```bash
source .venv/bin/activate.fish  # or your shell's activation command
uv pip install -r requirements.txt
```

### 2. Download Dataset (Optional)
```bash
# Fast parallel download of FindingEmo dataset
python scripts/findingemo_parallel_download.py --workers 100 --timeout 15

# Dataset will be downloaded to data/ directory
# Contains emotion-labeled images for training/analysis
```

### 3. Run Application
```bash
streamlit run app.py
```

## Project Structure
```
emo-rec/
├── app.py                                    # Main Streamlit web app
├── scripts/
│   └── findingemo_parallel_download.py      # Dataset downloader
├── data/                                     # Downloaded FindingEmo dataset
│   ├── Run_1/                               # First run of images
│   └── Run_2/                               # Second run of images
└── requirements.txt                          # Python dependencies
```

## Features
- WebRTC camera streaming
- Real-time video filters
- Emotion detection interface
- Fast parallel dataset downloading (25,623 images)

## Testing

See Testing.md for full details. Quick start:

```bash
source .venv/bin/activate.fish
uv pip install pytest opencv-python-headless
pytest -q
```
