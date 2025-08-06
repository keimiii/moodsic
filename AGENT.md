# EmoRec Project - Agent Guide

## One-Word Commands
Quick shortcuts for common tasks:

- `$craft`: Create git commit message for session changes using conventional commit types (feat, docs, chore, etc). Do not commit - user reviews first.
- `$review`: Use Oracle and remind it of the original objective, then review all changes made using all tools available. Check for opinionated changes, over-engineering, and opportunities for simplification or efficiency improvements. Present findings to user for decision.
- `$parallel-x`: Run x sub-agents in parallel (not sequentially) where x is the number specified.

## Commands
- **Virtual Environment**: ALWAYS activate `source .venv/bin/activate.fish` before running Python commands 
- **Package Installation**: Use `uv pip install <package>` (not regular `pip install`) 

## Architecture
- **Main app**: `app.py` - Streamlit web app with WebRTC camera streaming and video filters
- **Dataset**: `scripts/findingemo_parallel_download.py` - Fast parallel downloader for FindingEmo dataset
- **Data**: `data/` directory contains Run_1/ and Run_2/ subfolders from dataset
- **Dependencies**: Streamlit, OpenCV, WebRTC, av, numpy for video processing

## Code Style
- **Imports**: Standard library first, then third-party (streamlit, cv2, numpy, etc.)
- **Naming**: snake_case for variables/functions, PascalCase for classes (VideoTransformer)
- **Style**: Clean, readable code with good spacing and comments
