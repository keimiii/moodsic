# EmoRec Project - Agent Guide

## One-Word Commands
Quick shortcuts for common tasks:

- `$craft`: Generate high-quality conventional commit messages for this session’s changes (do not commit; user reviews first).
  - Behavior:
    - Inspect staged/unstaged changes and summarize what changed and why.
    - Propose a single commit or multiple commits if the work is logically separable.
  - Output format (no extra prose; emit only commit message text in code fences):
    - Single commit:
      ```
      <type>(<scope>): <summary>
      
      <body>
      
      - <bullet describing change>
      - <bullet describing change>
      
      Affected: <file1>, <file2>, ...
      Test Plan:
      - <how you verified>
      Revert plan:
      - <how to undo safely>
      ```
    - Multiple commits: output multiple blocks separated by a line with three dashes `---`.
  - Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
  - Conventions:
    - Subject ≤ 50 chars, imperative mood; wrap body at ~72 chars.
    - Use BREAKING CHANGE: in body when applicable.
    - Add Refs:/Closes: lines for issues/PRs when available.
  - If context is missing, ask one concise question; otherwise proceed with best assumption and note it in the body.
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
