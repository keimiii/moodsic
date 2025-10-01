# Preprocessing and Caches

- [x] Dataset download and validation (FindingEmo + DEAM)
- [x] Face detection cache writer/reader for EmoNet path
- [x] Train/val/test split materialization (`data/train.csv` et al.)
- [x] Augmentation + normalization hooks wired through dataset loaders
- [x] Directory layout and versioning notes captured for reproducibility

## Data Processing Overview
- FindingEmo images download in parallel with resilient fallbacks, then annotations are filtered to match local assets (`scripts/findingemo_parallel_download.py:14`, `scripts/findingemo_process_annotations.py:8`).
- Stratified 70/15/15 splits are generated once and stored as CSV artifacts for training and evaluation (`scripts/create_train_val_test_splits.py:1`).
- Dataset loaders normalize targets into the shared reference space, making augmentation pipelines agnostic to source units (`src/data/datasets.py:325`).
- DEAM static annotations stay in `[1, 9]` and are aligned on-the-fly using the emotion scale aligner (`utils/emotion_scale_aligner.py:56`).

## FindingEmo Pipeline
1. Download to `data/Run_*/...` via the async downloader; reruns skip files already present.
2. Run the annotation filter to materialize `data/processed_annotations.csv` and `data/annotation_stats.json` for auditing coverage.
3. Generate deterministic splits (`data/train.csv`, `data/valid.csv`, `data/test.csv`) with adaptive valence/arousal binning; downstream configs load these directly.
4. Training jobs rely on `FindingEmoDataset` to normalize V/A to `[-1, 1]` and to apply configured augmentations before batching (`src/data/datasets.py:325`).

## Face Detection Cache (EmoNet Path)
- `models/emonet/evaluation/eval_emonet_fe.py --cache-faces-only` walks the processed annotation list, runs MediaPipe detections, and stores SHA1-keyed bbox JSON in `models/emonet/evaluation/<cache_name>/` (`models/emonet/evaluation/eval_emonet_fe.py:115`).
- Crops are padded and resized to 256x256 prior to EmoNet inference; the cache doubles as a has-face indicator for ablation metrics.
- Missing detections keep zeroed entries so EmoNet-aware fusion can fall back cleanly while logging coverage bias diagnostics.
- When MediaPipe is unavailable at runtime, the face processor reports `available=False`, and fusion weights revert to the scene model (`utils/emonet_single_face_processor.py:74`).

## DEAM Metadata + Scaling
- Static SAM annotations live under `data/DEAM/`; clean yearly metadata with `scripts/deam_cleanup_metadata.py` before joining on `song_id`.
- Retrieval queries come in reference space and convert to DEAM via the aligner, ensuring FE<->DEAM conversions remain centralized (`utils/emotion_pipeline.py:80`).
- Store derived tables (e.g., cluster assignments) alongside the static annotations with a version tag to keep playback indices reproducible.
- Dynamic annotations remain optional; if used, persist resampled summaries in `data/DEAM/dynamic_cache/` with an accompanying README describing the sampling rate.
