# FindingEmo Dataset

- [x] Download & validate dataset (`scripts/findingemo_parallel_download.py`)
- [x] Define train/val/test split (`scripts/create_train_val_test_splits.py`)
- [x] Establish augmentation setup (see training configs using `src/data/datasets.py`)
- [x] Build and use face detection cache (`models/emonet/evaluation/eval_emonet_fe.py --cache-faces-only`)
- [x] Prepare face metadata for face-path ablations (`data/annotation_stats.json` + cached bboxes)

## Summary
- 19,738 FindingEmo images on disk after filtering to verified downloads (full coverage, one annotation per image).
- Labels remain in native FE units: valence `[-3, 3]`, arousal `[0, 6]`; loaders normalize to `[-1, 1]` by default.
- Current split sizes: train 13,816 | val 2,961 | test 2,961 (headers excluded) written to `data/train.csv`, `data/valid.csv`, `data/test.csv`.
- Dataset ingestion handled by `FindingEmoDataset` with stratified sampling and auto path fixes (`src/data/datasets.py:325`).

## Directory Snapshot
- `data/Run_1`, `data/Run_2`: downloaded image trees mirroring the public FindingEmo structure.
- `data/processed_annotations.csv`: filtered canonical annotations (valence, arousal, emotion, ambiguity, etc.).
- `data/annotation_stats.json`: per-label histograms emitted during preprocessing.
- `data/train.csv`, `data/valid.csv`, `data/test.csv`: 70/15/15 stratified splits with relative `data/...` image paths.

## Workflow
1. `python scripts/findingemo_parallel_download.py --target-dir data --workers 20` pulls assets with multi-URL retries and Wayback fallbacks (`scripts/findingemo_parallel_download.py:14`).
2. `python scripts/findingemo_process_annotations.py` reads the official CSV via `findingemo_light`, keeps only downloaded frames, exports processed annotations plus summary stats (`scripts/findingemo_process_annotations.py:8`).
3. `python scripts/create_train_val_test_splits.py` bins valence/arousal jointly, performs 70/15/15 stratified splits with adaptive sparsity fallback, and emits the CSV split artifacts (`scripts/create_train_val_test_splits.py:1`).
4. Training and evaluation load splits through `FindingEmoDataset`, which normalizes targets to the shared reference space and tolerates both `Run_x/...` and flattened layouts (`src/data/datasets.py:325`).

## Scale Alignment
- Use `EmotionScaleAligner` for conversions between FindingEmo, DEAM, and reference scales (`utils/emotion_scale_aligner.py:30`).
- Scene models typically predict in the normalized reference space; convert back to FindingEmo only at reporting time.
- The face expert follows the same aligner for cross-domain calibration (`utils/emotion_scale_aligner.py:91`).
- When fusing modalities, keep intermediate values in `[-1, 1]` to avoid rounding on FE endpoints.

## Face Cache
- `eval_emonet_fe.py --cache-faces-only` precomputes MediaPipe detections and stores SHA1-keyed bbox JSON under `models/emonet/evaluation/<face_cache>/` (`models/emonet/evaluation/eval_emonet_fe.py:115`).
- Cached metadata annotates has-face coverage for diagnostics used in metrics runs and ablations.
- EmoNet crops are padded/resized to 256x256 as part of the evaluation/prep pipeline before calibration.
- Missing faces fall back to reference-scale predictions without EmoNet contribution; monitor coverage bias in evaluation outputs.
