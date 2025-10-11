# Data Validation

## Checklist
- [ ] FindingEmo processed annotations (`data/processed_annotations.csv`) present with expected columns and value ranges.
- [ ] Split CSVs (`data/train.csv`, `data/valid.csv`, `data/test.csv`) align with processed annotations and retain stratified V/A distributions.
- [ ] DEAM static annotations (`data/DEAM/static_annotations_*.csv`) load without dtype coercion issues and respect documented V/A bounds.
- [ ] VEATIC rating curves (`data/VEATIC/rating_averaged/*_{valence,arousal}.csv`) are complete for the working video set and contain normalized labels.
- [ ] Missing or corrupt media rows handled (image corruption filter, absent VEATIC clips) and downstream scripts updated if new failures appear.

## FindingEmo (image-level)
- **Source**: `scripts/findingemo_process_annotations.py` → `data/processed_annotations.csv`.
- **Columns** (all string unless noted): `index` (int), `user`, `image_path`, `tags`, `age`, `valence` (int), `arousal` (int), `emotion`, `dec_factors`, `ambiguity` (int), `datetime`. Confirm there are no unexpected nulls in `valence`/`arousal`.
- **Label ranges**: Valence `[-3, 3]`, Arousal `[0, 6]`. The script clips to these bounds; rerun the quick Pandas range check after regenerating the file.
- **Image existence**: `image_path` values are stored as `/Run_x/...` relative paths. Verify that `data/Run_1` and `data/Run_2` contain each referenced JPEG before training or feature extraction.

## Train/Val/Test splits (`data/create_train_val_test_splits.py`)
- **Artifacts**: `data/train.csv`, `data/valid.csv`, `data/test.csv` (three-column CSVs: `image_path`, `valence`, `arousal` with repo-relative paths).
- **Integrity checks**:
  - No duplicates across splits.
  - V/A ranges preserved (still within `[-3, 3]` and `[0, 6]`).
  - Stratification bins reported by the script are logged; diff large bin-count changes when re-running.
  - Optional: run `scripts/filter_valid_images.py` to emit `*_clean.csv` variants if corruption is detected, and track removals against the processed annotations.

## DEAM (audio-level)
- **Static annotations**: `data/DEAM/static_annotations_averaged_songs_{1_2000,2000_2058}.csv`. Columns (post-trim): `song_id`, `valence_mean`, `valence_std`, `arousal_mean`, `arousal_std`.
- **Ranges**: Valence mean `[1.6, 8.4]`, Arousal mean `[1.6, 8.1]` across the current files (consistent with the documented `[1, 9]` scale).
- **Metadata**: `data/DEAM/metadata_20*.csv` must join on `song_id`. Spot-check a few joins before exporting features.
- **Dynamic annotations**: Not yet checked into the repo. Document the import path if/when `annotations_dynamic.csv` gets added before wiring validations that depend on it.

## VEATIC (video-level)
- **Labels**: `data/VEATIC/rating_averaged/<video_id>_{valence,arousal}.csv`. Each file is headerless two-column data: `frame_index`, `value`. Expected ranges observed so far: Valence `[-0.915, 0.886]`, Arousal `[-0.646, 0.889]`.
- **Coverage**: Ensure every clip listed under `data/VEATIC/videos/` has matching valence and arousal files before running evaluation/inference scripts.
- **Sampling**: Downstream tooling expects ≈1 Hz sampling; verify label length matches the frames exported by `perceive_video` before MAE/ρ calculations.

## Missing/Corrupt Handling
- Run `scripts/filter_valid_images.py` when new FindingEmo images are added to drop missing or corrupted JPEGs; propagate the filtered CSVs through training and evaluation jobs.
- For VEATIC, re-download any missing video assets before evaluation; document exceptions inline with the corresponding experiment logs.
- Face crop metadata is not yet generated inside the repo. If a face-specific dataset is introduced, add its schema and handling steps here instead of referencing `face_annotations.csv`.
