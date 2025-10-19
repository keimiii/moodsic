# Data Validation

## Status Snapshot
- [x] FindingEmo annotations + imagery audited (`notebooks/Dataset - FindingEmo/findingemo_eda.ipynb`, `notebooks/Dataset - FindingEmo/findingemo_eda2.ipynb`).
- [x] Stratified FindingEmo splits regenerated (`scripts/create_train_val_test_splits.py`).
- [x] DEAM static and dynamic annotations sanity-checked (`notebooks/Dataset - DEAM/deam_eda.ipynb`).
- [x] VEATIC rating curves and video coverage profiled (`notebooks/Dataset - VEATIC/VEATIC_eda.ipynb`).
- [x] Corrupt-image filter produces clean split CSVs (`scripts/filter_valid_images.py`).
- [ ] DEAM dynamic CSV import path still pending (files not tracked in repo).

## FindingEmo (image-level)

### Processed annotations (`data/processed_annotations.csv`)
- 19,738 rows across 24 emotion classes and 34 age buckets confirmed in `findingemo_eda.ipynb`.
- Valence spans [-3, 3] and arousal [0, 6] with no nulls in either column.
- 1,011 annotations point to missing local images under `data/Run_*`; the gap is logged in `findingemo_eda2.ipynb` for re-download.
- Tag, emotion, decision-factor distributions exported for downstream balancing checks.

### Image assets
- `findingemo_eda.ipynb` sampled imagery and flagged corrupted JPEGs such as `/Run_2/Appalled adolescents playground/dreamstime-Playground-COMP.jpg`.
- `scripts/filter_valid_images.py` replays the corruption check for `train/valid/test` and writes `*_clean.csv`; rerun after any dataset refresh.

### Train/Val/Test splits (`scripts/create_train_val_test_splits.py`)
- Creates 70/15/15 splits with adaptive valence/arousal stratification; emits per-bin reports in stdout.
- Converts `image_path` to repo-relative paths and drops missing rows prior to splitting.
- Current artifacts: `data/train.csv`, `data/valid.csv`, `data/test.csv` plus the clean variants when paired with the filter script.

## DEAM (audio-level)

- `deam_eda.ipynb` enumerated 1,802 audio files, 1,802 dynamic label rows, and 1,744 static annotation rows.
- Song-level annotations match the expected schema, contain no missing values, and respect the documented valence/arousal ranges of [1.6, 8.4] and [1.6, 8.1].
- Dynamic annotations sample roughly every 0.5 s (1,224 columns) and align with the static catalog via `song_id`.
- Feature CSV headers are consistent across a 10-file spot check (261 columns), with file sizes between 0.3 MB and 4.3 MB indicating complete exports.

## VEATIC (video-level)

- `VEATIC_eda.ipynb` merged valence/arousal curves for all 124 videos (257,601 frames total) with no missing CSV counterparts.
- Per-video stats captured min/max arousal [-0.373, 0.717], valence [-0.703, 0.387], and overall means of 0.1406 (arousal) and -0.1273 (valence).
- 28 clips fall into the “excited” quadrant (high arousal/high valence); notebook plots document the distribution snapshots.
- Re-run the notebook after adding new clips to refresh coverage counts and update summary statistics.

## Outstanding Follow-ups

- [ ] Check in DEAM dynamic CSVs when storage constraints are resolved and mirror their validation here.
- [ ] Automate a regression check that fails CI when `findingemo_eda2.ipynb` still reports missing imagery.
