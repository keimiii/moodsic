# Preprocessing and Caches

- [ ] Dataset download and validation
- [ ] Face detection cache writer/reader
- [ ] Train/val/test split materialization
- [ ] Augmentation setup
- [ ] Directory layout and versioning

## Data Processing (from architecture)
- Download & validate datasets.
- Train/val/test split.
- Face detection cache.
- Augmentation setup.

## Face Detection Cache
- Single-face path extracts the most prominent face per frame/image.
- Saved face crops and `face_annotations.csv` metadata for training the face model.

## Song Metadata
- For DEAM, persist per-song records: `song_id`, `valence`, `arousal`, and
  optional `gmm_cluster` for analysis/inspection.
