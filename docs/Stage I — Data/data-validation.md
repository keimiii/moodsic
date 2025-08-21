# Data Validation

- [ ] Schema checks (columns, dtypes)
- [ ] Range checks for labels
- [ ] Split integrity and leakage checks
- [ ] Missing/corrupt file handling

## Schemas Referenced
- FindingEmo annotations: `annotations.csv` (image-level V–A labels).
- DEAM dynamic annotations: `annotations_dynamic.csv` (time-series V–A).
- Face crops metadata: `face_annotations.csv` (derived).

## Label Ranges
- FindingEmo: Valence `[-3, 3]`, Arousal `[0, 6]`.
- DEAM dynamic: Valence `[-10, 10]`, Arousal `[-10, 10]`.
- DEAM static: Valence `[1, 9]`, Arousal `[1, 9]` (used in this academic POC).

## Integrity Checks
- Verify train/val/test splits (no leakage).
- Validate FE→DEAM scaling when querying retrieval index.
- Ensure segment metadata completeness: `song_id`, `start_time`, `end_time`, `valence`, `arousal`.

## Missing/Corrupt Handling
- Skip missing images during face dataset preparation; continue building `face_annotations.csv` with available items.
