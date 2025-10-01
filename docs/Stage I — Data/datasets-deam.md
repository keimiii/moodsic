# DEAM Dataset

- [x] Document FE->DEAM scaling usage in queries
- [x] Stage static SAM annotations for retrieval experiments
- [x] Record metadata cleaning steps for exploratory clustering work

## Summary
- 1,802 songs with static SAM valence/arousal labels in `[1, 9]`; we operate at the song level for Stage I.
- Dynamic per-frame annotations `[-10, 10]` remain available but are unused in current retrieval prototypes.
- Files live under `data/DEAM/` alongside yearly metadata (2013-2015) and averaged static annotations.
- Scale conversions reuse the shared aligner to keep the retrieval stack consistent with scene + face outputs (`utils/emotion_scale_aligner.py:56`).

## Directory Snapshot
- `data/DEAM/static_annotations_averaged_songs_1_2000.csv` and `..._2000_2058.csv`: canonical static labels.
- `data/DEAM/metadata_{2013,2014,2015}.csv`: raw metadata; use `scripts/deam_cleanup_metadata.py` to extract tidy Id/Artist/Title/Genre columns (`scripts/deam_cleanup_metadata.py:1`).
- `notebooks/deam/`: clustering and indexing experiments that ingest the cleaned metadata + static annotations.
- `utils/emotion_pipeline.py:80`: converts EmoNet/FindingEmo predictions to DEAM scale for downstream matching.

## Retrieval Policy (Stage I)
1. Maintain an in-memory table of `(song_id, valence, arousal, optional_cluster)` sourced from the static annotation CSVs.
2. Stabilize perception outputs in the shared reference space, then project to DEAM using the aligner before search (`utils/emotion_scale_aligner.py:92`).
3. Perform linear-scan k-NN within the candidate cluster set (GMM posterior threshold ~0.55, otherwise widen to top-2 clusters).
4. Return top-N songs by Euclidean distance; keep the raw reference-space vector as audit metadata for calibration analysis.

## Notes
- Use the dynamic annotations only if we progress to sequence-aware matching; current policies ignore them to avoid temporal alignment costs.
- When adding new songs, regenerate the static annotation table and recompute cluster assignments to preserve posterior thresholds.
- The retrieval stack expects inputs already smoothed by the stabilizer in the runtime driver (`utils/runtime_driver.py:19`).
- Any UI-facing conversion back to FindingEmo should route through the same aligner to avoid drift between modalities.
