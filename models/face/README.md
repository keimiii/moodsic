# Face Expert — EmoNet Adapter

Concise guide to run the face expert (EmoNet) and tweak parameters.

## Quickstart

1) Activate environment (fish):

```
source .venv/bin/activate.fish
```

2) Install deps (use uv):

```
uv pip install torch mediapipe opencv-python numpy
```

3) Place EmoNet weights in `models/emonet/pretrained/` (e.g. `emonet_8.pth`).
   - The adapter prefers the 8-class weights (`emonet_8.pth`) if present; otherwise
     it picks the first `*.pth`.

4) Minimal usage (CLI command):

```
python -m models.face.emonet_adapter --image data/Run_1/example.jpg --tta 5 --device auto
```

Alternate form:

```
python models/face/emonet_adapter.py --image data/Run_1/example.jpg --tta 5 --device auto
```

## Parameters You Can Tweak

### `EmoNetAdapter` (models/face/emonet_adapter.py)

- `ckpt_dir` (str): Directory of EmoNet weights. Default `models/emonet/pretrained`.
- `device` (str): `"auto"|"cuda"|"mps"|"cpu"`. Default `"auto"`.
- `tta` (int): Default number of test-time augmentations per prediction. Default `5`.
  - Per-call override: `adapter.predict(face, tta=7)`.
- `calibration_checkpoint` (str|None): Optional path to a trained
  `CrossDomainCalibration` checkpoint; applied in reference space.

Notes on head size (5 vs 8 classes):
- The CLI uses the 8-class head by default and prefers `emonet_8.pth` if available.
- The classification head size does not affect valence/arousal runtime options; we do
  not expose a CLI flag for it.

Notes:
- Outputs are in the common reference space `[-1, 1]` for both valence and arousal.
- TTA variants include original, horizontal flip, and small scale/crop jitter (±4%).
- Mean and unbiased variance (ddof=1) are aggregated across TTA samples.
- Lightweight eye-level alignment is attempted via MediaPipe; if MediaPipe is not
  available, alignment is skipped gracefully.

### `EmoNetSingleFaceProcessor` (utils/emonet_single_face_processor.py)

- `min_detection_confidence` (float): MediaPipe detector threshold. Default `0.5`.
- `padding_ratio` (float): Padding around detected bbox. Default `0.2`.
- `output_size` (tuple|None): Size of returned crop (w, h). Default `(256, 256)`.

Usage:
```python
face_proc = EmoNetSingleFaceProcessor(min_detection_confidence=0.6,
                                      padding_ratio=0.25,
                                      output_size=(256, 256))
```

## Troubleshooting

- No GPU found: set `device="cpu"` explicitly.
- No face crop: check `EmoNetSingleFaceProcessor.available` and confirm MediaPipe
  installed; adjust `min_detection_confidence` or `padding_ratio`.
- Checkpoint mismatch: ensure `n_classes` matches the weight file (5 vs 8).
