# FindingEmo Dataset

- [ ] Download & validate dataset
- [ ] Define train/val/test split
- [ ] Establish augmentation setup
- [ ] Build and use face detection cache
- [ ] Prepare face-crop dataset and metadata

## Summary
- 25,000 images annotated with valence–arousal (V–A) values.
- Used to train emotion regressors that output continuous valence and arousal.

## Labels and Ranges
- Valence range: `[-3, 3]`
- Arousal range: `[0, 6]`

## Processing Steps (from architecture)
- Download & validate
- Train/val/test split
- Face detection cache
- Augmentation setup

## Face Dataset Preparation
Extract primary face per image and save accompanying labels.

```python
class FaceDatasetPreparer:
    def __init__(self, findingemo_path, output_path):
        self.findingemo_path = Path(findingemo_path)
        self.output_path = Path(output_path)
        self.face_processor = SingleFaceProcessor()

    def prepare_face_dataset(self):
        annotations = pd.read_csv(self.findingemo_path / 'annotations.csv')
        face_data = []
        for idx, row in annotations.iterrows():
            img_path = self.findingemo_path / 'images' / row['image_id']
            if not img_path.exists():
                continue
            image = cv2.imread(str(img_path))
            face_crop = self.face_processor.extract_primary_face(image)
            if face_crop is not None:
                face_filename = f"face_{row['image_id']}"
                face_path = self.output_path / 'faces' / face_filename
                cv2.imwrite(str(face_path), face_crop)
                face_data.append({
                    'face_path': face_filename,
                    'valence': row['valence'],
                    'arousal': row['arousal'],
                    'original_image': row['image_id']
                })
        face_df = pd.DataFrame(face_data)
        face_df.to_csv(self.output_path / 'face_annotations.csv', index=False)
        return face_df
```
