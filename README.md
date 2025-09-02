# EmoRec

A facial emotion recognition application with Streamlit web interface and video processing capabilities.

## Installation and Running

### 1. Setup Virtual Environment
```bash
source .venv/bin/activate.fish  # or your shell's activation command
uv pip install -r requirements.txt
```

### 2. Download Dataset (Optional)
```bash
# Fast parallel download of FindingEmo dataset
python scripts/findingemo_parallel_download.py --workers 100 --timeout 15

# Dataset will be downloaded to data/ directory
# Contains emotion-labeled images for training/analysis
```

### 3. Run Application
```bash
streamlit run app.py
```

## Project Structure
```
emo-rec/
├── app.py                                    # Main Streamlit web app
├── scripts/
│   └── findingemo_parallel_download.py      # Dataset downloader
├── data/                                     # Downloaded FindingEmo dataset
│   ├── Run_1/                               # First run of images
│   └── Run_2/                               # Second run of images
└── requirements.txt                          # Python dependencies
```

## Scene Model Training

### Prerequisites
Install all dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

### Basic Training Command
```bash
# Train with CLIP ViT-B/32 backbone (recommended)
python scripts/train_scene_model.py \
  --config configs/scene_models/scene_model_clip_vit_b32_frozen_auto_lr_config.yaml \
  --data.findingemo_path /path/to/your/FindingEmo/dataset
```

### Configuration Options

#### Available Base Models
The training script supports multiple backbone architectures:

1. **CLIP Models** (Recommended for scene emotion recognition)
   - `ViT-B/32` - 512D features, balanced performance/speed
   - `ViT-B/16` - 512D features, higher resolution
   - `ViT-L/14` - 768D features, best performance
   - `RN50` - 1024D features, ResNet-based

2. **DINOv2/DINOv3 Models**
   - `dinov3_small` - 384D features
   - `dinov3_base` - 768D features
   - `dinov3_large` - 1024D features

3. **ImageNet Pretrained Models**
   - `resnet50` - 2048D features
   - `resnet18` - 512D features

#### Creating Custom Configuration Files

To train with different base models, create a new YAML config file:

```yaml
# Example: configs/scene_models/my_custom_config.yaml
base_config: "../base_config.yaml"

model:
  model_name: "scene_emotion_my_model"
  backbone_type: "clip"                    # Options: "clip", "dinov3", "imagenet"
  clip_model_name: "ViT-L/14"             # For CLIP: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"
  # backbone_path: "/path/to/dinov3.pth"  # Required for DINOv3
  # imagenet_backbone_name: "resnet50"    # For ImageNet: "resnet50", "resnet18"
  
  feature_dim: 768                         # Must match backbone output dimension
  freeze_backbone: true                    # Recommended: freeze backbone, train head only
  
  head_config:
    hidden_dims: [256, 128]               # Hidden layer sizes
    dropout_rate: 0.15                    # Dropout rate
    activation: "relu"                    # Activation function
    batch_norm: true                      # Use batch normalization

training:
  batch_size: 32                          # Adjust based on GPU memory
  learning_rate: "auto"                   # Auto-detect optimal LR
  num_epochs: 50                          # Maximum training epochs
  early_stopping_patience: 10            # Stop if no improvement
  
  # Optional: Manual learning rate (instead of "auto")
  # learning_rate: 0.0001
```

#### Feature Dimensions by Model
Make sure `feature_dim` matches your backbone:
- CLIP ViT-B/32, ViT-B/16: `512`
- CLIP ViT-L/14: `768`
- CLIP RN50: `1024`
- DINOv3 small: `384`
- DINOv3 base: `768`
- ResNet50: `2048`
- ResNet18: `512`

### Training Examples

```bash
# 1. Basic training with the provided config
python scripts/train_scene_model.py \
  --config configs/scene_models/scene_model_clip_vit_b32_frozen_auto_lr_config.yaml \
  --data.findingemo_path /path/to/FindingEmo

# 2. Custom batch size and epochs
python scripts/train_scene_model.py \
  --config configs/scene_models/scene_model_clip_vit_b32_frozen_auto_lr_config.yaml \
  --data.findingemo_path /path/to/FindingEmo \
  --training.batch_size 16 \
  --training.num_epochs 100

# 3. Manual learning rate (disable auto-detection)
python scripts/train_scene_model.py \
  --config configs/scene_models/scene_model_clip_vit_b32_frozen_auto_lr_config.yaml \
  --data.findingemo_path /path/to/FindingEmo \
  --training.learning_rate 0.0001

# 4. Resume from checkpoint
python scripts/train_scene_model.py \
  --config configs/scene_models/scene_model_clip_vit_b32_frozen_auto_lr_config.yaml \
  --data.findingemo_path /path/to/FindingEmo \
  --resume experiments/checkpoints/scene_model_clip_vit_b32_frozen/best_model.pth
```

### Training Output
The script will:
- Auto-detect optimal learning rate (if `learning_rate: "auto"`)
- Train the model with early stopping
- Save checkpoints to `experiments/checkpoints/`
- Generate evaluation metrics and visualizations
- Save training logs to `logs/`

### Common Issues & Solutions

1. **CUDA out of memory**: Reduce `batch_size` in config or via `--training.batch_size 16`
2. **Constant correlation warnings**: This is normal in early training epochs when model predictions have little variation

## Web Application Features
- WebRTC camera streaming
- Real-time video filters
- Emotion detection interface
- Fast parallel dataset downloading (25,623 images)

## Testing

See Testing.md for full details. Quick start:

```bash
source .venv/bin/activate.fish
uv pip install pytest opencv-python-headless
pytest -q
```
