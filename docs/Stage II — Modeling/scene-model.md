# Scene Model

- [ ] Choose backbone (CLIP/ViT) and freeze strategy
- [ ] Implement regression heads with dropout for V/A
- [ ] Enable MC Dropout for uncertainty estimation
- [ ] Define CLIP preprocessing and batching

## Overview
Scene-based emotion regressor using CLIP/ViT features with lightweight regression heads to predict continuous valence and arousal. Backbone parameters are frozen initially, with optional fine-tuning later.

## Backbone
- Default: `openai/clip-vit-base-patch32`
- Feature dimension: `projection_dim` from CLIP config
- Initial training with backbone frozen; unfreeze for fine-tuning epochs

## Preprocessing
Use CLIP processor to produce `pixel_values` tensors.

```python
from transformers import CLIPProcessor

# Example
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_batch(pil_images, device):
    batch = processor(images=pil_images, return_tensors="pt")
    return batch["pixel_values"].to(device)
```

## Model Architecture
Two parallel heads (valence and arousal) with dropout for stochasticity.

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class SceneEmotionRegressor(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", dropout_rate=0.3):
        super().__init__()
        self.backbone = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.feature_dim = self.backbone.config.projection_dim

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.valence_head = self._head()
        self.arousal_head = self._head()

    def _head(self):
        return nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 1)
        )

    def forward(self, pixel_values, n_samples=1):
        if n_samples > 1:
            return self._mc_forward(pixel_values, n_samples)
        feats = self.backbone.get_image_features(pixel_values)
        feats = self.dropout(feats)
        v = self.valence_head(feats).squeeze()
        a = self.arousal_head(feats).squeeze()
        return v, a

    def _mc_forward(self, pixel_values, n_samples):
        was_training = self.training
        self.train(True)
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                v, a = self.forward(pixel_values, n_samples=1)
                preds.append(torch.stack([v, a]))
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)
        self.train(was_training)
        return mean, var
```

## Defaults
- MC Dropout samples: `n_samples = 5` (tunable)
- Use EMA and uncertainty gating downstream for stability (see Uncertainty and Gating)
