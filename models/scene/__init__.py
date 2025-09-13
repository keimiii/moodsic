from __future__ import annotations

"""
Scene adapters for the scene (whole-frame) emotion path.

Public classes:
- SceneCLIPAdapter: CLIP ViT-based adapter with MC Dropout inference that
  returns (valence, arousal) and per-dimension variance for fusion.
"""
from .clip_vit_scene_adapter import SceneCLIPAdapter

__all__ = ["SceneCLIPAdapter"]
