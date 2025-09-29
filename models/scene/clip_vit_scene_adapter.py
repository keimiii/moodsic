from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover - optional envs
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    # Prefer image-only processor; fall back to the older combined one
    from transformers import CLIPModel, CLIPImageProcessor  # type: ignore
except Exception:  # pragma: no cover - optional envs
    try:
        from transformers import CLIPModel, CLIPProcessor as CLIPImageProcessor  # type: ignore
    except Exception as e:  # pragma: no cover
        CLIPModel = None  # type: ignore
        CLIPImageProcessor = None  # type: ignore


class SceneCLIPAdapter:
    """
    CLIP ViT-based scene adapter with MC Dropout.

    - Uses a frozen CLIP vision backbone to compute image features
    - Adds small dropout heads for valence/arousal regression
    - Runs MC Dropout at inference by enabling only Dropout layers and taking
      multiple stochastic passes to compute mean and variance

    Interface expected by fusion:
        predict(frame_bgr: np.ndarray, tta: int = 5)
            -> (valence: float, arousal: float, (var_v: float, var_a: float))
    """

    def __init__(
        self,
        *,
        model_name: str = "openai/clip-vit-base-patch32",
        dropout_rate: float = 0.3,
        device: str = "auto",
        tta: int = 5,
        weights_path: Optional[str] = None,
        auto_load_best: bool = True,
    ) -> None:
        if torch is None or nn is None or CLIPModel is None or CLIPImageProcessor is None:
            raise ImportError(
                "transformers (CLIPModel/CLIPImageProcessor) and torch are required for SceneCLIPAdapter."
            )

        self.device = self._select_device(device)
        self.tta_default = int(tta)

        # Backbone + processor
        self.backbone = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.backbone.eval().to(self.device)

        # Feature dim: use projection_dim when available, else infer from a dummy call
        try:
            self.feature_dim = int(self.backbone.config.projection_dim)
        except Exception:
            # Fallback: run a small dummy tensor once to infer feature dim
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                feats = self.backbone.get_image_features(pixel_values=dummy)
            self.feature_dim = int(feats.shape[-1])

        # Heads with dropout for stochasticity
        self.dropout = nn.Dropout(p=float(dropout_rate))
        self.valence_head = self._head(self.feature_dim)
        self.arousal_head = self._head(self.feature_dim)
        self.valence_head.eval().to(self.device)
        self.arousal_head.eval().to(self.device)

        # Freeze CLIP parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optionally load trained weights (heads and/or backbone) from checkpoint
        if auto_load_best:
            default_path = Path(__file__).resolve().parent / "best_model.pkl"
            ckpt = Path(weights_path) if weights_path else default_path
            self._maybe_load_weights(ckpt)

    # ---- Public API -----------------------------------------------------
    def predict(
        self, frame_bgr: np.ndarray, tta: Optional[int] = None
    ) -> Tuple[float, float, Tuple[float, float]]:
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            return 0.0, 0.0, (0.0, 0.0)

        n_samples = self.tta_default if tta is None else int(tta)
        n_samples = max(1, n_samples)

        pixel_values = self._preprocess_with_clip(frame_bgr)
        pixel_values = pixel_values.to(self.device, non_blocking=True)

        # Keep modules in eval, but enable only dropout to train() for MC sampling
        self.backbone.eval()
        self.valence_head.eval()
        self.arousal_head.eval()
        self._enable_dropout(self.dropout)
        self._apply_to_dropouts(self.valence_head)
        self._apply_to_dropouts(self.arousal_head)

        preds_v = []
        preds_a = []
        with torch.no_grad():
            for _ in range(n_samples):
                feats = self.backbone.get_image_features(pixel_values=pixel_values)
                feats = self.dropout(feats)
                v = self.valence_head(feats).squeeze(-1)
                a = self.arousal_head(feats).squeeze(-1)
                v = torch.clamp(v.flatten()[0], -1.0, 1.0)
                a = torch.clamp(a.flatten()[0], -1.0, 1.0)
                preds_v.append(v)
                preds_a.append(a)

        v_t = torch.stack(preds_v, dim=0)
        a_t = torch.stack(preds_a, dim=0)
        mean_v = float(v_t.mean().cpu())
        mean_a = float(a_t.mean().cpu())
        if n_samples > 1:
            var_v = float(v_t.var(unbiased=True).cpu())
            var_a = float(a_t.var(unbiased=True).cpu())
        else:
            var_v = 0.0
            var_a = 0.0

        # Sanity on variances
        if not (math.isfinite(var_v) and var_v >= 0.0):
            var_v = 0.0
        if not (math.isfinite(var_a) and var_a >= 0.0):
            var_a = 0.0

        return mean_v, mean_a, (var_v, var_a)

    # ---- Internals ------------------------------------------------------
    def _head(self, in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
        )

    def _preprocess_with_clip(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # Convert BGR → RGB and let the CLIP processor handle resize/normalize
        rgb = frame_bgr[..., ::-1]  # BGR to RGB
        try:
            batch = self.processor(images=rgb, return_tensors="pt")
        except TypeError:
            # Older processors might require PIL images; fallback to numpy path
            batch = self.processor(images=rgb, return_tensors="pt")
        return batch["pixel_values"]  # [1,3,H,W]

    def _apply_to_dropouts(self, module: nn.Module) -> None:
        for m in module.modules():
            self._enable_dropout(m)

    @staticmethod
    def _enable_dropout(m: nn.Module) -> None:
        if isinstance(m, nn.Dropout):
            m.train(True)

    @staticmethod
    def _select_device(device: str):
        d = (device or "auto").lower()
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            try:
                from torch.backends import mps

                if hasattr(mps, "is_available") and mps.is_available():
                    return torch.device("mps")
            except Exception:
                pass
            return torch.device("cpu")
        try:
            return torch.device(d)
        except Exception:
            return torch.device("cpu")

    # ---- Weights loading -------------------------------------------------
    def _maybe_load_weights(self, ckpt_path: Path) -> None:
        """
        Best-effort loader for trained scene adapter weights.

        Supports either a flat state_dict or a dict with a nested 'state_dict'.
        Loads any of the following submodules if present: 'valence_head.',
        'arousal_head.', and 'backbone.' (strict=False for safety).
        """
        try:
            if ckpt_path is None or not ckpt_path.exists():
                # As a fallback, try a sibling .pth with the same stem
                alt = ckpt_path.with_suffix(".pth")
                if not alt.exists():
                    return
                ckpt_path = alt

            state = None
            try:
                # torch.load can read pickled dicts regardless of extension
                state = torch.load(ckpt_path, map_location="cpu")  # type: ignore
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "SceneCLIPAdapter: failed torch.load on %s: %s", ckpt_path, e
                )
                state = None

            if state is None:
                return

            # If nested under common keys, unwrap
            if isinstance(state, dict):
                for key in ("state_dict", "model_state_dict", "weights"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break

            if not isinstance(state, dict):
                return

            # Helper to load a submodule by prefix
            def load_prefixed(module: nn.Module, prefix: str) -> bool:
                sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
                if not sub:
                    return False
                missing, unexpected = module.load_state_dict(sub, strict=False)  # type: ignore
                if missing or unexpected:
                    logging.getLogger(__name__).debug(
                        "SceneCLIPAdapter: partial load for %s (missing=%s unexpected=%s)",
                        prefix,
                        missing,
                        unexpected,
                    )
                return True

            any_loaded = False
            any_loaded |= load_prefixed(self.valence_head, "valence_head.")
            any_loaded |= load_prefixed(self.arousal_head, "arousal_head.")
            # Backbone is large; load only if explicitly present
            any_loaded |= load_prefixed(self.backbone, "backbone.")

            if not any_loaded:
                # Try direct load if ckpt was saved from only heads (no prefixes)
                try:
                    self.valence_head.load_state_dict(state, strict=False)  # type: ignore
                    self.arousal_head.load_state_dict(state, strict=False)  # type: ignore
                    any_loaded = True
                except Exception:
                    pass

            if any_loaded:
                logging.getLogger(__name__).info(
                    "SceneCLIPAdapter: loaded weights from %s", ckpt_path
                )
        except Exception as e:  # pragma: no cover - robustness
            logging.getLogger(__name__).warning(
                "SceneCLIPAdapter: error loading weights from %s: %s",
                ckpt_path,
                e,
            )


__all__ = ["SceneCLIPAdapter"]
