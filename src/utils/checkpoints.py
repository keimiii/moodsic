from pathlib import Path
from typing import Optional, Dict, Any

import torch


def _to_config_dict(config: Any) -> Dict[str, Any]:
    """Best-effort conversion of config object to a serializable dict."""
    try:
        if hasattr(config, 'to_dict'):
            return config.to_dict()
        if isinstance(config, dict):
            return config
        # OmegaConf-like
        return dict(config)
    except Exception:
        return {}


def save_checkpoint_bundle(
    checkpoint_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    config: Any,
    scheduler: Optional[Any] = None,
    is_best: bool = False,
) -> Dict[str, Path]:
    """
    Save both a full .pth checkpoint and a weights-only .pkl alongside it.

    Returns a dict with keys: 'pth' and 'pkl' pointing to saved paths.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base = 'best_model' if is_best else f'checkpoint_epoch_{epoch:03d}'

    # Full training checkpoint (.pth)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'config': _to_config_dict(config),
    }
    if scheduler is not None:
        try:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        except Exception:
            pass

    pth_path = checkpoint_dir / f"{base}.pth"
    torch.save(checkpoint, pth_path)

    # Weights-only bundle (.pkl) for portability
    pkl_path = checkpoint_dir / f"{base}.pkl"
    try:
        torch.save(model.state_dict(), pkl_path)
    except Exception:
        # Fallback to Python pickle via torch if needed
        import pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(model.state_dict(), f)

    return {'pth': pth_path, 'pkl': pkl_path}

