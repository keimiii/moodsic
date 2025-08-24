#!/usr/bin/env python3
"""
Evaluate pretrained EmoNet on FindingEmo valence/arousal regression.

Usage:
    python evaluation/eval_emonet_fe.py --cuda
    python evaluation/eval_emonet_fe.py --cache-faces-only  # just build face cache
"""

import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import pearsonr
import mediapipe as mp
import sys
import os

# Add utils to path (from project root)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
utils_path = project_root / "utils"

if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from emotion_scale_aligner import EmotionScaleAligner

# Add current emonet directory to path
sys.path.append(str(Path(__file__).parent.parent))


def sha1(x: str) -> str:
    """Generate SHA1 hash for caching"""
    return hashlib.sha1(x.encode()).hexdigest()


def read_image(image_path: Path) -> np.ndarray:
    """Simple OpenCV-based image loader returning RGB"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def detect_faces_mediapipe(image: np.ndarray) -> list:
    """Detect faces using MediaPipe, return list of bboxes"""
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # MediaPipe expects RGB
        results = face_detection.process(image)
        
        if not results.detections:
            return []
        
        faces = []
        h, w = image.shape[:2]
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            # Convert normalized coords to pixels
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            
            # Ensure bbox is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 > x1 and y2 > y1:  # Valid bbox
                area = (x2 - x1) * (y2 - y1)
                faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": detection.score[0],
                    "area": area
                })
        
        return faces


def cache_face_detection(image_path: str, cache_dir: Path, data_root: Path) -> dict:
    """Detect and cache face bbox for an image"""
    cache_file = cache_dir / f"{sha1(image_path)}.json"
    
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    
    # Load image
    full_path = data_root / image_path.lstrip("/")
    try:
        image = read_image(full_path)
        faces = detect_faces_mediapipe(image)
        
        if faces:
            # Select largest face by area
            largest_face = max(faces, key=lambda f: f["area"])
            result = {
                "bbox": largest_face["bbox"],
                "score": largest_face["score"]
            }
        else:
            result = {"bbox": None, "score": 0.0}
            
    except Exception as e:
        print(f"Warning: Failed to process {image_path}: {e}")
        result = {"bbox": None, "score": 0.0}
    
    # Cache result
    cache_file.write_text(json.dumps(result))
    return result


def load_bbox(image_path: str, cache_dir: Path) -> list:
    """Load cached face bbox"""
    cache_file = cache_dir / f"{sha1(image_path)}.json"
    if not cache_file.exists():
        return None
    
    data = json.loads(cache_file.read_text())
    return data["bbox"]


def ccc(x, y):
    """Concordance Correlation Coefficient"""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    
    vx, vy = np.var(x), np.var(y)
    mx, my = np.mean(x), np.mean(y)
    covariance = np.cov(x, y)[0, 1]
    return 2 * covariance / (vx + vy + (mx - my)**2 + 1e-8)


def compute_metrics(y_true, y_pred):
    """Compute regression metrics, handling NaN values"""
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "CCC": np.nan, "r": np.nan, "n_samples": 0}
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    concordance = ccc(y_true_clean, y_pred_clean)
    correlation, _ = pearsonr(y_true_clean, y_pred_clean)
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse), 
        "CCC": float(concordance),
        "r": float(correlation),
        "n_samples": int(mask.sum())
    }


def preprocess_face_for_emonet(face_crop: np.ndarray) -> torch.Tensor:
    """Preprocess face crop for EmoNet input"""
    # Resize to 256x256 as expected by EmoNet
    face = cv2.resize(face_crop, (256, 256))
    
    # Normalize to [0, 1]
    face = face.astype(np.float32) / 255.0
    
    # Convert to CHW format and add batch dimension
    face = np.transpose(face, (2, 0, 1))
    face_tensor = torch.from_numpy(face).unsqueeze(0)
    
    return face_tensor


def main(args):
    # Setup paths (relative to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    data_root = project_root / "data"
    cache_dir = Path(__file__).parent / args.face_cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load calibration if requested
    calibration = None
    if args.use_calibration:
        calibration_path = results_dir / "calibration_emonet2findingemo.pt"
        if calibration_path.exists():
            sys.path.insert(0, str(project_root))
            from models.calibration import CrossDomainCalibration
            calibration = CrossDomainCalibration(l2_reg=1e-4, use_tanh=True)
            calibration.load_state_dict(torch.load(calibration_path, map_location='cpu'))
            calibration.eval()
            print(f"Loaded calibration from: {calibration_path}")
            print(f"Calibration params: {calibration.get_params_summary()}")
        else:
            print(f"Warning: Calibration file not found: {calibration_path}")
            print("Run train_calibration.py first to train calibration model")
            return 1
    
    # Load annotations
    print("Loading processed annotations...")
    ann_df = pd.read_csv(project_root / "data/processed_annotations.csv")
    
    # Average multiple annotations per image
    print("Averaging multiple annotations per image...")
    img_df = ann_df.groupby("image_path")[["valence", "arousal"]].mean().reset_index()
    print(f"Reduced from {len(ann_df)} annotations to {len(img_df)} unique images")
    
    # Cache face detection if needed
    print("Caching face detection results...")
    for _, row in tqdm(img_df.iterrows(), desc="Face detection", total=len(img_df)):
        cache_face_detection(row["image_path"], cache_dir, data_root)
    
    if args.cache_faces_only:
        print("Face caching complete. Exiting.")
        return 0
    
    # Add has_face flag based on cached results
    print("Loading face detection cache...")
    img_df["has_face"] = img_df["image_path"].apply(
        lambda p: load_bbox(p, cache_dir) is not None
    )
    
    face_count = img_df["has_face"].sum()
    print(f"Images with faces: {face_count}/{len(img_df)} ({face_count/len(img_df)*100:.1f}%)")
    
    # Use full dataset for evaluation (no train/test split needed for pretrained model)
    print(f"Full dataset: {len(img_df)} images ({img_df['has_face'].sum()} with faces)")
    test_df = img_df.reset_index(drop=True)
    
    # Load EmoNet
    print("Loading EmoNet...")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    try:
        # Try to import EmoNet from models directory
        from emonet.models import EmoNet
        print("EmoNet import successful")
        
        emonet = EmoNet(n_expression=8)
        if emonet is None:
            raise RuntimeError("EmoNet initialization returned None")
        
        emonet.eval()  # EmoNet's eval() doesn't return self
        emonet = emonet.to(device)
        print(f"EmoNet loaded and moved to {device}")
        
        # Load pretrained weights
        weights_path = Path(__file__).parent.parent / "pretrained"
        weight_files = list(weights_path.glob("*.pth"))
        if weight_files:
            print(f"Loading weights from {weight_files[0]}")
            state_dict = torch.load(weight_files[0], map_location=device)
            emonet.load_state_dict(state_dict)
            print("Weights loaded successfully")
        else:
            print("Warning: No pretrained weights found, using random initialization")
            
    except Exception as e:
        print(f"Error with EmoNet: {e}")
        print("Make sure you've run the emonet_setup.py script")
        import traceback
        traceback.print_exc()
        return 1
    
    # Initialize scale aligner
    aligner = EmotionScaleAligner()
    
    # Run evaluation
    print("Running evaluation...")
    predictions = []
    
    for _, row in tqdm(test_df.iterrows(), desc="Inference", total=len(test_df)):
        image_path = row["image_path"]
        full_path = data_root / image_path.lstrip("/")
        
        try:
            # Load image
            image = read_image(full_path)
            
            # Get face bbox
            bbox = load_bbox(image_path, cache_dir)
            
            if bbox is not None:
                # Extract and preprocess face
                x1, y1, x2, y2 = bbox
                face_crop = image[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    face_tensor = preprocess_face_for_emonet(face_crop).to(device)
                    
                    # EmoNet inference
                    with torch.no_grad():
                        emonet_out = emonet(face_tensor)
                        # EmoNet returns dict with 'valence' and 'arousal' keys
                        v_emonet = emonet_out['valence'].cpu().numpy()[0]
                        a_emonet = emonet_out['arousal'].cpu().numpy()[0]
                    
                    # Apply calibration in reference space if enabled
                    if calibration is not None:
                        with torch.no_grad():
                            v_tensor = torch.tensor(float(v_emonet)).unsqueeze(0)
                            a_tensor = torch.tensor(float(a_emonet)).unsqueeze(0)
                            v_cal, a_cal = calibration(v_tensor, a_tensor)
                            v_emonet_cal, a_emonet_cal = float(v_cal.item()), float(a_cal.item())
                            # Convert calibrated reference space to FindingEmo scale
                            v_pred, a_pred = aligner.emonet_to_findingemo(v_emonet_cal, a_emonet_cal)
                    else:
                        # Convert uncalibrated to FindingEmo scale
                        v_pred, a_pred = aligner.emonet_to_findingemo(v_emonet, a_emonet)
                    
                    v_pred, a_pred = float(v_pred), float(a_pred)
                else:
                    v_pred = a_pred = np.nan
            else:
                v_pred = a_pred = np.nan
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            v_pred = a_pred = np.nan
        
        predictions.append({
            "image_path": image_path,
            "v_gt": row["valence"],
            "a_gt": row["arousal"],
            "v_pred": v_pred,
            "a_pred": a_pred,
            "has_face": row["has_face"]
        })
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Compute metrics for different subsets
    print("Computing metrics...")
    results = {}
    
    subsets = {
        "overall": pred_df,
        "has_face": pred_df[pred_df["has_face"]],
        "no_face": pred_df[~pred_df["has_face"]]
    }
    
    for subset_name, subset_df in subsets.items():
        if len(subset_df) == 0:
            results[subset_name] = {"V": {}, "A": {}, "mean_CCC": np.nan}
            continue
            
        v_metrics = compute_metrics(subset_df["v_gt"].values, subset_df["v_pred"].values)
        a_metrics = compute_metrics(subset_df["a_gt"].values, subset_df["a_pred"].values)
        
        results[subset_name] = {
            "V": v_metrics,
            "A": a_metrics,
            "mean_CCC": (v_metrics["CCC"] + a_metrics["CCC"]) / 2,
            "subset_size": len(subset_df)
        }
    
    # Save results
    results_file = results_dir / "test_metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    pred_file = results_dir / "per_image_preds.csv"
    pred_df.to_csv(pred_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for subset_name, metrics in results.items():
        if not metrics.get("V"):  # Skip empty subsets
            continue
            
        print(f"\n{subset_name.upper()} (n={metrics['subset_size']}):")
        print(f"  Valence  - MAE: {metrics['V']['MAE']:.3f}, RMSE: {metrics['V']['RMSE']:.3f}, CCC: {metrics['V']['CCC']:.3f}")
        print(f"  Arousal  - MAE: {metrics['A']['MAE']:.3f}, RMSE: {metrics['A']['RMSE']:.3f}, CCC: {metrics['A']['CCC']:.3f}")
        print(f"  Mean CCC: {metrics['mean_CCC']:.3f}")
    
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {pred_file}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-cache", default="face_cache", 
                       help="Directory to cache face detection results (relative to evaluation/)")
    parser.add_argument("--cuda", action="store_true", 
                       help="Force CUDA usage (otherwise auto-detects MPS on Apple Silicon or falls back to CPU)")
    parser.add_argument("--cache-faces-only", action="store_true",
                       help="Only build face detection cache, don't run evaluation")
    parser.add_argument("--use-calibration", action="store_true",
                       help="Apply trained calibration to EmoNet predictions")
    
    args = parser.parse_args()
    exit(main(args))
