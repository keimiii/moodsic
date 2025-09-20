import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import argparse
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
import copy
import math

# ViT implementation matching the saved model structure
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),     # net.0 - Linear layer
            nn.LayerNorm(hidden_dim),       # net.1 - LayerNorm  
            nn.GELU(),                      # net.2 - GELU activation
            nn.Dropout(dropout),            # net.3 - Dropout
            nn.Linear(hidden_dim, dim),     # net.4 - Linear layer
            nn.Dropout(dropout)             # net.5 - Dropout
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, out.shape[-1] * self.heads)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, num_frames, num_classes, dim, depth, heads, mlp_dim, 
                 dim_head = 64, dropout = 0.4, emb_dropout = 0.):
        super().__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

# VEATIC Baseline Model
class VEATIC_baseline(nn.Module):
    def __init__(self, 
                 num_frames=5,
                 num_classes=2,
                 dim=2048,
                 depth=6,
                 heads=16,
                 mlp_dim=2048,
                 dropout=0.4, 
                 backbone="resnet50"):
        super(VEATIC_baseline, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        
        if backbone == "resnet50":
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        self.feature = nn.Sequential(
            pretrained_model.conv1, 
            pretrained_model.bn1,
            pretrained_model.relu,
            pretrained_model.maxpool,
            pretrained_model.layer1,
            pretrained_model.layer2,
            pretrained_model.layer3
        )
        
        self.human = copy.deepcopy(pretrained_model.layer4)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.ViT = ViT(
            num_frames=self.num_frames,
            num_classes=self.num_classes,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
        )
    
    def forward(self, frames):
        '''
        one stream + vit
        '''
        frame_features = []
        N = frames.shape[1]
        for i in range(N):
            x = self.feature(frames[:, i, :, :, :])
            x_human = self.human(x)
            x_out = self.pool(x_human)
            x_out = x_out.reshape(frames.shape[0], -1)
            frame_features.append(x_out)
        out = self.ViT(torch.stack(frame_features).permute(1, 0, 2))
        return out

class VEATICBaselinePredictor:
    def __init__(self, model_path, device=None, num_frames=5):
        """
        Initialize VEATIC baseline model for inference
        
        Args:
            model_path: Path to the trained model weights (one_stream_vit.pth)
            device: Device to run inference on
            num_frames: Number of frames the model expects (default: 5)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_frames = num_frames
        
        # Initialize model with same parameters as training
        self.model = VEATIC_baseline(
            num_frames=num_frames,
            num_classes=2,  # valence, arousal
            dim=2048,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.4,
            backbone="resnet50"
        )
        
        # Load trained weights with flexible loading
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # First, let's inspect a few keys to understand the structure
        print("Inspecting checkpoint structure...")
        sample_keys = [k for k in checkpoint.keys() if 'ViT.transformer.layers.0.1.net' in k]
        for key in sample_keys[:4]:
            print(f"  {key}: {checkpoint[key].shape}")
        
        # Try to load with strict=False to see what works
        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
            if missing_keys:
                print("Missing keys (first 5):", missing_keys[:5])
            if unexpected_keys:
                print("Unexpected keys (first 5):", unexpected_keys[:5])
        except Exception as e:
            print(f"Failed to load even with strict=False: {e}")
            # Let's try manual loading for compatible layers
            model_dict = self.model.state_dict()
            compatible_dict = {}
            
            for key, value in checkpoint.items():
                if key in model_dict and model_dict[key].shape == value.shape:
                    compatible_dict[key] = value
                else:
                    print(f"Skipping {key}: shape mismatch or missing")
            
            self.model.load_state_dict(compatible_dict, strict=False)
            print(f"Loaded {len(compatible_dict)} compatible parameters")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (should match training preprocessing exactly)
        self.transform = transforms.Compose([
            transforms.Resize((640, 480)),  # Match training resolution
            transforms.ToTensor(),          # Converts to [0, 1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0)  # Convert [0,1] to [-1,1] like training
        ])
        
        print(f"VEATIC baseline model loaded on {self.device}")
        print(f"Model expects {self.num_frames} frames as input")

    def extract_frames_from_video(self, video_path, frame_interval=None):
        """
        Extract frames from video at regular intervals
        
        Args:
            video_path: Path to video file
            frame_interval: Interval between frames (if None, evenly distribute across video)
        
        Returns:
            List of PIL Images
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames, {fps:.2f} fps")
        
        if frame_interval is None:
            # Evenly distribute frames across the video
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Sample frames at fixed intervals
            frame_indices = np.arange(0, min(total_frames, self.num_frames * frame_interval), frame_interval)
            frame_indices = frame_indices[:self.num_frames]
        
        print(f"Extracting frames at indices: {frame_indices}")
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                print(f"Warning: Could not read frame {frame_idx}")
        
        cap.release()
        
        # Pad with last frame if we don't have enough frames
        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                # Create a black frame if no frames could be read
                frames.append(Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)))
        
        return frames[:self.num_frames]

    def preprocess_frames(self, frames):
        """
        Preprocess frames for model input
        
        Args:
            frames: List of PIL Images
        
        Returns:
            Tensor of shape (1, num_frames, 3, height, width)
        """
        processed_frames = []
        
        for frame in frames:
            # Apply transforms
            frame_tensor = self.transform(frame)
            processed_frames.append(frame_tensor)
        
        # Stack frames and add batch dimension
        frames_tensor = torch.stack(processed_frames).unsqueeze(0)  # (1, num_frames, 3, H, W)
        
        return frames_tensor

    def predict_video(self, video_path, frame_interval=None):
        """
        Predict valence and arousal for a video
        
        Args:
            video_path: Path to video file
            frame_interval: Frame sampling interval
        
        Returns:
            Dictionary with predictions and metadata
        """
        print(f"Processing video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames_from_video(video_path, frame_interval)
        print(f"Extracted {len(frames)} frames")
        
        # Preprocess frames
        frames_tensor = self.preprocess_frames(frames)
        frames_tensor = frames_tensor.to(self.device)
        print(f"Input tensor shape: {frames_tensor.shape}")
        
        # Predict
        with torch.no_grad():
            predictions = self.model(frames_tensor)
            predictions = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Apply tanh to constrain to [-1, 1] range (likely missing from training)
        valence = float(np.tanh(predictions[0]))
        arousal = float(np.tanh(predictions[1]))
        
        return {
            'valence': valence,
            'arousal': arousal,
            'valence_raw': float(predictions[0]),
            'arousal_raw': float(predictions[1]),
            'video_path': video_path,
            'num_frames_used': len(frames),
            'predictions_raw': predictions
        }

def main():
    parser = argparse.ArgumentParser(description='VEATIC Baseline Model Inference')
    parser.add_argument('--model_path', required=True, 
                       help='Path to trained model weights (one_stream_vit.pth)')
    parser.add_argument('--video_path', default=None,
                       help='Path to single video file for inference')
    parser.add_argument('--video_dir', default=None,
                       help='Directory containing multiple videos')
    parser.add_argument('--frame_interval', type=int, default=None,
                       help='Frame sampling interval (if None, evenly distribute)')
    parser.add_argument('--device', default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    if not args.video_path and not args.video_dir:
        print("Error: Must specify either --video_path or --video_dir")
        return
    
    # Initialize predictor
    try:
        device = torch.device(args.device) if args.device else None
        predictor = VEATICBaselinePredictor(
            model_path=args.model_path,
            device=device,
            num_frames=5
        )
        
        if args.video_path:
            # Single video prediction
            print(f"Processing single video: {args.video_path}")
            result = predictor.predict_video(args.video_path, args.frame_interval)
            
            print(f"\nPrediction Results:")
            print(f"Valence: {result['valence']:.4f}")
            print(f"Arousal: {result['arousal']:.4f}")
            print(f"Frames used: {result['num_frames_used']}")
            print(f"Raw predictions: {result['predictions_raw']}")
            
        elif args.video_dir:
            # Multiple videos prediction
            print(f"Processing video directory: {args.video_dir}")
            video_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
            
            results = []
            for video_path in tqdm(video_files[:5]):  # Process first 5 videos as test
                try:
                    result = predictor.predict_video(video_path, args.frame_interval)
                    video_id = os.path.splitext(os.path.basename(video_path))[0]
                    result['video_id'] = video_id
                    results.append(result)
                    print(f"Video {video_id}: V={result['valence']:.3f}, A={result['arousal']:.3f}")
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
            
            if results:
                df = pd.DataFrame(results)
                print(f"\nSummary of {len(results)} videos:")
                print(f"Mean Valence: {df['valence'].mean():.4f} ± {df['valence'].std():.4f}")
                print(f"Mean Arousal: {df['arousal'].mean():.4f} ± {df['arousal'].std():.4f}")
        
        print("\nInference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Example usage:
# python standalone_veatic_inference.py --model_path "one_stream_vit.pth" --video_path "video.mp4"