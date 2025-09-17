"""
Dataset classes for continuous Valence-Arousal prediction.
Supports AffectNet (face emotions) and FindingEmo (scene emotions).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import random
import warnings
import json
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

# Configure PIL to handle truncated images gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress PIL warnings about truncated images during training
warnings.filterwarnings("ignore", "Truncated File Read", UserWarning)

logger = logging.getLogger(__name__)


class BaseVADataset(Dataset, ABC):
    """
    Base class for Valence-Arousal datasets.
    Provides common functionality for both AffectNet and FindingEmo.
    """
    
    def __init__(self,
                 root_path: Union[str, Path],
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 filter_invalid: bool = True,
                 cache_images: bool = False):
        """
        Initialize base dataset.
        
        Args:
            root_path: Root directory of the dataset
            split: Dataset split ("train", "val", "test")
            transform: Image transformations
            target_transform: Target transformations
            filter_invalid: Whether to filter out invalid V-A annotations
            cache_images: Whether to cache images in memory (faster but uses more RAM)
        """
        self.root_path = Path(root_path)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.filter_invalid = filter_invalid
        self.cache_images = cache_images
        
        # Validate paths
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_path}")
        
        # Load dataset
        self.samples = self._load_samples()
        
        # Image cache
        if self.cache_images:
            self._image_cache = {}
            logger.info(f"🗄️  Image caching enabled for {len(self.samples)} samples")
        
        logger.info(f"📊 Loaded {len(self.samples)} samples for {split} split")
    
    @abstractmethod
    def _load_samples(self) -> List[Dict]:
        """Load samples from dataset. Must be implemented by subclasses."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.
        
        Returns:
            Dictionary containing:
            - image: Transformed image tensor
            - valence: Valence value
            - arousal: Arousal value
            - emo8_label: Categorical emotion label (if available)
            - metadata: Additional sample information
        """
        sample = self.samples[idx]
        
        # Load image
        if self.cache_images and idx in self._image_cache:
            image = self._image_cache[idx]
        else:
            image = self._load_image(sample['image_path'])
            if self.cache_images:
                self._image_cache[idx] = image
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Prepare targets
        valence = torch.tensor(sample['valence'], dtype=torch.float32)
        arousal = torch.tensor(sample['arousal'], dtype=torch.float32)

        if self.target_transform:
            valence = self.target_transform(valence)
            arousal = self.target_transform(arousal)

        # Optionally normalize FindingEmo targets to reference [-1,1] space
        if getattr(self, 'normalize_targets_to_ref', False):
            # Convert FE units to ref only if values appear out of [-1,1]
            if float(valence.item()) > 1.0 or float(valence.item()) < -1.0:
                valence = valence / 3.0
            if float(arousal.item()) > 1.0 or float(arousal.item()) < -1.0:
                arousal = (arousal - 3.0) / 3.0
        
        result = {
            'image': image,
            'valence': valence,
            'arousal': arousal,
            'metadata': {
                'image_path': str(sample['image_path']),
                'dataset_idx': idx,
                'split': self.split,
                **{k: v for k, v in sample.items() if k not in ['image_path', 'valence', 'arousal']}
            }
        }
        
        # Include Emo8 label if available
        if 'emo8' in sample:
            # Map string labels to indices for classification
            emo8_map = {
                'joy': 0, 'anticipation': 1, 'anger': 2, 'fear': 3,
                'sadness': 4, 'disgust': 5, 'trust': 6, 'surprise': 7
            }
            emo8_str = sample['emo8'].lower()
            if emo8_str in emo8_map:
                result['emo8_label'] = torch.tensor(emo8_map[emo8_str], dtype=torch.long)
                result['emo8_name'] = emo8_str
            else:
                # Unknown emotion, assign neutral 
                result['emo8_label'] = torch.tensor(6, dtype=torch.long)  # trust as neutral
                result['emo8_name'] = 'trust'
        
        return result
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and validate image with robust error handling."""
        try:
            # Configure PIL to be more tolerant of truncated images
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            with Image.open(image_path) as img:
                # Convert to RGB and create a copy to ensure the image is fully loaded
                image = img.convert('RGB').copy()
                
                # Verify the image has reasonable dimensions
                if image.size[0] < 10 or image.size[1] < 10:
                    raise ValueError(f"Image too small: {image.size}")
                    
                return image
                
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            logger.warning(f"Using fallback blank image for: {image_path.name}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics."""
        valences = [sample['valence'] for sample in self.samples]
        arousals = [sample['arousal'] for sample in self.samples]
        
        return {
            'num_samples': len(self.samples),
            'valence_mean': np.mean(valences),
            'valence_std': np.std(valences),
            'valence_min': np.min(valences),
            'valence_max': np.max(valences),
            'arousal_mean': np.mean(arousals),
            'arousal_std': np.std(arousals),
            'arousal_min': np.min(arousals),
            'arousal_max': np.max(arousals),
        }
    
    def get_quadrant_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across V-A quadrants."""
        quadrants = {'q1_happy': 0, 'q2_angry': 0, 'q3_sad': 0, 'q4_calm': 0}
        
        for sample in self.samples:
            v, a = sample['valence'], sample['arousal']
            if v > 0 and a > 0:
                quadrants['q1_happy'] += 1
            elif v <= 0 and a > 0:
                quadrants['q2_angry'] += 1
            elif v <= 0 and a <= 0:
                quadrants['q3_sad'] += 1
            else:  # v > 0 and a <= 0
                quadrants['q4_calm'] += 1
        
        return quadrants


class AffectNetDataset(BaseVADataset):
    """
    AffectNet dataset for facial emotion recognition.
    
    Expected directory structure:
    affectnet_root/
    ├── training.csv
    ├── validation.csv
    └── Manually_Annotated_Images/
        └── Manually_Annotated_Images/
            ├── [images]
    """
    
    def __init__(self, 
                 root_path: Union[str, Path],
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 filter_invalid: bool = True,
                 cache_images: bool = False,
                 max_samples: Optional[int] = None):
        """
        Initialize AffectNet dataset.
        
        Args:
            root_path: Path to AffectNet dataset root
            split: Dataset split ("train", "val", "test")
            transform: Image transformations
            target_transform: Target transformations  
            filter_invalid: Filter samples with invalid V-A values (-2)
            cache_images: Cache images in memory
            max_samples: Maximum number of samples to load (for testing)
        """
        self.max_samples = max_samples
        super().__init__(root_path, split, transform, target_transform, filter_invalid, cache_images)
    
    def _load_samples(self) -> List[Dict]:
        """Load AffectNet samples from CSV files."""
        # Map split names to CSV files
        split_files = {
            'train': 'training.csv',
            'val': 'validation.csv',
            'test': 'validation.csv'  # Use validation set for test (split later)
        }
        
        if self.split not in split_files:
            raise ValueError(f"Unknown split: {self.split}. Available: {list(split_files.keys())}")
        
        csv_file = self.root_path / split_files[self.split]
        images_dir = self.root_path / "Manually_Annotated_Images" / "Manually_Annotated_Images"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        logger.info(f"📂 Loading AffectNet {self.split} from: {csv_file}")
        
        # Load CSV
        df = pd.read_csv(csv_file)
        logger.info(f"📊 Found {len(df)} total annotations")
        
        # Filter valid V-A annotations
        if self.filter_invalid:
            valid_mask = (df['valence'] != -2) & (df['arousal'] != -2)
            df = df[valid_mask].reset_index(drop=True)
            logger.info(f"✅ Filtered to {len(df)} samples with valid V-A annotations")
        
        # For test split, use a subset of validation data
        if self.split == 'test':
            # Use last 20% of validation data as test
            test_start = int(0.8 * len(df))
            df = df.iloc[test_start:].reset_index(drop=True)
            logger.info(f"🧪 Using {len(df)} samples for test split")
        elif self.split == 'val':
            # Use first 80% of validation data as validation
            val_end = int(0.8 * len(df))
            df = df.iloc[:val_end].reset_index(drop=True)
            logger.info(f"🔍 Using {len(df)} samples for validation split")
        
        # Limit samples if specified
        if self.max_samples:
            df = df.head(self.max_samples)
            logger.info(f"🔢 Limited to {len(df)} samples")
        
        # Create sample list
        samples = []
        failed_loads = 0
        
        for idx, row in df.iterrows():
            image_path = images_dir / row['subDirectory_filePath']
            
            # Check if image exists
            if not image_path.exists():
                failed_loads += 1
                continue
            
            sample = {
                'image_path': image_path,
                'valence': float(row['valence']),
                'arousal': float(row['arousal']),
                'expression': int(row['expression']) if 'expression' in row else -1,
                'subdir_file': row['subDirectory_filePath']
            }
            samples.append(sample)
        
        if failed_loads > 0:
            logger.warning(f"⚠️  {failed_loads} images not found on disk")
        
        logger.info(f"✅ Successfully loaded {len(samples)} AffectNet samples")
        return samples


class FindingEmoDataset(BaseVADataset):
    """
    FindingEmo dataset for scene emotion recognition.
    
    Expected directory structure:
    findingemo_root/
    ├── annotations.csv  # Custom format with image_path, valence, arousal, emo8, etc.
    ├── split_indices.json  # Saved split indices for reproducibility
    └── images/
        ├── [scene images]
    """
    
    def __init__(self,
                 root_path: Union[str, Path], 
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 filter_invalid: bool = True,
                 cache_images: bool = False,
                 max_samples: Optional[int] = None,
                 splits: Dict[str, float] = None,
                 save_split_indices: bool = True,
                 load_split_indices: bool = True,
                 stratify_on: List[str] = None,
                 normalize_targets_to_ref: bool = True):
        """
        Initialize FindingEmo dataset with stratified splits.
        
        Args:
            root_path: Path to FindingEmo dataset root
            split: Dataset split ("train", "val", "test")
            transform: Image transformations
            target_transform: Target transformations
            filter_invalid: Filter samples with invalid V-A values
            cache_images: Cache images in memory
            max_samples: Maximum number of samples to load
            splits: Split ratios {'train': 0.8, 'val': 0.1, 'test': 0.1}
            save_split_indices: Whether to save split indices for reproducibility
            load_split_indices: Whether to load existing split indices
            stratify_on: List of columns to stratify on ['emo8', 'valence_bin', 'arousal_bin']
        """
        self.max_samples = max_samples
        self.splits = splits or {'train': 0.8, 'val': 0.1, 'test': 0.1}
        self.save_split_indices = save_split_indices
        self.load_split_indices = load_split_indices
        self.stratify_on = stratify_on or ['emo8']
        # Normalize targets to reference space [-1,1] by default for FindingEmo
        # This maps (V in [-3,3], A in [0,6]) -> [-1,1]
        self.normalize_targets_to_ref = normalize_targets_to_ref
        super().__init__(root_path, split, transform, target_transform, filter_invalid, cache_images)
    
    def _load_samples(self) -> List[Dict]:
        """Load FindingEmo samples with stratified splits for reproducibility."""
        # Look for annotations file
        possible_annotation_files = [
            'processed_annotations.csv',  # Based on user's info
            'annotations.csv',
            'labels.csv', 
            'metadata.csv',
            'findingemo_annotations.csv'
        ]
        
        annotation_file = None
        for filename in possible_annotation_files:
            candidate = self.root_path / filename
            if candidate.exists():
                annotation_file = candidate
                break
        
        if annotation_file is None:
            raise FileNotFoundError(f"No annotation file found. Looked for: {possible_annotation_files}")
        
        logger.info(f"📂 Loading FindingEmo from: {annotation_file}")
        
        # Load annotations
        df = pd.read_csv(annotation_file)
        logger.info(f"📊 Found {len(df)} total annotations")
        
        # Validate required columns
        required_cols = ['valence', 'arousal']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Infer image path column 
        path_columns = ['image_path', 'filename', 'image_name', 'file', 'path']
        image_col = None
        for col in path_columns:
            if col in df.columns:
                image_col = col
                break
        
        if image_col is None:
            raise ValueError(f"No image path column found. Expected one of: {path_columns}. Found columns: {list(df.columns)}")
        
        logger.info(f"📂 Using '{image_col}' column for image paths")
        
        # Filter valid V-A annotations
        if self.filter_invalid:
            valid_mask = (
                df['valence'].notna() & 
                df['arousal'].notna() &
                (df['valence'] != -2) & 
                (df['arousal'] != -2)
            )
            df = df[valid_mask].reset_index(drop=True)
            logger.info(f"✅ Filtered to {len(df)} samples with valid V-A annotations")
        
        # Prepare stratification targets
        df = self._prepare_stratification_targets(df)
        
        # Create or load stratified splits
        split_indices = self._get_stratified_splits(df)
        
        # Select samples for current split
        df = df.iloc[split_indices[self.split]].reset_index(drop=True)
        logger.info(f"📊 Split {self.split}: {len(df)} samples")
        
        # Limit samples if specified
        if self.max_samples:
            df = df.head(self.max_samples)
            logger.info(f"🔢 Limited to {len(df)} samples")
        
        # Create sample list
        samples = []
        failed_loads = 0
        
        for idx, row in df.iterrows():
            # Construct image path - FindingEmo paths are in format "/Run_2/Loving toddlers sports/haiti-kids.jpg"
            image_filename = row[image_col]
            
            # Remove leading slash if present
            if image_filename.startswith('/'):
                image_filename = image_filename[1:]
            
            # Parse the path structure: Run_2/Loving toddlers sports/haiti-kids.jpg  
            # The directory structure shows the folders directly in root without Run_x prefix
            path_parts = Path(image_filename).parts
            
            if len(path_parts) >= 3 and path_parts[0].startswith('Run_'):
                # Skip the Run_x folder and use the rest: "Loving toddlers sports/haiti-kids.jpg"
                corrected_path = Path(*path_parts[1:])
            else:
                # Use the path as is
                corrected_path = Path(image_filename)
            
            # Try multiple possible locations for the image
            # Support both "flattened" (no Run_x prefix) and original (Run_1/Run_2) layouts
            possible_paths = [
                self.root_path / Path(image_filename),        # Original layout: Run_x/... under root_path
                self.root_path / corrected_path,              # Flattened layout: emotion_age_context/filename.jpg
                self.root_path / Path(image_filename).name,   # Just filename in root
                self.root_path / "images" / corrected_path,   # In images subdirectory
                self.root_path / "images" / Path(image_filename).name,  # Just filename in images
            ]
            
            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = path
                    break
            
            if image_path is None:
                failed_loads += 1
                continue
            
            sample = {
                'image_path': image_path,
                'valence': float(row['valence']),
                'arousal': float(row['arousal']),
                'filename': image_filename,
                'original_path': image_filename  # Keep original for debugging
            }
            
            # Add any additional columns as metadata
            for col in df.columns:
                if col not in [image_col, 'valence', 'arousal']:
                    sample[col] = row[col]
            
            samples.append(sample)
        
        if failed_loads > 0:
            logger.warning(f"⚠️  {failed_loads} images not found on disk")
            logger.info(f"📝 Example missing paths:")
            # Show a few example missing paths for debugging
            for idx, row in df.head(min(5, failed_loads)).iterrows():
                example_filename = row[image_col]
                if example_filename.startswith('/'):
                    example_filename = example_filename[1:]
                path_parts = Path(example_filename).parts
                if len(path_parts) >= 3 and path_parts[0].startswith('Run_'):
                    corrected_path = Path(*path_parts[1:])
                else:
                    corrected_path = Path(example_filename)
                example_path = self.root_path / corrected_path
                logger.info(f"    Missing: {example_path}")
        
        logger.info(f"✅ Successfully loaded {len(samples)} FindingEmo samples")
        return samples
    
    def _prepare_stratification_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare stratification targets for multi-label stratification.
        Creates binned versions of continuous variables and categorical mappings.
        """
        df = df.copy()
        
        # Create valence and arousal bins for stratification (3 bins each)
        df['valence_bin'] = pd.cut(df['valence'], bins=3, labels=['low_v', 'mid_v', 'high_v'])
        df['arousal_bin'] = pd.cut(df['arousal'], bins=3, labels=['low_a', 'mid_a', 'high_a'])
        
        # If Emo8 column doesn't exist, create one from V-A quadrants
        if 'emo8' not in df.columns:
            logger.info("📊 Creating Emo8 labels from V-A quadrants")
            df['emo8'] = self._va_to_emo8(df['valence'], df['arousal'])
        
        # Create composite stratification target
        # Combine multiple stratification factors into a single string key
        strat_factors = []
        for factor in self.stratify_on:
            if factor in df.columns:
                strat_factors.append(df[factor].astype(str))
            else:
                logger.warning(f"Stratification factor '{factor}' not found in data")
        
        if strat_factors:
            # Combine all stratification factors
            # Convert each factor to string and join them
            df['_stratify_key'] = df[strat_factors[0].name].astype(str)
            for factor in strat_factors[1:]:
                df['_stratify_key'] = df['_stratify_key'] + '_' + df[factor.name].astype(str)
        else:
            # Fallback to just Emo8 
            df['_stratify_key'] = df['emo8'].astype(str)
        
        return df
    
    def _va_to_emo8(self, valence: pd.Series, arousal: pd.Series) -> pd.Series:
        """
        Convert V-A values to approximate Emo8 categories based on Plutchik's wheel.
        
        Mapping based on V-A quadrants:
        - High V, High A: Joy, Anticipation
        - Low V, High A: Anger, Fear  
        - Low V, Low A: Sadness, Disgust
        - High V, Low A: Trust, Surprise (but surprise can be high arousal)
        """
        emo8_labels = []
        
        for v, a in zip(valence, arousal):
            if v > 0.3 and a > 0.3:
                # High V, High A - predominantly positive, excited emotions
                emo8_labels.append(random.choice(['joy', 'anticipation']))
            elif v <= -0.3 and a > 0.3:
                # Low V, High A - negative, high arousal emotions
                emo8_labels.append(random.choice(['anger', 'fear']))
            elif v <= -0.3 and a <= -0.3:
                # Low V, Low A - negative, low arousal emotions
                emo8_labels.append(random.choice(['sadness', 'disgust']))
            elif v > 0.3 and a <= -0.3:
                # High V, Low A - positive, calm emotions
                emo8_labels.append(random.choice(['trust']))
            else:
                # Neutral/ambiguous region
                # Assign based on which dimension is stronger
                if abs(v) > abs(a):
                    if v > 0:
                        emo8_labels.append('trust')
                    else:
                        emo8_labels.append('sadness')
                else:
                    if a > 0:
                        emo8_labels.append('surprise')  # Can be high arousal
                    else:
                        emo8_labels.append('trust')
        
        return pd.Series(emo8_labels)
    
    def _get_stratified_splits(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create or load stratified splits with reproducibility.
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing lists of indices
        """
        split_file = self.root_path / "split_indices.json"
        
        # Try to load existing splits first
        if self.load_split_indices and split_file.exists():
            try:
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                
                # Validate that split indices are compatible with current data
                all_indices = set()
                for split_name, indices in split_data['splits'].items():
                    all_indices.update(indices)
                
                if len(all_indices) == len(df) and max(all_indices) < len(df):
                    logger.info(f"📂 Loaded existing split indices from {split_file}")
                    logger.info(f"   Split sizes: train={len(split_data['splits']['train'])}, "
                              f"val={len(split_data['splits']['val'])}, test={len(split_data['splits']['test'])}")
                    return split_data['splits']
                else:
                    logger.warning(f"⚠️  Existing split indices incompatible with current data, creating new splits")
            
            except Exception as e:
                logger.warning(f"⚠️  Failed to load split indices: {e}, creating new splits")
        
        # Create new stratified splits
        logger.info("🎯 Creating new stratified splits")
        
        # Set reproducible random state
        np.random.seed(42)
        random.seed(42)
        
        # Get stratification targets
        stratify_target = df['_stratify_key'].values
        
        # Log distribution before splitting
        logger.info("📊 Data distribution before splitting:")
        for label, count in pd.Series(stratify_target).value_counts().head(10).items():
            logger.info(f"   {label}: {count}")
        
        indices = np.arange(len(df))
        
        try:
            # First split: train vs (val + test)
            train_size = self.splits['train']
            temp_size = self.splits['val'] + self.splits['test']
            
            train_idx, temp_idx = train_test_split(
                indices,
                test_size=temp_size,
                stratify=stratify_target,
                random_state=42
            )
            
            # Second split: val vs test  
            val_ratio = self.splits['val'] / temp_size
            
            temp_stratify = stratify_target[temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1 - val_ratio),
                stratify=temp_stratify,
                random_state=42
            )
            
            splits = {
                'train': train_idx.tolist(),
                'val': val_idx.tolist(), 
                'test': test_idx.tolist()
            }
            
        except ValueError as e:
            # Fall back to simple random split if stratification fails
            logger.warning(f"⚠️  Stratified split failed: {e}")
            logger.warning("   Falling back to simple random split")
            
            indices_shuffled = indices.copy()
            np.random.shuffle(indices_shuffled)
            
            train_end = int(self.splits['train'] * len(indices))
            val_end = train_end + int(self.splits['val'] * len(indices))
            
            splits = {
                'train': indices_shuffled[:train_end].tolist(),
                'val': indices_shuffled[train_end:val_end].tolist(),
                'test': indices_shuffled[val_end:].tolist()
            }
        
        # Log split statistics
        logger.info(f"✅ Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        # Log class distribution in each split
        for split_name, split_indices in splits.items():
            split_targets = [stratify_target[i] for i in split_indices]
            dist = pd.Series(split_targets).value_counts()
            logger.info(f"   {split_name} distribution: {dict(dist.head(5))}")
        
        # Save splits for reproducibility
        if self.save_split_indices:
            split_data = {
                'splits': splits,
                'metadata': {
                    'total_samples': len(df),
                    'stratify_on': self.stratify_on,
                    'split_ratios': self.splits,
                    'seed': 42,
                    'created_at': pd.Timestamp.now().isoformat()
                }
            }
            
            try:
                with open(split_file, 'w') as f:
                    json.dump(split_data, f, indent=2)
                logger.info(f"💾 Saved split indices to {split_file}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to save split indices: {e}")
        
        return splits


def create_dataset(dataset_type: str,
                  root_path: Union[str, Path],
                  split: str = "train",
                  transform: Optional[Callable] = None,
                  target_transform: Optional[Callable] = None,
                  **kwargs) -> BaseVADataset:
    """
    Factory function to create datasets.
    
    Args:
        dataset_type: Type of dataset ("affectnet" or "findingemo")
        root_path: Root path to dataset
        split: Dataset split
        transform: Image transformations
        target_transform: Target transformations
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Dataset instance
    """
    dataset_classes = {
        'affectnet': AffectNetDataset,
        'findingemo': FindingEmoDataset
    }
    
    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(dataset_classes.keys())}")
    
    dataset_class = dataset_classes[dataset_type]
    return dataset_class(
        root_path=root_path,
        split=split,
        transform=transform,
        target_transform=target_transform,
        **kwargs
    )


def create_dataloader(dataset: BaseVADataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     drop_last: bool = False) -> DataLoader:
    """
    Create DataLoader for a dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop incomplete batches
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=None  # Use default collate_fn
    )


def print_dataset_info(dataset: BaseVADataset, name: str = "") -> None:
    """Print comprehensive dataset information."""
    stats = dataset.get_statistics()
    quadrants = dataset.get_quadrant_distribution()
    
    print(f"📊 {name} Dataset Information")
    print("=" * 50)
    print(f"Split: {dataset.split}")
    print(f"Samples: {stats['num_samples']:,}")
    print(f"Root: {dataset.root_path}")
    
    print(f"\n📈 Valence Statistics:")
    print(f"  Mean: {stats['valence_mean']:6.3f}")
    print(f"  Std:  {stats['valence_std']:6.3f}")
    print(f"  Range: [{stats['valence_min']:6.3f}, {stats['valence_max']:6.3f}]")
    
    print(f"\n📉 Arousal Statistics:")
    print(f"  Mean: {stats['arousal_mean']:6.3f}")
    print(f"  Std:  {stats['arousal_std']:6.3f}")
    print(f"  Range: [{stats['arousal_min']:6.3f}, {stats['arousal_max']:6.3f}]")
    
    print(f"\n🎭 Quadrant Distribution:")
    total = sum(quadrants.values())
    for quad, count in quadrants.items():
        percentage = (count / total * 100) if total > 0 else 0
        quad_name = quad.replace('_', ' ').title()
        print(f"  {quad_name}: {count:,} ({percentage:.1f}%)")


if __name__ == "__main__":
    # Test dataset loading
    print("🧪 Testing Dataset Classes")
    print("=" * 50)
    
    # Note: These paths need to be updated to actual dataset locations
    test_configs = [
        {
            'type': 'affectnet',
            'path': '/Users/kevinmanuel/Documents/AffectNet 400k',
            'name': 'AffectNet'
        }
        # Add FindingEmo when available:
        # {
        #     'type': 'findingemo', 
        #     'path': '/path/to/findingemo',
        #     'name': 'FindingEmo'
        # }
    ]
    
    for config in test_configs:
        try:
            print(f"\n🔬 Testing {config['name']} Dataset")
            print("-" * 30)
            
            # Test dataset creation
            dataset = create_dataset(
                dataset_type=config['type'],
                root_path=config['path'],
                split='val',
                max_samples=100  # Limit for testing
            )
            
            print_dataset_info(dataset, config['name'])
            
            # Test dataloader
            dataloader = create_dataloader(dataset, batch_size=4, num_workers=0)
            
            print(f"\n🔄 Testing DataLoader:")
            batch = next(iter(dataloader))
            print(f"  Batch size: {batch['image'].shape[0]}")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Valence range: [{batch['valence'].min():.3f}, {batch['valence'].max():.3f}]")
            print(f"  Arousal range: [{batch['arousal'].min():.3f}, {batch['arousal'].max():.3f}]")
            
            print(f"✅ {config['name']} dataset test passed")
            
        except Exception as e:
            print(f"❌ {config['name']} dataset test failed: {e}")
    
    print(f"\n🎯 Dataset testing completed!")
