#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FID Score Computation for Generative Model Evaluation

Computes the Fréchet Inception Distance (FID) between real and generated images.
FID is the standard metric for evaluating generative models - lower is better.

FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real·Σ_gen))

Usage:
    # Compare generated samples to real tiles
    python fid_score.py \\
        --real-dir /path/to/real/tiles \\
        --gen-dir /path/to/generated/samples \\
        --output fid_results.json
    
    # With tile zips (one zip per patient)
    python fid_score.py \\
        --real-zips /path/to/tile_zips \\
        --gen-dir /path/to/generated/samples \\
        --output fid_results.json
"""

import argparse
import json
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, cast

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

# Try to import scipy for matrix square root
try:
    from scipy import linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Install with: pip install scipy")


# ----------------------------------------------------------------------
#   Inception Feature Extractor
# ----------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """
    Extract features from the penultimate layer of InceptionV3.
    Output dimension: 2048
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        
        # Load pretrained InceptionV3
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = cast(nn.Linear, nn.Identity())  # Remove classification head
        self.model.eval()
        self.model.to(device)
        
        self.device = device
        
        # InceptionV3 expects 299x299 images, normalized
        self.transform = transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from a batch of images."""
        images = images.to(self.device)
        features = self.model(images)
        return features.cpu().numpy()


# ----------------------------------------------------------------------
#   Datasets
# ----------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Load images from a directory (supports nested folders)."""
    
    def __init__(self, root_dir: str, transform=None, max_images: Optional[int] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Find all images
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            self.image_paths.extend(self.root_dir.rglob(ext))
        
        self.image_paths = sorted(self.image_paths)
        
        if max_images and len(self.image_paths) > max_images:
            # Random sample
            np.random.seed(42)
            indices = np.random.choice(len(self.image_paths), max_images, replace=False)
            self.image_paths = [self.image_paths[i] for i in sorted(indices)]
        
        print(f"[ImageFolderDataset] Found {len(self.image_paths)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class ZipTileDataset(Dataset):
    """Load tiles from zip files (one or more zips)."""
    
    def __init__(self, zip_dir: str, transform=None, 
                 max_images: Optional[int] = None,
                 tiles_per_zip: int = 10):
        self.zip_dir = Path(zip_dir)
        self.transform = transform
        self.tiles_per_zip = tiles_per_zip
        
        # Find all zips
        self.zip_files = sorted(self.zip_dir.glob("*.zip"))
        print(f"[ZipTileDataset] Found {len(self.zip_files)} zip files")
        
        # Index tiles from each zip
        self.tile_index = []  # (zip_path, tile_name)
        
        for zip_path in tqdm(self.zip_files, desc="Indexing zips"):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    tiles = [n for n in zf.namelist() 
                             if n.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    # Sample tiles from this zip
                    if len(tiles) > tiles_per_zip:
                        np.random.seed(hash(zip_path.name) % 2**32)
                        tiles = list(np.random.choice(tiles, tiles_per_zip, replace=False))
                    
                    for tile in tiles:
                        self.tile_index.append((zip_path, tile))
            except zipfile.BadZipFile:
                print(f"Warning: Skipping bad zip: {zip_path}")
        
        if max_images and len(self.tile_index) > max_images:
            np.random.seed(42)
            indices = np.random.choice(len(self.tile_index), max_images, replace=False)
            self.tile_index = [self.tile_index[i] for i in sorted(indices)]
        
        print(f"[ZipTileDataset] Total tiles indexed: {len(self.tile_index)}")
    
    def __len__(self):
        return len(self.tile_index)
    
    def __getitem__(self, idx):
        zip_path, tile_name = self.tile_index[idx]
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(tile_name) as f:
                img = Image.open(BytesIO(f.read())).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        return img


# ----------------------------------------------------------------------
#   FID Computation
# ----------------------------------------------------------------------

def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of features."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(mu1: np.ndarray, sigma1: np.ndarray, 
                mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Compute the Fréchet Inception Distance.
    
    FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2*sqrt(Σ1*Σ2))
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for FID computation. Install with: pip install scipy")
    
    # Mean difference
    diff = mu1 - mu2
    mean_diff = np.sum(diff ** 2)
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(np.diagonal(covmean)), 0, atol=1e-3):
            print("Warning: Complex values in matrix square root, taking real part")
        covmean = covmean.real
    
    # Trace term
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    fid = mean_diff + trace
    return float(fid)


def extract_features_from_loader(
    loader: DataLoader,
    extractor: InceptionFeatureExtractor,
    desc: str = "Extracting features"
) -> np.ndarray:
    """Extract Inception features from all images in a DataLoader."""
    all_features = []
    
    for batch in tqdm(loader, desc=desc):
        features = extractor.extract(batch)
        all_features.append(features)
    
    return np.concatenate(all_features, axis=0)


def compute_fid_between_dirs(
    real_source: str,
    gen_source: str,
    device: str = "cuda",
    batch_size: int = 32,
    max_images: Optional[int] = 10000,
    real_is_zip: bool = False,
    tiles_per_zip: int = 10,
) -> dict:
    """
    Compute FID between real and generated image sources.
    
    Args:
        real_source: Path to real images (directory or zip directory)
        gen_source: Path to generated images (directory)
        device: cuda or cpu
        batch_size: Batch size for feature extraction
        max_images: Maximum images to use (for speed)
        real_is_zip: If True, real_source contains zip files
        tiles_per_zip: Tiles to sample per zip file
    
    Returns:
        dict with FID score and statistics
    """
    print("\n" + "=" * 60)
    print("FID SCORE COMPUTATION")
    print("=" * 60)
    print(f"Real source: {real_source}")
    print(f"Generated source: {gen_source}")
    print(f"Device: {device}")
    print("=" * 60 + "\n")
    
    # Initialize feature extractor
    print("[1/4] Loading Inception model...")
    extractor = InceptionFeatureExtractor(device=device)
    
    # Create datasets
    print("[2/4] Loading datasets...")
    
    if real_is_zip:
        real_dataset = ZipTileDataset(
            real_source, 
            transform=extractor.transform,
            max_images=max_images,
            tiles_per_zip=tiles_per_zip,
        )
    else:
        real_dataset = ImageFolderDataset(
            real_source, 
            transform=extractor.transform,
            max_images=max_images,
        )
    
    gen_dataset = ImageFolderDataset(
        gen_source, 
        transform=extractor.transform,
        max_images=max_images,
    )
    
    real_loader = DataLoader(real_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4, pin_memory=True)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # Extract features
    print("[3/4] Extracting features...")
    real_features = extract_features_from_loader(real_loader, extractor, "Real images")
    gen_features = extract_features_from_loader(gen_loader, extractor, "Generated images")
    
    print(f"  Real features shape: {real_features.shape}")
    print(f"  Generated features shape: {gen_features.shape}")
    
    # Compute statistics
    print("[4/4] Computing FID...")
    mu_real, sigma_real = compute_statistics(real_features)
    mu_gen, sigma_gen = compute_statistics(gen_features)
    
    fid_score = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    
    print("\n" + "=" * 60)
    print(f"FID SCORE: {fid_score:.4f}")
    print("=" * 60)
    print("Interpretation:")
    print("  FID < 10:   Excellent (nearly indistinguishable)")
    print("  FID 10-50:  Good (minor differences)")
    print("  FID 50-100: Fair (noticeable differences)")
    print("  FID > 100:  Poor (significant differences)")
    print("=" * 60 + "\n")
    
    results = {
        "fid_score": fid_score,
        "n_real_images": len(real_dataset),
        "n_gen_images": len(gen_dataset),
        "real_source": str(real_source),
        "gen_source": str(gen_source),
        "feature_dim": real_features.shape[1],
    }
    
    return results


# ----------------------------------------------------------------------
#   Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute FID score between real and generated images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input sources
    parser.add_argument("--real-dir", type=str,
                        help="Directory containing real images")
    parser.add_argument("--real-zips", type=str,
                        help="Directory containing real tile zip files")
    parser.add_argument("--gen-dir", type=str, required=True,
                        help="Directory containing generated images")
    
    # Options
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--max-images", type=int, default=10000,
                        help="Maximum images to use (for speed)")
    parser.add_argument("--tiles-per-zip", type=int, default=10,
                        help="Tiles to sample per zip file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output", "-o", type=str,
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.real_dir and not args.real_zips:
        parser.error("Either --real-dir or --real-zips is required")
    
    real_source = args.real_dir or args.real_zips
    real_is_zip = args.real_zips is not None
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Compute FID
    results = compute_fid_between_dirs(
        real_source=real_source,
        gen_source=args.gen_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_images=args.max_images,
        real_is_zip=real_is_zip,
        tiles_per_zip=args.tiles_per_zip,
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()
