#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image Quality Metrics for Reconstruction Evaluation

This module implements standard image quality metrics:
- MSE (Mean Squared Error): Average squared pixel differences
- PSNR (Peak Signal-to-Noise Ratio): Signal quality in decibels
- SSIM (Structural Similarity Index): Perceptual similarity

All metrics can operate on:
- PIL Images
- NumPy arrays (H, W, C) or (H, W) for grayscale
- PyTorch tensors (C, H, W) or (B, C, H, W)
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# Type alias for image inputs
ImageType = Union[Image.Image, np.ndarray, "torch.Tensor"]


def _to_numpy(img: ImageType) -> np.ndarray:
    """
    Convert various image formats to numpy array (H, W, C) in range [0, 255].
    
    Args:
        img: PIL Image, numpy array, or torch tensor
        
    Returns:
        Numpy array of shape (H, W, C) with values in [0, 255]
    """
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
    elif HAS_TORCH and isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        # Handle batch dimension
        if arr.ndim == 4:
            arr = arr[0]
        # Handle (C, H, W) -> (H, W, C)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        # Handle [-1, 1] normalization
        if arr.min() < 0:
            arr = (arr + 1) / 2
        # Handle [0, 1] normalization
        if arr.max() <= 1.0:
            arr = arr * 255
        arr = arr.astype(np.uint8)
    elif isinstance(img, np.ndarray):
        arr = img.copy()
        # Handle (C, H, W) -> (H, W, C)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        # Handle [-1, 1] normalization
        if arr.min() < 0:
            arr = (arr + 1) / 2
        # Handle [0, 1] normalization
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    # Ensure RGB
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]  # Remove alpha channel
    
    return arr


def compute_mse(
    original: ImageType,
    reconstructed: ImageType,
    per_channel: bool = False,
) -> Union[float, np.ndarray]:
    """
    Compute Mean Squared Error between original and reconstructed images.
    
    MSE = (1/N) * sum((original - reconstructed)^2)
    
    Lower values indicate better reconstruction (0 = perfect).
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        per_channel: If True, return MSE per RGB channel
        
    Returns:
        MSE value (float) or array of per-channel MSE values
    """
    orig = _to_numpy(original).astype(np.float64)
    recon = _to_numpy(reconstructed).astype(np.float64)
    
    if orig.shape != recon.shape:
        raise ValueError(
            f"Image shapes don't match: {orig.shape} vs {recon.shape}"
        )
    
    diff = (orig - recon) ** 2
    
    if per_channel:
        return np.mean(diff, axis=(0, 1))
    else:
        return float(np.mean(diff))


def compute_psnr(
    original: ImageType,
    reconstructed: ImageType,
    max_pixel: float = 255.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between original and reconstructed images.
    
    PSNR = 10 * log10(MAX^2 / MSE) = 20 * log10(MAX / sqrt(MSE))
    
    Higher values indicate better reconstruction (infinity = perfect).
    Typical values:
        - >40 dB: Excellent quality, imperceptible distortion
        - 30-40 dB: Good quality, minor distortion
        - 20-30 dB: Moderate quality, visible distortion
        - <20 dB: Poor quality, significant distortion
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        max_pixel: Maximum pixel value (255 for 8-bit images)
        
    Returns:
        PSNR value in decibels (dB)
    """
    mse = compute_mse(original, reconstructed)
    
    if mse == 0:
        return float("inf")
    
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return float(psnr)


def compute_ssim(
    original: ImageType,
    reconstructed: ImageType,
    win_size: int = 7,
    multichannel: bool = True,
    data_range: Optional[int] = None,
) -> float:
    """
    Compute Structural Similarity Index Measure between images.
    
    SSIM measures perceptual similarity based on:
    - Luminance: mean intensity comparison
    - Contrast: standard deviation comparison  
    - Structure: correlation of normalized signals
    
    Values range from -1 to 1:
        - 1.0: Perfect structural similarity
        - 0.0: No structural similarity
        - <0: Negative correlation (inverted structure)
    
    Typical interpretation:
        - >0.95: Excellent similarity
        - 0.85-0.95: Good similarity
        - 0.70-0.85: Moderate similarity
        - <0.70: Poor similarity
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        win_size: Window size for local statistics (should be odd, default 7)
        multichannel: If True, compute SSIM for each channel and average
        data_range: The data range of the input images. If None, uses 255.
        
    Returns:
        SSIM value in range [-1, 1]
    """
    orig = _to_numpy(original)
    recon = _to_numpy(reconstructed)
    
    if orig.shape != recon.shape:
        raise ValueError(
            f"Image shapes don't match: {orig.shape} vs {recon.shape}"
        )
    
    if data_range is None:
        data_range = 255
    
    if HAS_SKIMAGE:
        # Use scikit-image implementation
        ssim_value = skimage_ssim(
            orig,
            recon,
            win_size=win_size,
            channel_axis=2 if multichannel else None,
            data_range=data_range,
        )
        return float(ssim_value)
    else:
        # Fallback: simplified SSIM implementation
        return _compute_ssim_simple(orig, recon, win_size, data_range)


def _compute_ssim_simple(
    orig: np.ndarray,
    recon: np.ndarray,
    win_size: int = 7,
    data_range: int = 255,
) -> float:
    """
    Simplified SSIM implementation for when scikit-image is not available.
    
    Uses the formula:
    SSIM(x,y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) / 
                ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
    
    Where C1 = (K1*L)^2, C2 = (K2*L)^2, K1=0.01, K2=0.03, L=data_range
    """
    from scipy.ndimage import uniform_filter
    
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    orig = orig.astype(np.float64)
    recon = recon.astype(np.float64)
    
    # Compute means
    mu_orig = uniform_filter(orig, size=win_size)
    mu_recon = uniform_filter(recon, size=win_size)
    
    # Compute variances and covariance
    mu_orig_sq = mu_orig ** 2
    mu_recon_sq = mu_recon ** 2
    mu_orig_recon = mu_orig * mu_recon
    
    sigma_orig_sq = uniform_filter(orig ** 2, size=win_size) - mu_orig_sq
    sigma_recon_sq = uniform_filter(recon ** 2, size=win_size) - mu_recon_sq
    sigma_orig_recon = uniform_filter(orig * recon, size=win_size) - mu_orig_recon
    
    # SSIM formula
    numerator = (2 * mu_orig_recon + C1) * (2 * sigma_orig_recon + C2)
    denominator = (mu_orig_sq + mu_recon_sq + C1) * (sigma_orig_sq + sigma_recon_sq + C2)
    
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_all_metrics(
    original: ImageType,
    reconstructed: ImageType,
    include_per_channel: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute all available quality metrics between original and reconstructed images.
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        include_per_channel: If True, include per-channel MSE values
        
    Returns:
        Dictionary with keys:
            - 'mse': Mean Squared Error
            - 'psnr': Peak Signal-to-Noise Ratio (dB)
            - 'ssim': Structural Similarity Index
            - 'mse_per_channel': Per-channel MSE (if include_per_channel=True)
    """
    results = {
        "mse": compute_mse(original, reconstructed),
        "psnr": compute_psnr(original, reconstructed),
        "ssim": compute_ssim(original, reconstructed),
    }
    
    if include_per_channel:
        results["mse_per_channel"] = compute_mse(
            original, reconstructed, per_channel=True
        )
    
    return results


def compute_batch_metrics(
    originals: list,
    reconstructeds: list,
) -> Dict[str, np.ndarray]:
    """
    Compute metrics for a batch of image pairs.
    
    Args:
        originals: List of original images
        reconstructeds: List of reconstructed images
        
    Returns:
        Dictionary with arrays of metric values for each pair
    """
    if len(originals) != len(reconstructeds):
        raise ValueError(
            f"Batch sizes don't match: {len(originals)} vs {len(reconstructeds)}"
        )
    
    mse_values = []
    psnr_values = []
    ssim_values = []
    
    for orig, recon in zip(originals, reconstructeds):
        metrics = compute_all_metrics(orig, recon)
        mse_values.append(metrics["mse"])
        psnr_values.append(metrics["psnr"])
        ssim_values.append(metrics["ssim"])
    
    return {
        "mse": np.array(mse_values),
        "psnr": np.array(psnr_values),
        "ssim": np.array(ssim_values),
    }


def summarize_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics for batch metrics.
    
    Args:
        metrics: Dictionary of metric arrays from compute_batch_metrics
        
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    summary = {}
    for name, values in metrics.items():
        values = np.array(values)
        # Filter out infinities for PSNR
        finite_values = values[np.isfinite(values)]
        
        summary[name] = {
            "mean": float(np.mean(finite_values)) if len(finite_values) > 0 else float("nan"),
            "std": float(np.std(finite_values)) if len(finite_values) > 0 else float("nan"),
            "min": float(np.min(finite_values)) if len(finite_values) > 0 else float("nan"),
            "max": float(np.max(finite_values)) if len(finite_values) > 0 else float("nan"),
            "count": len(values),
            "finite_count": len(finite_values),
        }
    
    return summary
