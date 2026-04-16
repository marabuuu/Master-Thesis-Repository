# -*- coding: utf-8 -*-
"""
Maximum Mean Discrepancy (MMD) Loss for VAE training.

MMD is used as a regularizer in the latent space to encourage the 
learned distribution to match a prior (typically standard Gaussian).

Original author: Hakim Benkirane (CentraleSupelec, MICS laboratory)
"""

import torch
import torch.nn as nn


def compute_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel between two sets of samples.
    
    Parameters
    ----------
    x : torch.Tensor
        First set of samples (x_size, dim)
    y : torch.Tensor
        Second set of samples (y_size, dim)
    
    Returns
    -------
    torch.Tensor
        Kernel matrix of shape (x_size, y_size)
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy between two distributions.
    
    MMD measures the distance between the empirical distributions of x and y
    using an RBF kernel.
    
    Parameters
    ----------
    x : torch.Tensor
        First set of samples (batch_size, dim)
    y : torch.Tensor
        Second set of samples (batch_size, dim)
    
    Returns
    -------
    torch.Tensor
        Scalar MMD value
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


class MMDLoss(nn.Module):
    """
    MMD Loss module for use in VAE training.
    
    Encourages latent distribution to match a prior distribution.
    """
    
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self, x_hat, x, mean=None, log_var=None):
        """
        Compute MMD loss.
        
        Note: mean and log_var are kept for API compatibility but not used.
        """
        return compute_mmd(x_hat, x)
