# -*- coding: utf-8 -*-
"""
Variational Autoencoder (VAE) with MMD regularization.

This VAE uses Maximum Mean Discrepancy instead of KL divergence
to regularize the latent space, which can be more stable for
high-dimensional genomic data.

Original author: Hakim Benkirane (CentraleSupelec, MICS laboratory)
"""

import torch
import torch.nn as nn

from .loss import compute_mmd


class VAE(nn.Module):
    """
    Variational Autoencoder with MMD regularization.
    
    Maps high-dimensional input (e.g., gene expression) to a compact
    latent representation while preserving reconstruction ability.
    """
    
    def __init__(self, encoder, decoder, device):
        """
        Parameters
        ----------
        encoder : ProbabilisticEncoder
            Encoder network
        decoder : ProbabilisticDecoder
            Decoder network
        device : torch.device
            Device for computation (cuda/cpu)
        """
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._relocate()

    def _relocate(self):
        """Move encoder and decoder to the specified device."""
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
    def reparameterization(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Parameters
        ----------
        mean : torch.Tensor
            Mean of latent distribution
        var : torch.Tensor
            Standard deviation of latent distribution
        
        Returns
        -------
        torch.Tensor
            Sampled latent vector
        """
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass: encode, sample, decode.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed input
        z : torch.Tensor
            Sampled latent representation
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, z

    def loss_components(self, x: torch.Tensor, beta: float = 1.0):
        """
        Compute total loss and individual components.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        beta : float
            Weight for MMD regularization term
        
        Returns
        -------
        total : torch.Tensor
            Total loss (reconstruction + beta * MMD)
        recon : torch.Tensor
            Reconstruction loss (MSE)
        mmd : torch.Tensor
            MMD regularization loss
        """
        x_hat, z = self.forward(x)

        # Reconstruction loss
        reconstruction_loss = nn.MSELoss()
        recon = reconstruction_loss(x, x_hat)

        # MMD against standard Gaussian prior
        true_samples = torch.randn(z.shape[0], z.shape[1]).to(self.device)
        mmd = torch.sum(compute_mmd(true_samples, z))

        total = recon + beta * mmd
        return total, recon, mmd

    def loss(self, x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Compute total loss."""
        total, _, _ = self.loss_components(x, beta)
        return total
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation (mean only, no sampling).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Latent mean vector
        """
        self.eval()
        with torch.no_grad():
            mean, _ = self.encoder(x)
        return mean
