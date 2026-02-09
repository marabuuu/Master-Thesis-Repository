# -*- coding: utf-8 -*-
"""
Probabilistic Decoder for Variational Autoencoder.

Original author: Hakim Benkirane (CentraleSupelec, MICS laboratory)
"""

import torch.nn as nn
from collections import OrderedDict
from .layers import FullyConnectedLayer


class ProbabilisticDecoder(nn.Module):
    """
    Decoder network that maps latent representation back to input space.
    Used as the generative network in a VAE.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: list,
        output_dim: int,
        norm_layer=nn.BatchNorm1d,
        leaky_slope: float = 0.2,
        dropout: float = 0,
        debug: bool = False
    ):
        """
        Parameters
        ----------
        latent_dim : int
            Dimension of the latent representation
        hidden_dim : list
            List of dimensions for hidden layers
        output_dim : int
            Dimension of the output (e.g., number of genes)
        norm_layer : nn.Module
            Normalization layer class
        leaky_slope : float
            Coefficient for LeakyReLU (0 to 1)
        dropout : float
            Dropout rate (0 to 1)
        debug : bool
            If True, print intermediate tensor shapes
        """
        super(ProbabilisticDecoder, self).__init__()

        self.dt_layers = OrderedDict()

        self.dt_layers['InputLayer'] = FullyConnectedLayer(
            latent_dim, hidden_dim[0],
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout=dropout,
            activation=True
        )

        block_layer_num = len(hidden_dim)
        dropout_flag = True
        for num in range(1, block_layer_num):
            self.dt_layers[f'Layer{num}'] = FullyConnectedLayer(
                hidden_dim[num - 1], hidden_dim[num],
                norm_layer=norm_layer,
                leaky_slope=leaky_slope,
                dropout=dropout_flag * dropout,
                activation=True
            )
            # dropout for every other layer
            dropout_flag = not dropout_flag

        # Output layer (no activation - raw values for reconstruction)
        self.dt_layers['OutputLayer'] = FullyConnectedLayer(
            hidden_dim[-1], output_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout=0,
            activation=False,
            normalization=False
        )

        self.net = nn.Sequential(self.dt_layers)
        
    def forward(self, x):
        """
        Forward pass through decoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Latent tensor of shape (batch_size, latent_dim)
        
        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed tensor of shape (batch_size, output_dim)
        """
        x_hat = self.net(x)
        return x_hat
