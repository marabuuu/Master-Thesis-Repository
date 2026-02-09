# -*- coding: utf-8 -*-
"""
Probabilistic Encoder for Variational Autoencoder.

Original author: Hakim Benkirane (CentraleSupelec, MICS laboratory)
"""

import torch.nn as nn
from collections import OrderedDict
from .layers import FullyConnectedLayer


class ProbabilisticEncoder(nn.Module):
    """
    Encoder network that maps input to a latent distribution (mean, log_var).
    Used as the inference network in a VAE.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        latent_dim: int,
        norm_layer=nn.BatchNorm1d,
        leaky_slope: float = 0.2,
        dropout: float = 0,
        debug: bool = False
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of the input tensor (e.g., number of genes)
        hidden_dim : list
            List of dimensions for hidden layers
        latent_dim : int
            Dimension of the latent representation
        norm_layer : nn.Module
            Normalization layer class
        leaky_slope : float
            Coefficient for LeakyReLU (0 to 1)
        dropout : float
            Dropout rate (0 to 1)
        debug : bool
            If True, print intermediate tensor shapes
        """
        super(ProbabilisticEncoder, self).__init__()
        
        self.dt_layers = OrderedDict()

        self.dt_layers['InputLayer'] = FullyConnectedLayer(
            input_dim, hidden_dim[0],
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

        self.net = nn.Sequential(self.dt_layers)

        # Output layers for mean and log variance
        self.mean_layer = FullyConnectedLayer(
            hidden_dim[-1], latent_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout=0,
            activation=False,
            normalization=False
        )
        self.log_var_layer = FullyConnectedLayer(
            hidden_dim[-1], latent_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout=0,
            activation=False,
            normalization=False
        )
        
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        mean : torch.Tensor
            Mean of latent distribution (batch_size, latent_dim)
        log_var : torch.Tensor
            Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.net(x)
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        return mean, log_var
