# -*- coding: utf-8 -*-
"""
Layer utilities for building neural network architectures.

Original author: Hakim Benkirane (CentraleSupelec, MICS laboratory)
"""

import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    """
    Configurable fully connected layer block: Linear => Norm1D => Activation => Dropout
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        norm_layer=nn.BatchNorm1d,
        leaky_slope: float = 0.2,
        dropout: float = 0.2,
        activation: bool = True,
        normalization: bool = True,
        activation_name: str = 'LeakyReLU'
    ):
        """
        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        norm_layer : nn.Module
            Normalization layer class (default: BatchNorm1d)
        leaky_slope : float
            Negative slope for LeakyReLU (default: 0.2)
        dropout : float
            Dropout probability (default: 0.2)
        activation : bool
            Whether to apply activation (default: True)
        normalization : bool
            Whether to apply normalization (default: True)
        activation_name : str
            Name of activation function (default: 'LeakyReLU')
        """
        super(FullyConnectedLayer, self).__init__()
        self.fc_block = nn.Sequential()
        
        # Linear
        self.fc_block.add_module('linear', nn.Linear(input_dim, output_dim))
        
        # Norm
        if normalization:
            self.fc_block.add_module('norm', norm_layer(output_dim))
        
        # Dropout
        if 0 < dropout <= 1:
            self.fc_block.add_module('dropout', nn.Dropout(p=dropout))
        
        # Activation
        if activation:
            activation_layer = {
                'relu': nn.ReLU(),
                'sigmoid': nn.Sigmoid(),
                'leakyrelu': nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
                'tanh': nn.Tanh(),
                'softmax': nn.Softmax(dim=1)
            }.get(activation_name.lower(), None)
            
            if activation_layer is not None:
                self.fc_block.add_module(activation_name.lower(), activation_layer)
            else:
                if activation_name.lower() != 'no':
                    raise NotImplementedError(f'Activation function [{activation_name}] is not implemented')

    def forward(self, x):
        return self.fc_block(x)
