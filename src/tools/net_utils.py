# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Sets-up the different types of layers that can be used.
"""

import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0.2, activation=True, normalization=True, activation_name='LeakyReLU'):
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
                    raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

    def forward(self, x):
        y = self.fc_block(x)
        return y