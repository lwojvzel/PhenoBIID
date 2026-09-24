"""Unmodified definitions extracted from the registered source snapshot."""
import warnings
import torch
from torch import nn
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

class NeuralReadout(nn.Module):
    def __init__(self, head, dimensions, bins):
        super().__init__()
        if head != 'tabm' or len(bins) != dimensions:
            raise ValueError('Expected TabM and one bin array per numerical feature')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The .*feature has just two bin edges')
            embedding = PiecewiseLinearEmbeddings(bins, 12, activation=False, version='B')
        self.network = TabM.make(n_num_features=dimensions, d_out=1, k=16,
            n_blocks=3, d_block=128, dropout=.1, num_embeddings=embedding)

    def forward(self, x):
        return self.network(x)[..., 0]
