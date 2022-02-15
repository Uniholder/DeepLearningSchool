from math import sqrt

import torch
import torch.nn as nn

from layers.attention import MultiHeadAttention
from layers.embedding.embeddings import Embeddings
from layers.positional_wise.position_wise import PositionWise


class Encoder(nn.Module):
    def __init__(
        self, name, n_layers, n_heads, clipping_distance, dropout=0.1, **kwargs
    ):
        super(Encoder, self).__init__()

        self.embeddings = Embeddings(name, **kwargs)
        self.h_s = self.embeddings.h_s

        self.scale = 1 / (sqrt(2 * n_layers))
        self.layers = nn.ModuleList(
            [
                EncoderLayer(n_heads, self.h_s, clipping_distance, dropout)
                for _ in range(n_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(self.h_s)

        with torch.no_grad():
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if layer.bias is not None:
                    layer.bias.zero_()
                layer.weight.normal_(0, 0.125 / sqrt(layer.weight.size(-1)))

        for layer in self.layers:
            layer.attention.fc.weight.mul_(self.scale)
            layer.position_wise.fc[-1].weight.mul_(self.scale)

    def forward(self, input, **kwargs):
        input, mask = self.embeddings(input)

        _mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)

        for layer in self.layers:
            input = layer(input, _mask, **kwargs)

        return self.layer_norm(input), mask


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, h_s, clipping_distance, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(n_heads, h_s, dropout, clipping_distance)

        self.position_wise = PositionWise(h_s, h_s * 4, dropout)

    def forward(self, input, mask=None, **kwargs):
        result = self.attention(input, input, input, mask, **kwargs)
        return self.position_wise(result, **kwargs)
