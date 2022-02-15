import torch
import torch.nn as nn


class PositionalEmbeddings(nn.Module):
    def __init__(self, size, clipping_distance=4):
        super(PositionalEmbeddings, self).__init__()

        self.k = clipping_distance

        self.embeddigns = nn.Embedding(2 * self.k + 1, size)

    def forward(self, input):
        input = torch.clamp(input, -self.k, self.k) + self.k
        return self.embeddigns(input)
