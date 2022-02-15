from math import sqrt

import torch
import torch.nn as nn

from ..embedding.positional_embeddings import PositionalEmbeddings


class ScaledDotProductAttention(nn.Module):
    def __init__(self, size, clipping_distance=None):
        super(ScaledDotProductAttention, self).__init__()

        self.scaling = 1 / (sqrt(size))

        self.positional_embeddings = (
            PositionalEmbeddings(size, clipping_distance)
            if clipping_distance is not None
            else None
        )

    def forward(self, q, k, v, mask=None, q_indices_shift=None):

        positional_logits = None
        if self.positional_embeddings is not None:
            q_indices = torch.arange(0, q.size(1), device=q.device)
            k_indices = torch.arange(0, k.size(1), device=k.device)

            if q_indices_shift is not None:
                q_indices = q_indices + q_indices_shift.long()

            q_indices = q_indices.unsqueeze(-1).repeat(1, k.size(1))
            k_indices = k_indices.unsqueeze(0).repeat(q.size(1), 1)
            indices = k_indices - q_indices
            positional_embeddings = self.positional_embeddings(indices)

            positional_logits = torch.bmm(
                q.transpose(0, 1), positional_embeddings.transpose(1, 2)
            ).transpose(0, 1)

        logits = torch.bmm(q, k.transpose(1, 2))
        if positional_logits is not None:
            logits = logits + positional_logits
        logits = logits * self.scaling

        if mask is not None:
            logits.masked_fill_(mask.bool(), -float("inf"))

        attention = torch.softmax(logits, dim=2)

        return torch.bmm(attention, v)
