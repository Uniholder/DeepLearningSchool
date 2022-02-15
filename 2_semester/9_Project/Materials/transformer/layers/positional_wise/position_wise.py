import torch.nn as nn

from ..checkpoint import checkpoint
from ..gelu import GELU


class PositionWise(nn.Module):
    def __init__(self, size, inner_size, dropout=0.1):
        super(PositionWise, self).__init__()

        self.fc = nn.Sequential(
            nn.LayerNorm(size, eps=1e-12),
            nn.Linear(size, inner_size),
            GELU(),
            nn.Linear(inner_size, size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, checkpoint_gradients=False):
        if checkpoint_gradients:
            return input + self.dropout(
                checkpoint(self.fc, 1, *(tuple(input) + tuple(self.parameters())))
            )
        else:
            return input + self.dropout(self.fc(input))
