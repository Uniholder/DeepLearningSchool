import torch
import torch.nn as nn

from .scaled_dot_product import ScaledDotProductAttention
from ..checkpoint import checkpoint


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, h_s, dropout=0.1, clipping_distance=None):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.h_s = h_s
        self.p_s = h_s // n_heads

        self.q = nn.Sequential(
            nn.LayerNorm(h_s), nn.Linear(h_s, self.p_s * self.n_heads, bias=False)
        )
        self.k = nn.Sequential(
            nn.LayerNorm(h_s), nn.Linear(h_s, self.p_s * self.n_heads, bias=False)
        )
        self.v = nn.Sequential(
            nn.LayerNorm(h_s), nn.Linear(h_s, self.p_s * self.n_heads, bias=False)
        )

        self.attention = ScaledDotProductAttention(self.p_s, clipping_distance)

        self.fc = nn.Linear(self.n_heads * self.p_s, h_s)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q, k, v, mask=None, q_indices_shift=None, checkpoint_gradients=False
    ):
        # yet anothoer checkpoint gradients workaround
        if q_indices_shift is None:
            q_indices_shift = torch.tensor([0.0], requires_grad=True, device=q.device)

        if checkpoint_gradients:
            return q + self.dropout(
                checkpoint(
                    self.call,
                    5,
                    *(
                        tuple([q, k, v, mask, q_indices_shift])
                        + tuple(self.parameters())
                    )
                )
            )
        else:
            return q + self.dropout(self.call(q, k, v, mask, q_indices_shift))

    def call(self, q, k, v, mask=None, q_indices_shift=None):
        b_s, q_l, _ = q.size()
        _, k_l, _ = k.size()

        q = self.split_heads(self.q(q))
        k = self.split_heads(self.k(k))
        v = self.split_heads(self.v(v))

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).view(-1, q_l, k_l)

        result = self.attention(q, k, v, mask, q_indices_shift)

        result = self.join_heads(result)

        return self.fc(result)

    def split_heads(self, input):
        b_s, s_l, _ = input.size()
        return (
            input.view(b_s, s_l, self.n_heads, self.p_s)
            .transpose(1, 2)
            .contiguous()
            .view(-1, s_l, self.p_s)
        )

    def join_heads(self, input):
        _, s_l, p_s = input.size()
        return (
            input.view(-1, self.n_heads, s_l, p_s)
            .transpose(1, 2)
            .contiguous()
            .view(-1, s_l, self.n_heads * p_s)
        )
