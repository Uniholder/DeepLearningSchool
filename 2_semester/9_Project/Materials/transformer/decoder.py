from math import sqrt

import torch
import torch.nn as nn

from layers.attention import MultiHeadAttention
from layers.embedding.embeddings import Embeddings
from layers.positional_wise.position_wise import PositionWise
from other import Beam


class Decoder(nn.Module):
    def __init__(
        self, name, n_layers, n_heads, clipping_distance, dropout=0.1, **kwargs
    ):
        super(Decoder, self).__init__()

        self.embeddings = Embeddings(name, **kwargs)
        self.h_s = self.embeddings.h_s

        self.scale = 1 / (sqrt(3 * n_layers))
        self.layers = nn.ModuleList(
            [
                DecoderLayer(n_heads, self.h_s, clipping_distance, dropout)
                for _ in range(n_layers)
            ]
        )

        self.out = nn.Sequential(
            nn.LayerNorm(self.h_s, eps=1e-8), nn.Linear(self.h_s, self.embeddings.v_s)
        )

        with torch.no_grad():
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if layer.weight.size(0) != self.embeddings.v_s:
                    if layer.bias is not None:
                        layer.bias.zero_()
                    layer.weight.normal_(0, 0.125 / sqrt(layer.weight.size(-1)))
                else:
                    layer.weight.zero_()

        for layer in self.layers:
            layer.inner_attention.fc.weight.mul_(self.scale)
            layer.outer_attention.fc.weight.mul_(self.scale)
            layer.position_wise.fc[-1].weight.mul_(self.scale)

    def forward(self, input, encoder_out, encoder_mask, **kwargs):
        input, _ = self.embeddings(input)
        mask = self._autoregressive_mask(input)

        encoder_mask = encoder_mask.unsqueeze(1).repeat(1, input.size(1), 1)

        for layer in self.layers:
            input = layer(input, encoder_out, mask, encoder_mask, **kwargs)

        return self.out(input)

    def generate(self, encoder_out, n_beams=5, n_generations=4):

        device = encoder_out.device

        # 1 is go token idx
        input = torch.tensor([[1]], dtype=torch.long, device=device)
        input, _ = self.embeddings(input)

        cache = []

        for layer in self.layers:
            cache += [input]
            input = layer(input, encoder_out)

        out = torch.softmax(self.out(input).squeeze(), dim=-1)
        beams = Beam.start_search(out, n_beams, cache)

        encoder_out = encoder_out.repeat(n_beams, 1, 1)

        ended_beams = []

        for step in range(encoder_out.size(1) * 100):

            _beams = []
            any_beam_ended = False
            for beam in beams:
                if beam.ended:
                    ended_beams += [beam]
                    any_beam_ended = True
                else:
                    _beams += [beam]
            beams = _beams

            if len(ended_beams) >= n_generations:
                break

            if any_beam_ended:
                encoder_out = encoder_out[0].unsqueeze(0).repeat(len(beams), 1, 1)

            input = torch.tensor(
                [[beam.data[-1]] for beam in beams], dtype=torch.long, device=device
            )

            input, _, = self.embeddings(input)

            for i, layer in enumerate(self.layers):

                for j, beam in enumerate(beams):
                    beam.cache[i] = torch.cat([beam.cache[i], input[j].unsqueeze(0)], 1)
                cache = torch.cat([beam.cache[i] for beam in beams], 0)

                input = layer(
                    (input, cache),
                    encoder_out,
                    q_indices_shift=torch.tensor([step + 1], device=input.device),
                )

            out = torch.softmax(self.out(input).squeeze(), dim=-1)

            beams = Beam.update(beams, out)

        if len(ended_beams) >= n_generations:
            return [[int(i) for i in b.data] for b in ended_beams[:n_generations]]
        else:
            return [[int(i) for i in b.data] for b in ended_beams] + [
                [int(i) for i in b.data + [2]]
                for b in beams[: (n_generations - len(ended_beams))]
            ]

    @staticmethod
    def _autoregressive_mask(input):
        b_s, s_l, *_ = input.size()
        mask = torch.ones(s_l, s_l, device=input.device).float().tril_(-1)
        mask.requires_grad = True
        return mask.transpose(0, 1).repeat(b_s, 1).view(b_s, s_l, s_l)


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, h_s, clipping_distance, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.inner_attention = MultiHeadAttention(
            n_heads, h_s, dropout, clipping_distance
        )
        self.outer_attention = MultiHeadAttention(n_heads, h_s, dropout)

        self.position_wise = PositionWise(h_s, h_s * 4, dropout)

    def forward(
        self,
        inner_input,
        outer_input,
        inner_mask=None,
        outer_mask=None,
        q_indices_shift=None,
        **kwargs
    ):

        if isinstance(inner_input, tuple):
            q, inner_input = inner_input
        else:
            q = inner_input

        result = self.inner_attention(
            q, inner_input, inner_input, inner_mask, q_indices_shift, **kwargs
        )
        result = self.outer_attention(
            result, outer_input, outer_input, outer_mask, **kwargs
        )
        return self.position_wise(result, **kwargs)
