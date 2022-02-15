import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, clipping_distance, dropout=0.1, **kwargs):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            "encoder", n_layers, n_heads, clipping_distance, dropout, **kwargs
        )
        self.decoder = Decoder(
            "decoder", n_layers, n_heads, clipping_distance, dropout, **kwargs
        )

    def forward(self, input, decoder_input=None, **kwargs):

        if decoder_input is not None:
            input, mask = self.encoder(input, **kwargs)
            return self.decoder(decoder_input, input, mask, **kwargs)
        else:
            with torch.no_grad():
                input, mask = self.encoder(input)
                return self.decoder.generate(input, **kwargs)
