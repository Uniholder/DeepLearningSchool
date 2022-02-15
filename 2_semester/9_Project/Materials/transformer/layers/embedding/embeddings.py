import numpy as np
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, name, **kwargs):
        super(Embeddings, self).__init__()

        if name + "_embeddings_path" in kwargs:
            embeddings = np.load(kwargs[name + "_embeddings_path"])
            self.v_s, self.h_s = embeddings.shape

            self.embeddings = nn.Embedding(self.v_s, self.h_s, padding_idx=0)
            self.embeddings.weight = nn.Parameter(
                torch.from_numpy(embeddings).float(), requires_grad=False
            )
        elif name + "_embeddings_size" in kwargs:
            self.v_s, self.h_s = kwargs[name + "_embeddings_size"]
            self.embeddings = nn.Embedding(self.v_s, self.h_s, padding_idx=0)
        else:
            raise NameError(
                '"embeddings_path" or "embeddings_size" have to be provided'
            )

    def forward(self, input):

        b_s, s_l = input.size()

        mask = torch.eq(input, 0).float()
        res = self.embeddings(input)

        if not res.requires_grad:
            res.requires_grad = True
        mask.requires_grad = True

        return res, mask
