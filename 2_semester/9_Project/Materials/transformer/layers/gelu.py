import math

import torch
import torch.jit as jit


class GELU(jit.ScriptModule):
    __constants__ = ["sqrt"]
    sqrt = math.sqrt(2 / math.pi)

    @jit.script_method
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(self.sqrt * (x + 0.044715 * torch.pow(x, 3))))
