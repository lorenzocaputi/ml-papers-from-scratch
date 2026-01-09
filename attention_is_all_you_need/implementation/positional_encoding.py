import torch
import math
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # create the zeros PE matrix
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(-math.log(10000) * torch.arange(0, d_model, 2) / d_model) # (d_model/2,)

        # fill pe with broadcasting
        pe[:, 0::2] = torch.sin(pos * div_term) # even indices
        pe[:, 1::2] = torch.cos(pos * div_term) # odd indices

        # reshape to (1, max_len, d_model) for broadcast over batches
        pe = pe.unsqueeze(0)

        # register as a buffer (not as a parameter, moves with the device)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Applies positional encoding to a tensor x

        Args:
            x: (B, T, D)
        Returns:
            x + positional_encoding: (B, T, D)
        """
        T = x.shape[1]
        return x + self.pe[:, :T, :] # "all batches, sequence elements up to T, all dimensions"


