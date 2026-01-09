import torch
from attention_is_all_you_need.implementation.positional_encoding import SinusoidalPositionalEncoding


def test_sinusoidal_positional_encoding_shape_and_determinism():
    torch.manual_seed(0)

    B, T, D = 4, 5, 16
    x = torch.randn(B, T, D)

    spe = SinusoidalPositionalEncoding(D, max_len=50)
    y1 = spe(x)
    y2 = spe(x)

    # check shape
    assert y1.shape == (B, T, D)

    # check determinism
    assert torch.allclose(y1, y2)

    # check different positions have different encodings
    assert not torch.allclose(y1[:, 0, :], y1[:, 1, :])

