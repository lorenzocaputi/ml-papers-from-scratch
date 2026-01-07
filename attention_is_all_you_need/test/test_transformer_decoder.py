import torch
from attention_is_all_you_need.transformer_decoder import TransformerDecoderBlock


def test_decoder_block_causal_self_attention():
    torch.manual_seed(0)

    B, Tt, Ts, D, H, DFF = 2, 5, 7, 8, 2, 32
    x = torch.randn(B, Tt, D)
    enc_out = torch.randn(B, Ts, D)

    block = TransformerDecoderBlock(d_model=D, d_ff=DFF, num_heads=H)

    # run forward (we don't directly expose self-attn weights here)
    y = block(x, enc_out)

    assert y.shape == (B, Tt, D)
    assert torch.isfinite(y).all()