import torch
from attention_is_all_you_need.attention import scaled_dot_product_attention_v0, scaled_dot_product_attention_v1, scaled_dot_product_attention_v2
from attention_is_all_you_need.attention import MultiHeadSelfAttention, TransformerEncoderBlock


def test_scaled_dot_product_attention_v0():
    torch.manual_seed(0)

    T, D = 4, 6
    Q = torch.randn(T, D)
    K = torch.randn(T, D)
    V = torch.randn(T, D)

    output, attention_score = scaled_dot_product_attention_v0(Q, K, V)

    # check shapes
    assert output.size() == (T, D)
    assert attention_score.size() == (T, T)

    # attention rows should sum to one
    row_sums = attention_score.sum(dim=-1)
    ones = torch.ones(T, dtype=row_sums.dtype)
    assert torch.allclose(row_sums, ones, atol=1e-6, rtol=0.0)

    # each attention weight in [0,1]
    assert (attention_score >= 0).all()
    assert (attention_score <= 1).all()

    # finite output
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention_score).all()


def test_scaled_dot_product_attention_v1():
    torch.manual_seed(0)

    B, T, D = 4, 4, 6
    Q = torch.randn(B, T, D)
    K = torch.randn(B, T, D)
    V = torch.randn(B, T, D)

    output, attention_score = scaled_dot_product_attention_v1(Q, K, V)

    # check shapes
    assert output.size() == (B, T, D)
    assert attention_score.size() == (B, T, T)

    # attention rows should sum to one
    row_sums = attention_score.sum(dim=-1)
    ones = torch.ones(B, T, dtype=row_sums.dtype)
    assert torch.allclose(row_sums, ones, atol=1e-6, rtol=0.0)

    # each attention weight in [0,1]
    assert (attention_score >= 0).all()
    assert (attention_score <= 1).all()

    # finite output
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention_score).all()


def test_scaled_dot_product_attention_v2():
    torch.manual_seed(0)

    B, H, T, Dh = 3, 4, 5, 6
    Q = torch.randn(B, H, T, Dh)
    K = torch.randn(B, H, T, Dh)
    V = torch.randn(B, H, T, Dh)

    output, attention_score = scaled_dot_product_attention_v2(Q, K, V)

    # check shapes
    assert output.size() == (B, H, T, Dh)
    assert attention_score.size() == (B, H, T, T)

    # attention rows should sum to one
    row_sums = attention_score.sum(dim=-1)
    ones = torch.ones(B, H, T, dtype=row_sums.dtype)
    assert torch.allclose(row_sums, ones, atol=1e-6, rtol=0.0)

    # each attention weight in [0,1]
    assert (attention_score >= 0).all()
    assert (attention_score <= 1).all()

    # finite output
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention_score).all()


def test_multihead_self_attention_forward_and_backward():
    torch.manual_seed(0)

    B, T, D, H = 2, 4, 8, 4
    x = torch.randn(B, T, D, requires_grad=True)

    mha = MultiHeadSelfAttention(d_model=D, num_heads=H)
    y, attn = mha(x)

    # check shapes
    assert y.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    # check attention weights sum to 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, H, T), atol=1e-6)

    # gradients flow
    loss = y.sum() # a simple toy loss which depends on all values in y
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mhsa_causal_mask_blocks_future_attention():
    torch.manual_seed(0)

    B, T, D, H = 2, 4, 8, 4
    x = torch.randn(B, T, D)
    mha = MultiHeadSelfAttention(d_model=D, num_heads=H)
    y, attn = mha.forward(x, pad_is_real=None, causal=True)

    # check shapes
    assert y.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    # check attention weights are lower triangular
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    upper = upper.view(1, 1, T, T) # broadcast to (B, H, T, T)
    blocked_mask = attn.masked_select(upper)
    assert torch.allclose(blocked_mask, torch.zeros_like(blocked_mask), atol = 1e-6)


def test_mhsa_padding_mask_blocks_pad_keys():
    torch.manual_seed(0)

    B, T, D, H = 2, 6, 8, 2
    x = torch.randn(B, T, D)

    # create a sample padding mask (first sequence has 4 real tokens, second has 2 real tokens)
    pad_is_real = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, False, False, False, False],
    ])

    mha = MultiHeadSelfAttention(d_model=D, num_heads=H)
    y, attn = mha.forward(x, pad_is_real=pad_is_real, causal=False)

    assert y.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    # check that 0 attention weight is put on the padded keys
    pad_keys = (~pad_is_real).view(B, 1, 1, T)
    padded_mass = attn.masked_select(pad_keys)
    assert torch.allclose(padded_mass, torch.zeros_like(padded_mass), atol=1e-6, rtol=0.0)

    # For queries that are real tokens, attention over keys should still sum to 1.
    # (Rows for padded queries aren't meaningful; we don't care.)
    real_queries = pad_is_real.view(B, 1, T, 1)  # (B, 1, T, 1)
    row_sums = attn.sum(dim=-1, keepdim=True)  # (B, H, T, 1)
    row_sums_real_queries = row_sums.masked_select(real_queries)

    assert torch.allclose(row_sums_real_queries, torch.ones_like(row_sums_real_queries), atol=1e-6, rtol=0.0)


def test_mhsa_combined_causal_and_padding_masks():
    torch.manual_seed(0)

    B, T, D, H = 2, 6, 8, 2
    x = torch.randn(B, T, D)

    # create a sample padding mask (first sequence has 4 real tokens, second has 2 real tokens)
    pad_is_real = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, False, False, False, False],
    ])

    mha = MultiHeadSelfAttention(d_model=D, num_heads=H)
    y, attn = mha.forward(x, pad_is_real=pad_is_real, causal=True)

    assert y.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    # check causal mask works: attention weights are lower triangular
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    upper = upper.view(1, 1, T, T) # broadcast to (B, H, T, T)
    blocked_mask = attn.masked_select(upper)
    assert torch.allclose(blocked_mask, torch.zeros_like(blocked_mask), atol = 1e-6)

    # check that the padding mask works: 0 attention weight is put on the padded keys
    pad_keys = (~pad_is_real).view(B, 1, 1, T)
    padded_mass = attn.masked_select(pad_keys)
    assert torch.allclose(padded_mass, torch.zeros_like(padded_mass), atol=1e-6, rtol=0.0)


def test_transformer_encoder_block_forward_and_backward():
    torch.manual_seed(0)

    B, T, D, H, DFF = 2, 5, 8, 2, 32
    x = torch.randn(B, T, D, requires_grad=True)

    pad_is_real = torch.tensor([
        [True, True, True, True, False],
        [True, True, True, False, False],
    ])

    block = TransformerEncoderBlock(d_model=D, d_ff=DFF, num_heads=H)
    y = block(x, pad_is_real)

    # check output shape
    assert y.shape == (B, T, D)
    assert torch.isfinite(y).all()

    # check gradient propagates without exploding
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


