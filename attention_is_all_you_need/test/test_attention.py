import torch
from attention_is_all_you_need.implementation.attention import (
    scaled_dot_product_attention_v0, 
    scaled_dot_product_attention_v1, 
    scaled_dot_product_attention_v2
    )

from attention_is_all_you_need.implementation.attention import MultiHeadSelfAttention, MultiHeadAttention


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


def test_mha_general_shapes():
    torch.manual_seed(0)

    B, Tq, Tk, D, H = 2, 4, 7, 8, 2
    q = torch.randn(B, Tq, D)
    k = torch.randn(B, Tk, D)
    v = torch.randn(B, Tk, D)

    mha = MultiHeadAttention(d_model=D, num_heads=H)
    y, attn = mha(q, k, v, attn_mask=None)

    assert y.shape == (B, Tq, D)
    assert attn.shape == (B, H, Tq, Tk)

    # attention rows sum to 1 over keys
    row_sums = attn.sum(dim=-1)  # (B, H, Tq)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=0.0)

    # attention weights are probabilities
    assert (attn >= 0).all()
    assert (attn <= 1).all()


def test_mha_general_backward():
    torch.manual_seed(0)

    B, Tq, Tk, D, H = 2, 3, 5, 8, 2
    q = torch.randn(B, Tq, D, requires_grad=True)
    k = torch.randn(B, Tk, D, requires_grad=True)
    v = torch.randn(B, Tk, D, requires_grad=True)

    mha = MultiHeadAttention(d_model=D, num_heads=H)

    y, attn = mha(q, k, v, attn_mask=None)
    loss = y.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()

    # some parameter grads exist
    grads = [p.grad for p in mha.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_mha_general_key_padding_mask_blocks_columns():
    """
    Mask out some key positions (columns). Those attention weights should be ~0.
    """
    torch.manual_seed(0)

    B, Tq, Tk, D, H = 2, 4, 6, 8, 2
    q = torch.randn(B, Tq, D)
    k = torch.randn(B, Tk, D)
    v = torch.randn(B, Tk, D)

    # True = real key, False = pad key
    key_is_real = torch.tensor([
        [True, True, True, False, False, False],
        [True, True, False, False, False, False],
    ])

    # Broadcast to (B, 1, 1, Tk)
    attn_mask = key_is_real.view(B, 1, 1, Tk)

    mha = MultiHeadAttention(d_model=D, num_heads=H)
    y, attn = mha(q, k, v, attn_mask=attn_mask)

    assert y.shape == (B, Tq, D)
    assert attn.shape == (B, H, Tq, Tk)

    # all attention mass on masked key columns should be ~0
    masked_key_cols = (~key_is_real).view(B, 1, 1, Tk)  # True where keys are padded
    masked_mass = attn.masked_select(masked_key_cols)
    assert torch.allclose(masked_mass, torch.zeros_like(masked_mass), atol=1e-6, rtol=0.0)

    # rows still sum to 1 (softmax renormalizes over allowed keys)
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=0.0)


def test_mha_general_causal_mask_for_self_attention_case():
    """
    Even though MultiHeadAttention is general, it should handle a causal mask
    when Tq == Tk (decoder self-attn scenario).
    """
    torch.manual_seed(0)

    B, T, D, H = 2, 5, 8, 2
    x = torch.randn(B, T, D)

    # causal mask (T,T) -> (1,1,T,T)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)

    mha = MultiHeadAttention(d_model=D, num_heads=H)
    y, attn = mha(x, x, x, attn_mask=causal)

    assert y.shape == (B, T, D)
    assert attn.shape == (B, H, T, T)

    # upper triangle must be ~0
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).view(1, 1, T, T)
    future_mass = attn.masked_select(upper)
    assert torch.allclose(future_mass, torch.zeros_like(future_mass), atol=1e-6, rtol=0.0)

    # rows sum to 1
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=0.0)


