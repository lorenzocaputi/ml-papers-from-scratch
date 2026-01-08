import torch
from attention_is_all_you_need.transformer_seq2seq import TransformerSeq2Seq

def test_seq2seq_forward_backward():
    torch.manual_seed(0)

    B, Ts, Tt = 2, 6, 5
    V = 30
    D, H, DFF = 16, 4, 64

    src = torch.randint(1, V, (B, Ts))
    tgt = torch.randint(1, V, (B, Tt))

    src_pad_is_real = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, True, True, True, False],
    ])

    tgt_pad_is_real = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, False],
    ])

    model = TransformerSeq2Seq(
        vocab_size=V,
        d_model=D,
        d_ff=DFF,
        num_heads=H,
        num_layers_enc=2,
        num_layers_dec=2,
        max_len=32,
    )

    logits = model(src, tgt, src_pad_is_real=src_pad_is_real, tgt_pad_is_real=tgt_pad_is_real)
    assert logits.shape == (B, Tt, V)

    loss = logits.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
