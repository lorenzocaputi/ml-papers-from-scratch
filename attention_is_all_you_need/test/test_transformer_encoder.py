import torch
from attention_is_all_you_need.transformer_encoder import TransformerEncoderForTokenClassification

def test_transformer_encoder_forward_backward():
    torch.manual_seed(0)

    B, T = 2, 6
    D, H, DFF, L = 16, 4, 32, 2
    V = 20 
    tokens = torch.randint(0, V, (B, T))
    pad_is_real = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, True, True, True, False],
    ])

    model = TransformerEncoderForTokenClassification(
        vocab_size=V, d_model=D, d_ff=DFF, num_heads=H, num_layers=L, max_len=32
        )
    
    logits = model(tokens, pad_is_real)

    # check logits shape
    assert logits.shape == (B, T, V)

    # check gradients propagate and some gradients exist
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)



