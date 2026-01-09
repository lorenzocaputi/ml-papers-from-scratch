import torch
import torch.nn as nn
from attention_is_all_you_need.toy_data import make_copy_batch
from attention_is_all_you_need.transformer_encoder import TransformerEncoderForTokenClassification


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # toy setup
    vocab_size = 50
    pad_id = 0
    T = 16

    model = TransformerEncoderForTokenClassification(
        vocab_size=vocab_size,
        d_model=64,
        d_ff=256,
        num_heads=4,
        num_layers=2,
        max_len=T,
    ).to(device)


    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # ignore pad tokens in the loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Make a tiny fixed dataset (10 samples) to overfit
    fixed_inputs = []
    for _ in range(10):
        x, y, pad_is_real = make_copy_batch(
            batch_size=1, max_len=T, vocab_size=vocab_size, pad_id=pad_id, min_len=4, device=device
        )
        fixed_inputs.append((x, y, pad_is_real))

    model.train()

    for step in range(1, 2001):
        # sample one of the 10 examples
        x, y, pad_is_real = fixed_inputs[step % 10]

        logits = model(x, pad_is_real=pad_is_real)  # (B, T, V)

        # reshape for CE: (B*T, V) vs (B*T,)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)  # (B, T)
                # accuracy only on real tokens
                correct = (pred == y) & pad_is_real
                acc = correct.sum().item() / pad_is_real.sum().item()

            print(f"step={step:4d}  loss={loss.item():.4f}  acc={acc:.3f}")

    # Show one qualitative example
    model.eval()
    x, y, pad_is_real = fixed_inputs[0]
    with torch.no_grad():
        pred = model(x, pad_is_real=pad_is_real).argmax(dim=-1)

    print("\nExample:")
    print("x   :", x[0].tolist())
    print("pred:", pred[0].tolist())
    print("y   :", y[0].tolist())


if __name__ == "__main__":
    main()
