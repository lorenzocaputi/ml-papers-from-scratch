import torch

def make_copy_batch(
    batch_size: int,
    max_len: int,
    vocab_size: int,
    pad_id: int = 0,
    min_len: int = 1,
    device: str | torch.device = "cpu",
):

    """
    Creates a batch for a copy task:
      input tokens: (B, T)
      target tokens: (B, T)  (same as input)
      pad_is_real: (B, T) boolean

    We generate variable-length sequences and pad to max_len.
    """

    assert 0 <= pad_id < vocab_size
    assert 1 <= min_len <= max_len

    lengths = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,), device=device)

    # start putting padding all over x
    x = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    pad_is_real = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for b in range(batch_size):
        L = int(lengths[b].item())
        # sample tokens excluding pad_id to avoid ambiguity
        tokens = torch.randint(1, vocab_size, (L,), device=device)
        # fill the batch with tokens and the padding mask with booleans
        x[b, :L] = tokens
        pad_is_real[b, :L] = True


    y = x.clone()  # copy task target
    return x, y, pad_is_real



