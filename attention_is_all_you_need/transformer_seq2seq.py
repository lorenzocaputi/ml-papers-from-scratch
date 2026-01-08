import torch
import torch.nn as nn
from attention_is_all_you_need.transformer_encoder import TransformerEncoder
from attention_is_all_you_need.transformer_decoder import TransformerDecoder

class TransformerSeq2Seq(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int,
                 d_ff: int, 
                 num_heads: int,
                 num_layers_enc: int,
                 num_layers_dec: int,
                 max_len: int = 512):
        super().__init__()

        self.enc = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers_enc,
            max_len=max_len
        )
        self.dec = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers_dec,
            max_len=max_len
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, 
                src_tokens: torch.Tensor,                     # (B, Ts)
                tgt_tokens: torch.Tensor,                     # (B, Tt)
                src_pad_is_real: torch.Tensor | None = None,  # (B, Ts)
                tgt_pad_is_real: torch.Tensor | None = None,  # (B, Tt)
                ):
        
        enc_out = self.enc(tokens=src_tokens, 
                           pad_is_real=src_pad_is_real) # (B, Ts, D)
        
        dec_out = self.dec(tgt_tokens=tgt_tokens, 
                           enc_out=enc_out, 
                           src_pad_is_real=src_pad_is_real, 
                           tgt_pad_is_real=tgt_pad_is_real) # (B, Tt, D)
        
        logits = self.lm_head(dec_out) # (B, Tt, V)

        return logits
        

