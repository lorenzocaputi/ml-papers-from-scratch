import torch
import torch.nn as nn

from attention_is_all_you_need.attention import (
    MultiHeadAttention,
    FeedForward,
    make_causal_mask,
    make_padding_mask
)

from attention_is_all_you_need.positional_encoding import SinusoidalPositionalEncoding


class TransformerDecoderBlock(nn.Module):
    """
    Post-norm decoder block:
      1) masked self-attn
      2) cross-attn (decoder attends to encoder outputs)
      3) FFN
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self,
                x: torch.Tensor,                                # (B, Tt, D)
                enc_out: torch.Tensor,                          # (B, Ts, D)
                src_pad_is_real: torch.Tensor | None = None,    # (B, Ts)
                tgt_pad_is_real: torch.Tensor | None = None     # (B, Tt)
                ):   
        
        B, Tt, D = x.shape
        Ts = enc_out.shape[1]
        
        # 1. Masked multi-head self attention

        self_mask = make_causal_mask(Tt) # (Tt, Tt)
        self_mask.view(1, 1, Tt, Tt) # broadcasts across batch and heads dimensions

        if tgt_pad_is_real is not None:
            tgt_pad_is_real = make_padding_mask(tgt_pad_is_real)
            tgt_key_mask = tgt_pad_is_real.view(B, 1, 1, Tt) # (B, 1, 1, Tt)
            self_mask = self_mask & tgt_key_mask

        attn_out, _ = self.self_attn(q=x, k=x, v=x, attn_mask=self_mask)
        x = self.norm1(x + attn_out)

        # 2. encoder-decoder cross attention
        cross_mask = None
        if src_pad_is_real is not None:
            src_pad_is_real = make_padding_mask(src_pad_is_real) # (B, Ts)
            cross_mask = src_pad_is_real.view(B, 1, 1, Ts)     # (B, 1, 1, Ts)
        
        attn_out, _ = self.cross_attn(q=x, k=enc_out, v=enc_out, attn_mask=cross_mask)
        x = self.norm2(x + attn_out)

        # 3. Feed Forward
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 d_ff: int, 
                 num_heads: int, 
                 num_layers: int,
                 max_len: int = 512):
        
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                tgt_tokens: torch.Tensor,
                enc_out: torch.Tensor,
                src_pad_is_real: torch.Tensor | None = None,
                tgt_pad_is_real: torch.Tensor | None = None):
        
        x = self.embed(tgt_tokens)
        x = x + self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, enc_out, src_pad_is_real=src_pad_is_real, tgt_pad_is_real=tgt_pad_is_real)

        x = self.norm(x)

        return x

