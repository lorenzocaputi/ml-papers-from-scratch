import torch
import torch.nn as nn
from attention_is_all_you_need.implementation.attention import MultiHeadSelfAttention, FeedForward
from attention_is_all_you_need.implementation.positional_encoding import SinusoidalPositionalEncoding


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int):
        super().__init__()
        self.msa = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_is_real=None, causal=False):
        # x: (B, T, D)

        # Self-attention + residual + norm
        attn_out, _ = self.msa(x, pad_is_real=pad_is_real, causal=causal)
        x = self.norm1(x + attn_out) # (B, T, D)
        
        # FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out) # (B, T, D)
        
        return x


class TransformerEncoder(nn.Module):
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
        self.layers = nn.ModuleList([TransformerEncoderBlock(d_model, d_ff, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, pad_is_real: torch.Tensor | None = None):
        """
        Turns a set of sequences of tokens into a context-aware latent represention

        args:
            tokens: (B, T)
            pad_is_real: (B, T)
        returns:
            x: (B, T, D) 
        """
        x = self.embed(tokens)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, pad_is_real=pad_is_real)
        
        x = self.norm(x)
        return x


class TransformerEncoderForTokenClassification(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 d_ff: int, 
                 num_heads: int, 
                 num_layers: int, 
                 max_len: int = 512):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, d_ff, num_heads, num_layers, max_len)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens: torch.Tensor, pad_is_real: torch.Tensor | None = None):
        x = self.encoder(tokens, pad_is_real=pad_is_real) # (B, T, D)
        logits = self.lm_head(x) # (B, T, V) 

        return logits


