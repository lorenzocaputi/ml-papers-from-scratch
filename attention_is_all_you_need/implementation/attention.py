import torch
import math
import torch.nn as nn

#### WARM UP

def scaled_dot_product_attention_v0(Q, K, V):
    """
    Scaled dot product attention, version 0: single head, single batch.
    
    Args:
        Q: (T, D)
        K: (T, D)
        V: (T, D)
    
    Returns:
        output: (T, D)
        attention_weights: (T, T)
    """
    d = K.size(-1)
    scores = Q @ K.transpose(0, 1) / math.sqrt(d)
    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights

def scaled_dot_product_attention_v1(Q, K, V):
    """
    Scaled dot product attention, version 1: with batch dimension.
    
    Args:
        Q: (B, T, D)
        K: (B, T, D)
        V: (B, T, D)
    
    Returns:
        output: (B, T, D)
        attention_weights: (B, T, T)
    """
    d = K.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights


def scaled_dot_product_attention_v2(Q, K, V):
    """
    Scaled dot product attention, version 2: with batch dimension and multi-head.
    
    Args:
        Q: (B, H, T, Dh)
        K: (B, H, T, Dh)
        V: (B, H, T, Dh)
    
    Returns:
        output: (B, H, T, Dh)
        attention_weights: (B, H, T, T)
    """
    d = K.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ V

    return output, attention_weights

###### ACTUAL IMPLEMENTATION

def make_causal_mask(T:int, device=None) -> torch.Tensor:
    """
    Returns a boolean mask of shape (T, T) where mask[i, j] = True
    means token i is allowed to attend to token j.

    Causal means: i can attend to j only if j <= i.
    """

    # lower triangular including diagonal
    return torch.tril(torch.ones(T,T, dtype=torch.bool, device=device))

    
def make_padding_mask(pad_is_real: torch.Tensor, device=None) -> torch.Tensor:
    """
    pad_is_real: (B, T) boolean tensor, True for real tokens, False for pad.
    Returns a mask broadcastable to attention scores. 
    In this implementation it is just a datatype check, but can be made more
    sophisticated.

    We'll use it later as (B, 1, 1, T) so it masks key positions (columns).
    """
    if pad_is_real.dtype != torch.bool:
        raise ValueError("pad_is_true should be a boolean tensor of shape (B,T)") 
    return pad_is_real

## this is what they ask in interviews together with the general MHA module below

def scaled_dot_product_attention_masked(Q: torch.Tensor, 
                                        K: torch.Tensor, 
                                        V: torch.Tensor,
                                        attn_mask: torch.Tensor | None = None):
    
    """
    Scaled dot-product attention with masks. Batch + heads version.

    Args:
        Q, K, V: (B, H, T, Dh)
        attn_mask: boolean mask broadcastable to (B, H, T, T),
                   where True = allowed, False = blocked.

    Returns:
        output: (B, H, T, Dh)
        attention_weights: (B, H, T, T)
    """
    d = K.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise ValueError("attention mask should be a boolean tensor broadcatable to (B, H, T, T)")
        # fill the values with -inf where the mask is False
        scores = scores.masked_fill(~attn_mask, float("-inf")) # ~ negates the mask: masked_fill() replaces true values

    attn = torch.softmax(scores, dim=-1)
    output = attn @ V

    return output, attn
        

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model) # all heads in the same layer for now, then we'll split them
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, 
                x,
                pad_is_real: torch.Tensor | None = None,
                causal: bool = False):
        """
        Args:
            x: (B, T, D)
            pad_is_real: optional boolean mask (B, T). True=real token, False=pad token.
                         This masks *keys* (the columns in attention).
            causal: if True, apply causal mask (no looking ahead).

        Returns:
            y: (B, T, D)
            attn: (B, H, T, T)
        """
        B, T, D = x.shape
        H, Dh = self.num_heads, self.d_heads

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # split heads: (B, T, D) -> (B, H, T, Dh). Split D into two axis: H and Dh
        Q = Q.view(B, T, H, Dh).transpose(1,2)
        K = K.view(B, T, H, Dh).transpose(1,2)
        V = V.view(B, T, H, Dh).transpose(1,2)

        # create attention mask
        attn_mask = None

        if causal:
            causal_mask = make_causal_mask(T, device=x.device) # (T, T)
            # expand it to (1, 1, T, T) for broadcasting to (B, H, T, T)
            attn_mask = causal_mask.view(1, 1, T, T)

        if pad_is_real is not None:
            pad_is_real = make_padding_mask(pad_is_real) # (B, T)
            # expand to (B, 1, 1, T) for broadcasting to (B, H, T, T)
            keys_mask = pad_is_real.view(B, 1, 1, T)

            # combine both masks
            attn_mask = keys_mask if attn_mask is None else (attn_mask & keys_mask)

        # out: (B, H, T, Dh); attn: (B, H, T, T)
        out, attn = scaled_dot_product_attention_masked(Q, K, V, attn_mask=attn_mask) 

        # merge again the heads: (B, H, T, Dh) -> (B, T, D)
        out = out.transpose(1,2).contiguous().view(B, T, D)

        y = self.W_o(out)

        return y, attn
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x: (B, T, D)
        return self.net(x)
    

## this is what they ask in interviews together with the scaled_dot_product_attention function above

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model) # all heads in the same layer for now, then we'll split them
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def forward(self, 
                q,
                k,
                v,
                attn_mask: torch.Tensor | None = None):
        """
        More general for of MHA. In general, the number of tokens acting as queries is not the same
        as the number of tokens acting as keys and values. 
        For instance, in encoder-decoder cross attention, the number of tokens acting as queries is 
        the current length of the output sequence (which changes at each step), while the number of 
        tokens acting as keys and values is the length of the input sequence (fixed).
 
        Args:
            q: (B, Tq, D)
            k,v: (B, Tk, D)
            attn_mask: bool mask broadcastable to (B, H, Tq, Tk)

        Returns:
            y: (B, Tq, D)
            attn: (B, H, Tq, Tk)
        """
        
        B, Tq, D = q.shape
        Tk = k.shape[1]
        H, Dh = self.num_heads, self.d_heads

        Q = self.W_q(q).view(B, Tq, H, Dh).transpose(1,2) # (B, H, Tq, Dh)
        K = self.W_k(k).view(B, Tk, H, Dh).transpose(1,2) # (B, H, Tk, Dh)
        V = self.W_v(v).view(B, Tk, H, Dh).transpose(1,2) # (B, H, Tk, Dh)

        # out: (B, H, Tq, Dh); attn: (B, H, Tq, Tk)
        out, attn = scaled_dot_product_attention_masked(Q, K, V, attn_mask=attn_mask) 

        # merge again the heads: (B, H, Tq, Dh) -> (B, Tq, D)
        out = out.transpose(1,2).contiguous().view(B, Tq, D)

        y = self.W_o(out)

        return y, attn
    

