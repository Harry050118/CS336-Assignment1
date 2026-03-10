import torch
import torch.nn as nn
from cs336_basics.modules.rope import RoPEEmbedding
from cs336_basics.modules.linear import Linear

def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    '''
    normal softmax may cause overflow when the input logits are large, 
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))),
    where max(x) is the maximum value in the input logits along the specified dimension. 
    '''

    # .values is used to get the values of the max logits, since torch.max returns a namedtuple (values, indices)
    # (B, heads, L, L) → (B, heads, L, 1)
    max_logits = torch.max(logits, dim=dim, keepdim=True).values    
    
    # (B, heads, L, L) - (B, heads, L, 1) → (B, heads, L, L)
    exp_logits = torch.exp(logits - max_logits)

    # (B, heads, L, L) → (B, heads, L, 1)
    sum_exp_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    return exp_logits / sum_exp_logits

def scaled_dot_product_attention(
    query: torch.Tensor,   # (B, heads, L_query, d_k)
    key:   torch.Tensor,   # (B, heads, L_key,   d_k)
    value: torch.Tensor,   # (B, heads, L_key,   d_v)
    mask:  torch.Tensor | None = None,
) -> torch.Tensor:
    
    '''
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    '''

    d_k = query.size(-1)  

    # (B, heads, L_query, d_k) @ (B, heads, d_k, L_key) → (B, heads, L_query, L_key)
    scores = query @ key.transpose(-2, -1) / (d_k ** 0.5)  

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))  # Causal mask
    attn_weights = stable_softmax(scores, dim=-1) 

    # (B, heads, L_query, L_key) @ (B, heads, L_key, d_k) → (B, heads, L_query, d_k)
    output = attn_weights @ value
    return output

class MultiHeadAttention(nn.Module):
    '''
    different heads can attend to different parts of the input sequence, 
    allowing the model to capture a wider range of dependencies and relationships between tokens. 
    '''
    def __init__(
            self, 
            d_model: int,                           # the dimension of the input and output features
            num_heads: int,                         
            num_kv_heads: int | None = None,        
            use_rope: bool = False,
            theta: float = 10000.0,                 # the base period of the RoPE positional encoding
            max_seq_len: int = 2048,                # the maximum sequence length for which the RoPE positional encoding is precomputed
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        assert self.d_model % self.num_heads == 0,      "d_model must be divisible by num_heads"

        self.q_proj  = Linear(d_model, d_model, device=device, dtype=dtype)  
        self.k_proj  = Linear(d_model, d_model, device=device, dtype=dtype)  
        self.v_proj  = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)  # project the concatenated output of all heads back to d_model

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool))  # (max_seq_len, max_seq_len) lower triangular matrix for causal masking  
        self.register_buffer('causal_mask', mask, persistent=False)  # buffer is a persistent tensor that is not a model parameter, but will be saved and loaded with the model

        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
    
    def forward(
        self,
        x: torch.Tensor,                              # (B, L, d_model)
        token_positions: torch.Tensor | None = None,  # use for RoPE positional encoding, (B, L) or (L,)
    ) -> torch.Tensor:
        
        B, L, _ = x.size()
        Q = self.q_proj(x)                            # (B, L, d_model)
        K = self.k_proj(x)                            # (B, L, d_model)
        V = self.v_proj(x)                            # (B, L, d_model)

        # Q: (B, L, d_model)
        #  → view:      (B, L, num_heads, d_k)
        #  → transpose: (B, num_heads, L, d_k)
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = self.rope(Q, token_positions)  # (B, num_heads, L, d_k)
            K = self.rope(K, token_positions)  # (B, num_heads, L, d_k)
        
        # causal mask: (max_seq_len, max_seq_len) → (L, L)
        # (L, L) → (1, 1, L, L) broadcast to (B, num_heads, L, L)
        causal_mask = self.causal_mask[:L, :L].unsqueeze(0).unsqueeze(0)

        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)            # (B, num_heads, L, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B, num_heads, L, d_k) → (B, L, num_heads, d_k) → (B, L, d_model)
        output = self.output_proj(attn_output)  # (B, L, d_model)
        return output