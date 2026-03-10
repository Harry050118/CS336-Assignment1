import torch
import torch.nn as nn

from cs336_basics.modules.attention import MultiHeadAttention
from cs336_basics.modules.RMSnorm import RMSNorm
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.embedding import Embedding
from cs336_basics.modules.ffn import ffn

class TransformerBlock(nn.Module):
    '''
    A single transformer block :
        y = x + MHA(RMSNorm(x))
        z = y + SwiGLU(RMSNorm(y))
    '''
    def __init__(
        self, 
        d_model: int = 512,
        num_heads: int = 16,
        d_ff: int = 1344,
        theta: float = 10000.0,
        max_seq_len: int = 256
    ):
        super().__init__()

        self.ln1  = RMSNorm(d_model)
        self.ln2  = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            theta=theta,
            max_seq_len=max_seq_len
        )
        self.ffn = ffn(d_model=d_model, d_ff=d_ff)

    def forward(
        self, 
        x: torch.Tensor,                                # (batch_size, seq_len, d_model)            
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
       
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(nn.Module):
    '''
    Complete transformer language model
    '''
    def __init__(
        self, 
        vocab_size: int = 10000,
        context_length: int = 256,
        d_model: int = 512,
        num_heads: int = 16,
        d_ff: int = 1344,
        num_layers: int = 4,
        theta: float = 10000.0
    ):
        super().__init__()
        
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # combine all the transformer blocks in series: [transformer_block_1, transformer_block_2, ... transformer_block_n]
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length
            ) for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self, 
        token_idx: torch.Tensor,                        # (batch_size, context_length)
        token_positions: torch.Tensor | None = None,    # (batch_size, context_length)
    ) -> torch.Tensor:
            
        x = self.token_embeddings(token_idx)            # (batch_size, context_length, d_model)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)   

        x = self.ln_final(x)                       
        logits = self.lm_head(x)              # (batch_size, context_length, vocab_size)
        return logits
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device