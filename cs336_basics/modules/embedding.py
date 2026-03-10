import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,       # 查找表行数
        embedding_dim: int,    # 查找表列数
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty((vocab_size, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=3.0)

    def forward(
        self, 
        input: torch.Tensor
    ) -> torch.Tensor:
        '''
        input : (batch_size, seq_len), 每个元素是一个token的id
        output: (batch_size, seq_len, embedding_dim)
        '''
        return self.weight[input]

    