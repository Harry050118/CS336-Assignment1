import torch
import torch.nn as nn
from cs336_basics.modules.linear import Linear

class ffn(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.up = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.down = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.gate = Linear(d_model, d_ff, device=device, dtype=dtype)

    @ staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        逐元素乘法（" * "), 而不是矩阵乘法(" @ ")
        SwiGLU = (Silu(x @ W1) * (x @ W2)) @ W3
        '''
        swiglu = self.down(self.silu(self.up(x)) * self.gate(x))
        return swiglu
                 