import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
            self, 
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # 初始化为1，使得RMSNorm是纯粹归一化；如果为empty/zeros则会导致训练初期数值不稳定
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(
        self, 
        x: torch.Tensor      # (batch_size, seq_len, d_model)
    ) -> torch.Tensor:
        '''
        RMS(x) = sqrt(1/d * sum(x_i^2) + eps)
        RMSNorm(x) = x / RMS(x) * weight
        '''
        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)             
        RMSNorm_x = x / rms * self.weight
        return RMSNorm_x.to(input_dtype)
    