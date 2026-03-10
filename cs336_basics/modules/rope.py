import torch
import torch.nn as nn

class RoPEEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,           # 基础频率，通常 10000.0
        d_k: int,               # 每个注意力头的维度
        max_seq_len: int,      
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # [1.0, 0.562, 0.316, ...] 
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device=None):
        # positions: (seq_len,) → [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)

        # (seq_len,) × (d_k/2,) → (seq_len, d_k/2)
        # theta[i][j] = position_i × inv_freq_j
        theta = torch.einsum("i, j -> ij", positions, self.inv_freq)

        # repeat_interleave 扩展到完整维度
        # (seq_len, d_k/2) → (seq_len, d_k)
        theta_full = theta.repeat_interleave(2, dim=-1)

        # 缓存 cos 和 sin，persistent=False 不存入 state_dict
        self.register_buffer("cos_cached", theta_full.cos(), persistent=False)
        self.register_buffer("sin_cached", theta_full.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:

        # 输入: [q0, q1, q2, q3] 
        # 奇数位: x[..., ::2]  → [q0, q2]
        # 偶数位: x[..., 1::2] → [q1, q3]
        # 输出:   [-q1, q0, -q3, q2]
        x_odd  = x[..., ::2]    # 取第 0,2,4,... 位
        x_even = x[..., 1::2]   # 取第 1,3,5,... 位

        # [-x_even, x_odd]
        # stack → (..., d_k/2, 2) → flatten → (..., d_k)
        return torch.stack((-x_even, x_odd), dim=-1).flatten(start_dim=-2)

    def forward(
        self,
        x: torch.Tensor,                              # (B, n_heads, L, d_k)
        token_positions: torch.Tensor | None = None,  # 自定义位置编号
    ) -> torch.Tensor:

        seq_len = x.shape[-2]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len, x.device)

        if token_positions is None:
            cos = self.cos_cached[:seq_len]     # (L, d_k)
            sin = self.sin_cached[:seq_len]     # (L, d_k)
        else:
            cos = self.cos_cached[token_positions]   # (..., L, d_k)
            sin = self.sin_cached[token_positions]   # (..., L, d_k)

        input_dtype = x.dtype
        x = x.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        # x_rotated = x*cos + rotate_half(x)*sin
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)

        return x_rotated.to(input_dtype)
