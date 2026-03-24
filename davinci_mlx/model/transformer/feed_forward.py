"""Feed-forward networks with clamped activations [-7, 7].

SwiGLU7: gated (3 matrices, layers 4-39). GELU7: non-gated (2 matrices, layers 0-3).
"""
import mlx.core as mx
import mlx.nn as nn
from davinci_mlx.kernels.fused_ops import silu_mul


class SwiGLU7FFN(nn.Module):
    def __init__(self, hidden_size=5120, intermediate_size=13652):
        super().__init__()
        self.up_gate_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        gate_up = self.up_gate_proj(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        hidden = silu_mul(gate, up)
        hidden = mx.clip(hidden, -7.0, 7.0)
        return self.down_proj(hidden)


class GELU7FFN(nn.Module):
    def __init__(self, hidden_size=5120, intermediate_size=20480):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        hidden = nn.gelu_approx(self.up_proj(x))
        hidden = mx.clip(hidden, -7.0, 7.0)
        return self.down_proj(hidden)
