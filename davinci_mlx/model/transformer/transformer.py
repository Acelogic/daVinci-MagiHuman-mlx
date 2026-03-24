"""Transformer block: RMSNorm -> Attention -> residual -> RMSNorm -> FFN -> residual.

No AdaLN. daVinci is timestep-free.
"""
import mlx.core as mx
import mlx.nn as nn
from davinci_mlx.model.transformer.attention import Attention
from davinci_mlx.model.transformer.feed_forward import SwiGLU7FFN, GELU7FFN

GELU_LAYERS = {0, 1, 2, 3}


class MultiModalityRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class AttentionWithNorm(nn.Module):
    def __init__(self, hidden_size, num_heads_q, num_heads_kv, head_dim):
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads_q, num_heads_kv, head_dim)

    def __call__(self, x, cos_freqs=None, sin_freqs=None, positions=None, mask=None):
        return self.attn(self.pre_norm(x), cos_freqs, sin_freqs, positions, mask)


class MLPWithNorm(nn.Module):
    def __init__(self, hidden_size, layer_idx,
                 swiglu_intermediate=None, gelu_intermediate=None):
        super().__init__()
        self.pre_norm = MultiModalityRMSNorm(hidden_size)
        if layer_idx in GELU_LAYERS:
            intermediate = gelu_intermediate or hidden_size * 4
            self.ffn = GELU7FFN(hidden_size, intermediate)
        else:
            intermediate = swiglu_intermediate or (int(hidden_size * 4 * 2 / 3) // 4 * 4)
            self.ffn = SwiGLU7FFN(hidden_size, intermediate)

    def __call__(self, x):
        return self.ffn(self.pre_norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=5120, num_heads_q=40, num_heads_kv=8,
                 head_dim=128, layer_idx=0):
        super().__init__()
        self.attention = AttentionWithNorm(hidden_size, num_heads_q, num_heads_kv, head_dim)
        self.mlp = MLPWithNorm(hidden_size, layer_idx)

    def __call__(self, x, cos_freqs=None, sin_freqs=None, positions=None, mask=None):
        x = x + self.attention(x, cos_freqs, sin_freqs, positions, mask)
        x = x + self.mlp(x)
        return x
