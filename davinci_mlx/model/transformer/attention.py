"""GQA attention with sigmoid gating for daVinci single-stream DiT.

Fused QKV+G projection -> split -> QK norm -> RoPE -> flash attention -> gate.
"""
import math
import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size=5120, num_heads_q=40, num_heads_kv=8, head_dim=128):
        super().__init__()
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        q_dim = num_heads_q * head_dim
        kv_dim = num_heads_kv * head_dim
        qkvg_dim = q_dim + 2 * kv_dim + q_dim  # Q + K + V + Gate

        self.linear_qkv = nn.Linear(hidden_size, qkvg_dim, bias=True)
        self.q_norm_weight = mx.ones((head_dim,))
        self.k_norm_weight = mx.ones((head_dim,))
        self.linear_proj = nn.Linear(q_dim, hidden_size, bias=True)

    def __call__(self, x, cos_freqs=None, sin_freqs=None, positions=None, mask=None):
        B, T, _ = x.shape
        qkvg = self.linear_qkv(x)

        q_dim = self.num_heads_q * self.head_dim
        kv_dim = self.num_heads_kv * self.head_dim
        q = qkvg[:, :, :q_dim]
        k = qkvg[:, :, q_dim:q_dim + kv_dim]
        v = qkvg[:, :, q_dim + kv_dim:q_dim + 2 * kv_dim]
        gate = qkvg[:, :, q_dim + 2 * kv_dim:]

        q = q.reshape(B, T, self.num_heads_q, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads_kv, self.head_dim).transpose(0, 2, 1, 3)

        q = mx.fast.rms_norm(q, self.q_norm_weight, eps=1e-6)
        k = mx.fast.rms_norm(k, self.k_norm_weight, eps=1e-6)

        if cos_freqs is not None:
            from davinci_mlx.model.transformer.rope import apply_rotary_emb
            q = apply_rotary_emb(q, cos_freqs, sin_freqs, positions)
            k = apply_rotary_emb(k, cos_freqs, sin_freqs, positions)

        # MLX flash attention natively handles GQA — do NOT repeat K/V
        attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        attn_out = self.linear_proj(attn_out)
        attn_out = attn_out * mx.sigmoid(gate)
        return attn_out
