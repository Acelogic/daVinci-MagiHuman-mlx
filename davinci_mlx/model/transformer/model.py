"""DaVinci DiT: 15B single-stream transformer.

Sandwich: adapter -> 40 blocks -> final projection.
Video + text concatenated, self-attention only, timestep-free.
"""
import mlx.core as mx
import mlx.nn as nn
from davinci_mlx.model.transformer.transformer import TransformerBlock, MultiModalityRMSNorm
from davinci_mlx.model.transformer.rope import ElementWiseFourierEmbed, precompute_freqs


class DaVinciModel(nn.Module):
    def __init__(self, hidden_size=5120, num_layers=40, num_heads_q=40,
                 num_heads_kv=8, head_dim=128, video_in_channels=192,
                 text_in_channels=3584):
        super().__init__()
        self.hidden_size = hidden_size
        self.video_in_channels = video_in_channels

        # Adapter: embed each modality to hidden_size
        self.video_embedder = nn.Linear(video_in_channels, hidden_size, bias=True)
        self.text_embedder = nn.Linear(text_in_channels, hidden_size, bias=True)

        # Positional encoding (learnable Fourier bands, loaded from weights)
        self.rope = ElementWiseFourierEmbed(num_bands=64)

        # 40 transformer blocks
        self.blocks = [
            TransformerBlock(hidden_size, num_heads_q, num_heads_kv, head_dim, i)
            for i in range(num_layers)
        ]

        # Final video output head
        self.final_norm_video = MultiModalityRMSNorm(hidden_size)
        self.final_linear_video = nn.Linear(hidden_size, video_in_channels, bias=False)

        # Store head_dim for RoPE precomputation
        self._head_dim = head_dim

    def __call__(self, video_tokens, text_tokens):
        num_video = video_tokens.shape[1]

        # Embed to hidden_size
        video_hidden = self.video_embedder(video_tokens)
        text_hidden = self.text_embedder(text_tokens)

        # Concatenate: [video || text]
        x = mx.concatenate([video_hidden, text_hidden], axis=1)

        # Precompute RoPE frequencies
        total_len = x.shape[1]
        cos_freqs, sin_freqs = precompute_freqs(dim=self._head_dim, max_pos=total_len)
        positions = mx.arange(total_len)

        # Run through all blocks
        for block in self.blocks:
            x = block(x, cos_freqs, sin_freqs, positions)

        # Extract video tokens and project to output
        video_out = x[:, :num_video, :]
        video_out = self.final_norm_video(video_out)
        return self.final_linear_video(video_out)
