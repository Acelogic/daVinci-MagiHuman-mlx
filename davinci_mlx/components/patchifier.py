"""Video latent patchifier: spatial (B,C,T,H,W) <-> sequence (B,N,D)."""
import mlx.core as mx


class VideoLatentPatchifier:
    def __init__(self, patch_size: int = 2):
        self.p = patch_size

    def patchify(self, latent: mx.array) -> mx.array:
        B, C, T, H, W = latent.shape
        p = self.p
        Hp, Wp = H // p, W // p
        x = latent.reshape(B, C, T, Hp, p, Wp, p)
        x = x.transpose(0, 2, 3, 5, 1, 4, 6)
        x = x.reshape(B, T * Hp * Wp, C * p * p)
        return x

    def unpatchify(self, x: mx.array, num_frames: int, height: int, width: int) -> mx.array:
        B, N, D = x.shape
        p = self.p
        Hp, Wp = height // p, width // p
        C = D // (p * p)
        x = x.reshape(B, num_frames, Hp, Wp, C, p, p)
        x = x.transpose(0, 4, 1, 2, 5, 3, 6)
        x = x.reshape(B, C, num_frames, height, width)
        return x
