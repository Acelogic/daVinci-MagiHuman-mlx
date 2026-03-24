"""Tests for video latent patchifier."""
import mlx.core as mx
from davinci_mlx.components.patchifier import VideoLatentPatchifier


def test_patchify_shape():
    patchifier = VideoLatentPatchifier(patch_size=2)
    latent = mx.random.normal((1, 48, 4, 16, 16))
    result = patchifier.patchify(latent)
    # N = 4 * 8 * 8 = 256, D = 48 * 2 * 2 = 192
    assert result.shape == (1, 256, 192), f"Got {result.shape}"


def test_unpatchify_roundtrip():
    patchifier = VideoLatentPatchifier(patch_size=2)
    latent = mx.random.normal((1, 48, 4, 16, 16))
    patchified = patchifier.patchify(latent)
    restored = patchifier.unpatchify(patchified, num_frames=4, height=16, width=16)
    assert restored.shape == latent.shape
    diff = mx.max(mx.abs(restored - latent)).item()
    assert diff < 1e-5, f"Max diff: {diff}"


def test_patchify_256p():
    patchifier = VideoLatentPatchifier(patch_size=2)
    latent = mx.random.normal((1, 48, 16, 16, 16))
    result = patchifier.patchify(latent)
    assert result.shape == (1, 1024, 192), f"Got {result.shape}"
