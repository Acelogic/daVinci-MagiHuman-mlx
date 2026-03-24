"""Tests for pipeline utilities."""
import mlx.core as mx
from davinci_mlx.pipeline.common import validate_dimensions, compute_latent_shape


def test_validate_256p():
    validate_dimensions(256, 256, 65)  # should not raise


def test_validate_bad_resolution():
    import pytest
    with pytest.raises(ValueError):
        validate_dimensions(100, 100, 65)


def test_compute_latent_shape():
    shape = compute_latent_shape(256, 256, 65)
    # T = ceil(65/4) = 17, H = 256/16 = 16, W = 256/16 = 16
    assert shape == (48, 17, 16, 16), f"Got {shape}"


def test_compute_latent_shape_540p():
    shape = compute_latent_shape(544, 960, 65)
    # T=17, H=544/16=34, W=960/16=60
    assert shape == (48, 17, 34, 60), f"Got {shape}"
