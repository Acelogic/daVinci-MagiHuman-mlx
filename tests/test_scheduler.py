"""Tests for flow-matching scheduler."""
import mlx.core as mx
from davinci_mlx.components.scheduler import FlowMatchingScheduler


def test_get_sigmas_8_steps():
    scheduler = FlowMatchingScheduler()
    sigmas = scheduler.get_sigmas(num_steps=8)
    assert len(sigmas) == 9
    assert float(sigmas[0]) == 1.0
    assert float(sigmas[-1]) == 0.0


def test_euler_step():
    scheduler = FlowMatchingScheduler()
    sample = mx.ones((1, 4, 4))
    denoised = mx.zeros((1, 4, 4))
    sigma = mx.array(1.0)
    sigma_next = mx.array(0.5)
    result = scheduler.step(sample, denoised, sigma, sigma_next)
    expected = mx.full((1, 4, 4), 0.5)
    diff = mx.max(mx.abs(result - expected)).item()
    assert diff < 1e-5, f"Max diff: {diff}"
