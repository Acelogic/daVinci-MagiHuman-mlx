"""Flow-matching scheduler for distilled inference."""
import mlx.core as mx


class FlowMatchingScheduler:
    def get_sigmas(self, num_steps: int = 8) -> mx.array:
        return mx.linspace(1.0, 0.0, num_steps + 1)

    def step(self, sample: mx.array, denoised: mx.array,
             sigma: mx.array, sigma_next: mx.array) -> mx.array:
        velocity = (sample - denoised) / sigma
        return sample + velocity * (sigma_next - sigma)
