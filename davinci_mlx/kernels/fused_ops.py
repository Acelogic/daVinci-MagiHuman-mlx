"""Fused Metal kernels for performance-critical operations."""
import mlx.core as mx

_SILU_MUL_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx < a_shape[0]) {
        T x = a[idx];
        T sigmoid_x = T(1) / (T(1) + exp(-x));
        out[idx] = x * sigmoid_x * b[idx];
    }
"""

_silu_mul_kernel = None


def _get_kernel():
    global _silu_mul_kernel
    if _silu_mul_kernel is None:
        _silu_mul_kernel = mx.fast.metal_kernel(
            name="silu_mul",
            input_names=["a", "b"],
            output_names=["out"],
            source=_SILU_MUL_SOURCE,
        )
    return _silu_mul_kernel


def silu_mul(a: mx.array, b: mx.array) -> mx.array:
    """Fused silu(a) * b using Metal kernel."""
    kernel = _get_kernel()
    original_shape = a.shape
    n = a.size

    # Flatten to 1D so we can use a_shape[0] as the element count
    a_flat = a.reshape((n,))
    b_flat = b.reshape((n,))

    result = kernel(
        inputs=[a_flat, b_flat],
        template=[("T", a.dtype)],
        output_shapes=[(n,)],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
    )[0]

    return result.reshape(original_shape)
