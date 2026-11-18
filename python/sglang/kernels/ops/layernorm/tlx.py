"""gfx950 residual-stream boundary from fbtriton ``tlx.ops.norm_residual_norm``, vendored in ``sglang.kernels.tlx_ops``.

``norm_residual_norm`` RMS-normalizes a BF16 branch (weight after the norm, or the input-scale form that applies it
first), rounds it to BF16, updates the FP32 stream as ``alpha * stream + beta * y`` (per-channel or scalar gate) or
``stream + y``, and RMS-normalizes the new stream, optionally rounded to BF16 first. It returns the BF16 output and
the FP32 stream. Rejected (see ``norm_residual_norm_supported``): hidden sizes other than 4096, 8192 and 16384, other
dtypes, non-contiguous or empty inputs. No FP8 or router copy of the output. The caller gates the architecture.
Runtime requirements are in ``sglang/kernels/jit/utils/tlx_runtime.py``.
"""

import torch

_HIDDEN_SIZES = (4096, 8192, 16384)


def norm_residual_norm_supported(stream: torch.Tensor, branch: torch.Tensor) -> bool:
    """Whether ``norm_residual_norm`` handles these operands; host-only, never reads device data."""
    return (
        stream.dtype == torch.float32
        and branch.dtype == torch.bfloat16
        and stream.dim() == 2
        and stream.shape == branch.shape
        and stream.shape[0] > 0
        and stream.shape[1] in _HIDDEN_SIZES
        and stream.is_contiguous()
        and branch.is_contiguous()
    )


def norm_residual_norm(
    stream: torch.Tensor,
    branch: torch.Tensor,
    *,
    post_weight: torch.Tensor | None,
    post_norm_input_scale: bool,
    pre_weight: torch.Tensor | None,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    post_eps: float,
    pre_eps: float,
    pre_norm_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(output, new_stream)``; ``alpha``/``beta`` FP32 ``(H,)`` or ``(1,)``, both or neither."""
    from sglang.kernels.tlx_ops.kernels.fused.norm_residual_norm.gfx950 import (
        norm_residual_norm as _norm_residual_norm,
    )

    return _norm_residual_norm(
        stream,
        branch,
        post_weight,
        post_norm_input_scale,
        pre_weight,
        alpha,
        beta,
        post_eps,
        pre_eps,
        pre_norm_bf16,
        None,
        None,
    )


def rmsnorm_supported(x: torch.Tensor, weight: torch.Tensor | None) -> bool:
    """Whether ``rmsnorm`` runs these operands on the TLX kernel; host-only, never reads device data.

    Only 4096-feature BF16/FP16 rows take the TLX kernel; ``tlx.ops.rmsnorm`` sends every other width to a plain
    Triton kernel, which this wrapper does not offer.
    """
    from sglang.kernels.tlx_ops.kernels.norm.gfx950 import use_tlx_kernel

    return (
        x.dim() == 2
        and x.shape[0] > 0
        and x.stride(1) == 1
        and use_tlx_kernel(x.shape[1], x.dtype)
        and (
            weight is None
            or (
                weight.shape == (x.shape[1],)
                and weight.dtype in (x.dtype, torch.float32)
                and weight.is_contiguous()
            )
        )
    )


def rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """``x * rstd * weight`` per row in FP32 (no weight: ``x * rstd``), in ``x``'s dtype; contiguous output."""
    from sglang.kernels.tlx_ops.kernels.norm.gfx950 import rmsnorm as _rmsnorm

    return _rmsnorm(x, weight, eps=eps)[0]


def rmsnorm_gate_supported(x: torch.Tensor, gate: torch.Tensor) -> bool:
    """Whether ``rmsnorm_gate`` handles these operands; host-only, never reads device data.

    Head dim 64 and a weightless norm are the caller's to check; here: BF16 ``x`` and ``gate`` of one ``(T, H * 64)``
    shape, ``H`` a multiple of 8, contiguous rows.
    """
    return (
        x.dim() == 2
        and gate.shape == x.shape
        and x.shape[0] > 0
        and x.shape[1] % (8 * 64) == 0
        and x.dtype == gate.dtype == torch.bfloat16
        and x.stride(1) == 1
        and gate.stride(1) == 1
    )


def rmsnorm_gate(x: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """Per 64-element head: ``bf16(x * rsqrt(mean(x * x) + eps) * sigmoid(gate))`` in FP32; contiguous output."""
    from sglang.kernels.tlx_ops.kernels.fused.rmsnorm_gate.gfx950 import (
        rmsnorm_gate as _rmsnorm_gate,
    )

    return _rmsnorm_gate(x, gate, eps, None)
