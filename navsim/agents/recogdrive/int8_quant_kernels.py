from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None
_EXTENSION_NAME = "recogdrive_int8_quant_v3"


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    source_dir = Path(__file__).resolve().parent / "csrc"
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    _EXTENSION = load(
        name=_EXTENSION_NAME,
        sources=[
            str(source_dir / "int8_quant_kernel.cpp"),
            str(source_dir / "int8_quant_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=bool(int(os.environ.get("RECOGDRIVE_VERBOSE_EXT_BUILD", "0"))),
    )
    return _EXTENSION


def fused_quantize_activation_per_token_int8(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise ValueError("fused int8 activation quantization requires a CUDA tensor")
    if x.ndim != 2:
        raise ValueError("fused int8 activation quantization requires a 2D tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported activation dtype for fused int8 quantization: {x.dtype}")
    ext = _load_extension()
    return ext.quantize_activation_per_token_int8(x.contiguous(), float(eps))


def fused_rmsnorm_quantize_activation_per_token_int8(
    x: torch.Tensor,
    weight: torch.Tensor,
    rms_eps: float = 1e-6,
    quant_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise ValueError("fused RMSNorm + int8 activation quantization requires a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("fused RMSNorm + int8 activation quantization requires a CUDA weight tensor")
    if x.ndim != 2:
        raise ValueError("fused RMSNorm + int8 activation quantization requires a 2D tensor")
    if weight.ndim != 1:
        raise ValueError("RMSNorm weight must be a 1D tensor")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("activation hidden size must match RMSNorm weight")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported activation dtype for fused RMSNorm + quantization: {x.dtype}")
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    ext = _load_extension()
    return ext.rmsnorm_quantize_activation_per_token_int8(
        x.contiguous(),
        weight.contiguous(),
        float(rms_eps),
        float(quant_eps),
    )


def fused_rmsnorm_static_quantize_activation_int8(
    x: torch.Tensor,
    weight: torch.Tensor,
    rms_eps: float = 1e-6,
    static_scale: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise ValueError("fused RMSNorm + static int8 activation quantization requires a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("fused RMSNorm + static int8 activation quantization requires a CUDA weight tensor")
    if x.ndim != 2:
        raise ValueError("fused RMSNorm + static int8 activation quantization requires a 2D tensor")
    if weight.ndim != 1:
        raise ValueError("RMSNorm weight must be a 1D tensor")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("activation hidden size must match RMSNorm weight")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported activation dtype for fused RMSNorm + static quantization: {x.dtype}")
    if static_scale <= 0:
        raise ValueError("static_scale must be positive")
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    ext = _load_extension()
    return ext.rmsnorm_static_quantize_activation_int8(
        x.contiguous(),
        weight.contiguous(),
        float(rms_eps),
        float(static_scale),
    )


def fused_layernorm_quantize_activation_per_token_int8(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    layernorm_eps: float = 1e-6,
    quant_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise ValueError("fused LayerNorm + int8 activation quantization requires a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("fused LayerNorm + int8 activation quantization requires a CUDA weight tensor")
    if not bias.is_cuda:
        raise ValueError("fused LayerNorm + int8 activation quantization requires a CUDA bias tensor")
    if x.ndim != 2:
        raise ValueError("fused LayerNorm + int8 activation quantization requires a 2D tensor")
    if weight.ndim != 1:
        raise ValueError("LayerNorm weight must be a 1D tensor")
    if bias.ndim != 1:
        raise ValueError("LayerNorm bias must be a 1D tensor")
    if x.shape[-1] != weight.shape[0] or x.shape[-1] != bias.shape[0]:
        raise ValueError("activation hidden size must match LayerNorm weight and bias")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported activation dtype for fused LayerNorm + quantization: {x.dtype}")
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    if bias.dtype != x.dtype:
        bias = bias.to(dtype=x.dtype)
    ext = _load_extension()
    return ext.layernorm_quantize_activation_per_token_int8(
        x.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        float(layernorm_eps),
        float(quant_eps),
    )
