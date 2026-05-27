from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    source_dir = Path(__file__).resolve().parent / "csrc"
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    _EXTENSION = load(
        name="recogdrive_int8_quant",
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
