from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

from navsim.agents.recogdrive.int8_quant_kernels import (
    fused_quantize_activation_per_token_int8,
    fused_rmsnorm_quantize_activation_per_token_int8,
)


def _time_cuda(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * normed.to(dtype=x.dtype)


def benchmark_shape(rows: int, cols: int, dtype: torch.dtype, warmup: int, repeat: int) -> dict[str, float | int | str]:
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    weight = torch.randn(cols, device="cuda", dtype=dtype) * 0.02 + 1.0

    separate = lambda: fused_quantize_activation_per_token_int8(_rmsnorm(x, weight, 1e-6), eps=1e-6)
    fused = lambda: fused_rmsnorm_quantize_activation_per_token_int8(x, weight, rms_eps=1e-6, quant_eps=1e-6)
    rms_only = lambda: _rmsnorm(x, weight, 1e-6)

    q_ref, scale_ref = separate()
    q_fused, scale_fused = fused()
    torch.cuda.synchronize()

    return {
        "rows": rows,
        "cols": cols,
        "dtype": str(dtype).replace("torch.", ""),
        "rmsnorm_only_ms": _time_cuda(rms_only, warmup, repeat),
        "rmsnorm_then_quant_ms": _time_cuda(separate, warmup, repeat),
        "fused_rmsnorm_quant_ms": _time_cuda(fused, warmup, repeat),
        "q_max_abs_diff": int((q_ref.to(torch.int16) - q_fused.to(torch.int16)).abs().max().item()),
        "scale_max_abs_diff": float((scale_ref - scale_fused).abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    shapes = [
        ("llm_hidden", 580, 1536),
        ("vision_hidden", 2304, 1024),
        ("diffusion_hidden", 64, 1024),
    ]
    results = []
    for name, rows, cols in shapes:
        item = benchmark_shape(rows, cols, dtype, args.warmup, args.repeat)
        item["name"] = name
        results.append(item)

    payload = {"results": results}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
