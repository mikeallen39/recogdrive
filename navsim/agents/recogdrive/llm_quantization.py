from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class QuantizationSummary:
    mode: str
    replaced_linears: int


def _fake_quant_symmetric(
    tensor: torch.Tensor,
    num_bits: int = 8,
    scale_dim: int | Tuple[int, ...] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    qmax = (1 << (num_bits - 1)) - 1
    values = tensor.float()
    if scale_dim is None:
        scale = values.detach().abs().amax().clamp(min=eps) / qmax
    else:
        scale = values.detach().abs().amax(dim=scale_dim, keepdim=True).clamp(min=eps) / qmax
    quantized = torch.round(values / scale).clamp(-qmax, qmax)
    return (quantized * scale).to(dtype=tensor.dtype)


def _fake_quant_activation_per_token(x: torch.Tensor) -> torch.Tensor:
    return _fake_quant_symmetric(x, scale_dim=-1)


class W8A8FakeQuantLinear(nn.Module):
    """Inference-only fake W8A8 Linear.

    Weights are fake-quantized once per output channel. Activations are dynamically
    fake-quantized per token on every forward. Matmul still runs with the original
    floating dtype, so this validates quantization sensitivity rather than speed.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = _fake_quant_symmetric(linear.weight.detach(), scale_dim=1)
        self.weight = nn.Parameter(weight, requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _fake_quant_activation_per_token(x)
        return F.linear(x, self.weight, self.bias)


def apply_llm_w8a8_fake_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
) -> QuantizationSummary:
    replaced = 0

    def convert(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if any(child_prefix.endswith(suffix) for suffix in skip_name_suffixes):
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, name, W8A8FakeQuantLinear(child))
                replaced += 1
            else:
                convert(child, child_prefix)

    convert(module)
    return QuantizationSummary(mode="w8a8_fake", replaced_linears=replaced)
