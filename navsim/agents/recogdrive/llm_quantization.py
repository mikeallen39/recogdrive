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
    replaced_convs: int = 0


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


def _fake_quant_activation_per_token(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    return _fake_quant_symmetric(x, num_bits=num_bits, scale_dim=-1)


def _fake_quant_activation_per_sample(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    if x.ndim <= 1:
        return _fake_quant_symmetric(x, num_bits=num_bits)
    return _fake_quant_symmetric(x, num_bits=num_bits, scale_dim=tuple(range(1, x.ndim)))


class FakeQuantLinear(nn.Module):
    """Inference-only fake quant Linear.

    Weights are fake-quantized once per output channel. Activations are dynamically
    fake-quantized per token on every forward. Matmul still runs with the original
    floating dtype, so this validates quantization sensitivity rather than speed.
    """

    def __init__(self, linear: nn.Linear, weight_bits: int = 8, activation_bits: int = 8):
        super().__init__()
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        weight = _fake_quant_symmetric(linear.weight.detach(), num_bits=self.weight_bits, scale_dim=1)
        self.weight = nn.Parameter(weight, requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _fake_quant_activation_per_token(x, num_bits=self.activation_bits)
        return F.linear(x, self.weight, self.bias)


class FakeQuantConv2d(nn.Module):
    """Inference-only fake quant Conv2d for vision patch embedding."""

    def __init__(self, conv: nn.Conv2d, weight_bits: int = 8, activation_bits: int = 8):
        super().__init__()
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        weight = _fake_quant_symmetric(conv.weight.detach(), num_bits=self.weight_bits, scale_dim=(1, 2, 3))
        self.weight = nn.Parameter(weight, requires_grad=False)
        if conv.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _fake_quant_activation_per_sample(x, num_bits=self.activation_bits)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


W8A8FakeQuantLinear = FakeQuantLinear
W8A8FakeQuantConv2d = FakeQuantConv2d


def apply_fake_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
    quantize_conv2d: bool = False,
    mode: str = "w8a8_fake",
    weight_bits: int = 8,
    activation_bits: int = 8,
) -> QuantizationSummary:
    replaced_linears = 0
    replaced_convs = 0

    def convert(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced_linears, replaced_convs
        for name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if any(child_prefix.endswith(suffix) for suffix in skip_name_suffixes):
                continue
            if isinstance(child, nn.Linear):
                setattr(
                    parent,
                    name,
                    FakeQuantLinear(child, weight_bits=weight_bits, activation_bits=activation_bits),
                )
                replaced_linears += 1
            elif quantize_conv2d and isinstance(child, nn.Conv2d):
                setattr(
                    parent,
                    name,
                    FakeQuantConv2d(child, weight_bits=weight_bits, activation_bits=activation_bits),
                )
                replaced_convs += 1
            else:
                convert(child, child_prefix)

    convert(module)
    return QuantizationSummary(mode=mode, replaced_linears=replaced_linears, replaced_convs=replaced_convs)


def apply_w8a8_fake_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
    quantize_conv2d: bool = False,
    mode: str = "w8a8_fake",
) -> QuantizationSummary:
    return apply_fake_quant(
        module,
        skip_name_suffixes=skip_name_suffixes,
        quantize_conv2d=quantize_conv2d,
        mode=mode,
        weight_bits=8,
        activation_bits=8,
    )


def apply_llm_w8a8_fake_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
) -> QuantizationSummary:
    return apply_w8a8_fake_quant(
        module,
        skip_name_suffixes=skip_name_suffixes,
        quantize_conv2d=False,
        mode="w8a8_fake",
    )


def apply_llm_fake_quant(
    module: nn.Module,
    weight_bits: int,
    activation_bits: int,
    mode: str,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
) -> QuantizationSummary:
    return apply_fake_quant(
        module,
        skip_name_suffixes=skip_name_suffixes,
        quantize_conv2d=False,
        mode=mode,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
    )


def apply_vision_w8a8_fake_quant(
    module: nn.Module,
    quantize_conv2d: bool = True,
) -> QuantizationSummary:
    return apply_w8a8_fake_quant(
        module,
        skip_name_suffixes=(),
        quantize_conv2d=quantize_conv2d,
        mode="w8a8_fake",
    )
