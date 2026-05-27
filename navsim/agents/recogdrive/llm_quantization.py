from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from navsim.agents.recogdrive.int8_quant_kernels import fused_quantize_activation_per_token_int8

try:
    from sgl_kernel import int8_scaled_mm as sgl_int8_scaled_mm
except ImportError as exc:
    sgl_int8_scaled_mm = None
    _SGL_KERNEL_IMPORT_ERROR = exc
else:
    _SGL_KERNEL_IMPORT_ERROR = None


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


def _quantize_weight_per_output_channel_int8(weight: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    qmax = torch.iinfo(torch.int8).max
    values = weight.detach().float()
    scale = values.abs().amax(dim=1, keepdim=True).clamp(min=eps) / qmax
    quantized = torch.round(values / scale).clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale


def _quantize_activation_per_token_int8(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("RECOGDRIVE_USE_FUSED_INT8_QUANT", "1") == "1":
        return fused_quantize_activation_per_token_int8(x, eps=eps)
    qmax = torch.iinfo(torch.int8).max
    values = x.float()
    scale = values.detach().abs().amax(dim=1, keepdim=True).clamp(min=eps) / qmax
    quantized = torch.round(values / scale).clamp(-qmax, qmax).to(torch.int8)
    return quantized.contiguous(), scale


class W8A8Int8Linear(nn.Module):
    """True W8A8 Linear using SGLang's fused int8 scaled-mm kernel.

    This is an inference-only dynamic-activation path. It intentionally requires
    sgl_kernel instead of falling back to torch._int_mm, because torch._int_mm is
    much slower for this workload and would pollute latency comparisons.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if sgl_int8_scaled_mm is None:
            raise ImportError("w8a8_int8 requires sgl_kernel.int8_scaled_mm") from _SGL_KERNEL_IMPORT_ERROR
        qweight, weight_scale = _quantize_weight_per_output_channel_int8(linear.weight)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("weight_scale", weight_scale.flatten().contiguous())
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        x_2d = x.reshape(-1, self.in_features).contiguous()
        x_q, x_scale = _quantize_activation_per_token_int8(x_2d)
        out = sgl_int8_scaled_mm(
            x_q,
            self.qweight.t(),
            x_scale.flatten().contiguous(),
            self.weight_scale,
            output_dtype,
            self.bias,
        )
        return out.reshape(*x.shape[:-1], self.out_features)


class W8A8Qwen2UpGateMLP(nn.Module):
    """Qwen2 MLP wrapper that quantizes the shared FFN input once for gate/up."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        if sgl_int8_scaled_mm is None:
            raise ImportError("w8a8_int8 requires sgl_kernel.int8_scaled_mm") from _SGL_KERNEL_IMPORT_ERROR
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size
        self.act_fn = mlp.act_fn
        self.down_proj = mlp.down_proj

        gate_qweight, gate_weight_scale = _quantize_weight_per_output_channel_int8(mlp.gate_proj.weight)
        up_qweight, up_weight_scale = _quantize_weight_per_output_channel_int8(mlp.up_proj.weight)
        self.register_buffer("gate_qweight", gate_qweight.contiguous())
        self.register_buffer("gate_weight_scale", gate_weight_scale.flatten().contiguous())
        self.register_buffer("up_qweight", up_qweight.contiguous())
        self.register_buffer("up_weight_scale", up_weight_scale.flatten().contiguous())

        self.gate_bias = (
            None
            if mlp.gate_proj.bias is None
            else nn.Parameter(mlp.gate_proj.bias.detach().clone(), requires_grad=False)
        )
        self.up_bias = (
            None
            if mlp.up_proj.bias is None
            else nn.Parameter(mlp.up_proj.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        x_2d = x.reshape(-1, self.hidden_size).contiguous()
        x_q, x_scale = _quantize_activation_per_token_int8(x_2d)
        x_scale = x_scale.flatten().contiguous()
        gate = sgl_int8_scaled_mm(
            x_q,
            self.gate_qweight.t(),
            x_scale,
            self.gate_weight_scale,
            output_dtype,
            self.gate_bias,
        )
        up = sgl_int8_scaled_mm(
            x_q,
            self.up_qweight.t(),
            x_scale,
            self.up_weight_scale,
            output_dtype,
            self.up_bias,
        )
        hidden = self.act_fn(gate) * up
        return self.down_proj(hidden.reshape(*x.shape[:-1], self.intermediate_size))


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


def apply_w8a8_int8_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
    include_name_suffixes: Tuple[str, ...] | None = None,
    mode: str = "w8a8_int8",
) -> QuantizationSummary:
    replaced_linears = 0

    def convert(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced_linears
        for name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if any(child_prefix.endswith(suffix) for suffix in skip_name_suffixes):
                continue
            if include_name_suffixes is not None and not any(
                child_prefix.endswith(suffix) for suffix in include_name_suffixes
            ):
                convert(child, child_prefix)
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, name, W8A8Int8Linear(child))
                replaced_linears += 1
            else:
                convert(child, child_prefix)

    convert(module)
    return QuantizationSummary(mode=mode, replaced_linears=replaced_linears, replaced_convs=0)


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


def apply_llm_w8a8_int8_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
) -> QuantizationSummary:
    return apply_w8a8_int8_quant(
        module,
        skip_name_suffixes=skip_name_suffixes,
        mode="w8a8_int8",
    )


def apply_llm_w8a8_int8_up_gate_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
) -> QuantizationSummary:
    replaced_linears = 0

    def is_qwen2_mlp(child: nn.Module) -> bool:
        return all(hasattr(child, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")) and isinstance(
            child.gate_proj, nn.Linear
        ) and isinstance(child.up_proj, nn.Linear)

    def convert(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced_linears
        for name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if any(child_prefix.endswith(suffix) for suffix in skip_name_suffixes):
                continue
            if is_qwen2_mlp(child):
                setattr(parent, name, W8A8Qwen2UpGateMLP(child))
                replaced_linears += 2
            else:
                convert(child, child_prefix)

    convert(module)
    return QuantizationSummary(mode="w8a8_int8_up_gate", replaced_linears=replaced_linears, replaced_convs=0)


def apply_llm_w8a8_int8_up_gate_linear_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
) -> QuantizationSummary:
    return apply_w8a8_int8_quant(
        module,
        skip_name_suffixes=skip_name_suffixes,
        include_name_suffixes=("gate_proj", "up_proj"),
        mode="w8a8_int8_up_gate_linear",
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


def apply_vision_w8a8_int8_quant(module: nn.Module) -> QuantizationSummary:
    return apply_w8a8_int8_quant(
        module,
        skip_name_suffixes=(),
        mode="w8a8_int8",
    )
