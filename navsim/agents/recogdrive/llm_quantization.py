from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from navsim.agents.recogdrive.int8_quant_kernels import (
    fused_layernorm_quantize_activation_per_token_int8,
    fused_quantize_activation_per_token_int8,
    fused_rmsnorm_quantize_activation_per_token_int8,
    fused_rmsnorm_static_quantize_activation_int8,
)

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
        logger as qwen2_logger,
    )
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None
    apply_rotary_pos_emb = None
    eager_attention_forward = None
    qwen2_logger = None

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
            self.weight_scale if self.weight_scale.dtype == torch.float32 else self.weight_scale.float(),
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
            self.gate_weight_scale if self.gate_weight_scale.dtype == torch.float32 else self.gate_weight_scale.float(),
            output_dtype,
            self.gate_bias,
        )
        up = sgl_int8_scaled_mm(
            x_q,
            self.up_qweight.t(),
            x_scale,
            self.up_weight_scale if self.up_weight_scale.dtype == torch.float32 else self.up_weight_scale.float(),
            output_dtype,
            self.up_bias,
        )
        hidden = self.act_fn(gate) * up
        return self.down_proj(hidden.reshape(*x.shape[:-1], self.intermediate_size))


class W8A8Qwen2UpGateConcatMLP(nn.Module):
    """Qwen2 MLP wrapper using one activation quantization and one concat gate/up GEMM."""

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
        self.register_buffer("gate_up_qweight", torch.cat([gate_qweight, up_qweight], dim=0).contiguous())
        self.register_buffer(
            "gate_up_weight_scale",
            torch.cat([gate_weight_scale.flatten(), up_weight_scale.flatten()], dim=0).contiguous(),
        )

        if mlp.gate_proj.bias is None and mlp.up_proj.bias is None:
            self.gate_up_bias = None
        else:
            gate_bias = (
                torch.zeros(self.intermediate_size, dtype=mlp.gate_proj.weight.dtype, device=mlp.gate_proj.weight.device)
                if mlp.gate_proj.bias is None
                else mlp.gate_proj.bias.detach()
            )
            up_bias = (
                torch.zeros(self.intermediate_size, dtype=mlp.up_proj.weight.dtype, device=mlp.up_proj.weight.device)
                if mlp.up_proj.bias is None
                else mlp.up_proj.bias.detach()
            )
            self.gate_up_bias = nn.Parameter(torch.cat([gate_bias, up_bias], dim=0).clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        x_2d = x.reshape(-1, self.hidden_size).contiguous()
        x_q, x_scale = _quantize_activation_per_token_int8(x_2d)
        return self.forward_quantized(x_q, x_scale, x.shape[:-1], output_dtype)

    def forward_quantized(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        input_shape: Tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        gate_up = sgl_int8_scaled_mm(
            x_q,
            self.gate_up_qweight.t(),
            x_scale.flatten().contiguous(),
            self.gate_up_weight_scale
            if self.gate_up_weight_scale.dtype == torch.float32
            else self.gate_up_weight_scale.float(),
            output_dtype,
            self.gate_up_bias,
        )
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        hidden = self.act_fn(gate) * up
        return self.down_proj(hidden.reshape(*input_shape, self.intermediate_size))


class W8A8Qwen2QKVAttention(nn.Module):
    """Qwen2 attention wrapper with shared activation quantization for q/k/v projections."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        if sgl_int8_scaled_mm is None:
            raise ImportError("w8a8_int8 requires sgl_kernel.int8_scaled_mm") from _SGL_KERNEL_IMPORT_ERROR
        if ALL_ATTENTION_FUNCTIONS is None or apply_rotary_pos_emb is None or eager_attention_forward is None:
            raise ImportError("w8a8_int8_qkv requires transformers Qwen2 attention helpers")

        self.config = attention.config
        self.layer_idx = attention.layer_idx
        self.head_dim = attention.head_dim
        self.num_key_value_groups = attention.num_key_value_groups
        self.scaling = attention.scaling
        self.attention_dropout = attention.attention_dropout
        self.is_causal = attention.is_causal
        self.o_proj = attention.o_proj

        self.hidden_size = attention.q_proj.in_features
        self.q_out_features = attention.q_proj.out_features
        self.k_out_features = attention.k_proj.out_features
        self.v_out_features = attention.v_proj.out_features
        q_qweight, q_weight_scale = _quantize_weight_per_output_channel_int8(attention.q_proj.weight)
        k_qweight, k_weight_scale = _quantize_weight_per_output_channel_int8(attention.k_proj.weight)
        v_qweight, v_weight_scale = _quantize_weight_per_output_channel_int8(attention.v_proj.weight)
        self.register_buffer("qkv_qweight", torch.cat([q_qweight, k_qweight, v_qweight], dim=0).contiguous())
        self.register_buffer(
            "qkv_weight_scale",
            torch.cat(
                [q_weight_scale.flatten(), k_weight_scale.flatten(), v_weight_scale.flatten()],
                dim=0,
            ).contiguous(),
        )

        if attention.q_proj.bias is None and attention.k_proj.bias is None and attention.v_proj.bias is None:
            self.qkv_bias = None
        else:
            q_bias = self._bias_or_zeros(attention.q_proj)
            k_bias = self._bias_or_zeros(attention.k_proj)
            v_bias = self._bias_or_zeros(attention.v_proj)
            self.qkv_bias = nn.Parameter(torch.cat([q_bias, k_bias, v_bias], dim=0).clone(), requires_grad=False)

    @staticmethod
    def _bias_or_zeros(linear: nn.Linear) -> torch.Tensor:
        if linear.bias is not None:
            return linear.bias.detach()
        return torch.zeros(linear.out_features, dtype=linear.weight.dtype, device=linear.weight.device)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        output_dtype = hidden_states.dtype
        hidden_states_2d = hidden_states.reshape(-1, self.hidden_size).contiguous()
        x_q, x_scale = _quantize_activation_per_token_int8(hidden_states_2d)
        return self.forward_quantized(
            x_q=x_q,
            x_scale=x_scale,
            input_shape=input_shape,
            output_dtype=output_dtype,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    def forward_quantized(
        self,
        x_q,
        x_scale,
        input_shape,
        output_dtype,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        hidden_shape = (*input_shape, -1, self.head_dim)
        qkv = sgl_int8_scaled_mm(
            x_q,
            self.qkv_qweight.t(),
            x_scale.flatten().contiguous(),
            self.qkv_weight_scale if self.qkv_weight_scale.dtype == torch.float32 else self.qkv_weight_scale.float(),
            output_dtype,
            self.qkv_bias,
        ).reshape(*input_shape, -1)

        query_states, key_states, value_states = qkv.split(
            [self.q_out_features, self.k_out_features, self.v_out_features],
            dim=-1,
        )
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                qwen2_logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    "Falling back to eager attention."
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class W8A8Qwen2DecoderLayerRMSQuant(nn.Module):
    """Qwen2 decoder layer that fuses RMSNorm producer with W8A8 activation quantization."""

    def __init__(self, layer: nn.Module, static_act_scale: float | None = None):
        super().__init__()
        self.hidden_size = layer.hidden_size
        self.self_attn = W8A8Qwen2QKVAttention(layer.self_attn)
        self.mlp = W8A8Qwen2UpGateConcatMLP(layer.mlp)
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.static_act_scale = static_act_scale

    @staticmethod
    def _rms_eps(norm: nn.Module) -> float:
        if hasattr(norm, "variance_epsilon"):
            return float(norm.variance_epsilon)
        if hasattr(norm, "eps"):
            return float(norm.eps)
        return 1e-6

    def _rmsnorm_quantize(self, hidden_states: torch.Tensor, norm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_2d = hidden_states.reshape(-1, self.hidden_size).contiguous()
        if self.static_act_scale is not None:
            return fused_rmsnorm_static_quantize_activation_int8(
                hidden_2d,
                norm.weight,
                rms_eps=self._rms_eps(norm),
                static_scale=self.static_act_scale,
            )
        return fused_rmsnorm_quantize_activation_per_token_int8(
            hidden_2d,
            norm.weight,
            rms_eps=self._rms_eps(norm),
            quant_eps=1e-6,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        input_shape = hidden_states.shape[:-1]
        x_q, x_scale = self._rmsnorm_quantize(hidden_states, self.input_layernorm)
        hidden_states, self_attn_weights = self.self_attn.forward_quantized(
            x_q=x_q,
            x_scale=x_scale,
            input_shape=input_shape,
            output_dtype=hidden_states.dtype,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        x_q, x_scale = self._rmsnorm_quantize(hidden_states, self.post_attention_layernorm)
        hidden_states = self.mlp.forward_quantized(
            x_q=x_q,
            x_scale=x_scale,
            input_shape=input_shape,
            output_dtype=hidden_states.dtype,
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class W8A8InternVisionAttention(nn.Module):
    """InternViT attention with fused LayerNorm-provided activation quantization for qkv."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        if sgl_int8_scaled_mm is None:
            raise ImportError("w8a8_int8 requires sgl_kernel.int8_scaled_mm") from _SGL_KERNEL_IMPORT_ERROR
        self.config = attention.config
        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.scale = attention.scale
        self.attn_drop = attention.attn_drop
        self.proj_drop = attention.proj_drop
        self.qk_normalization = attention.qk_normalization
        if self.qk_normalization:
            self.q_norm = attention.q_norm
            self.k_norm = attention.k_norm
        self.use_flash_attn = attention.use_flash_attn
        if self.use_flash_attn:
            self.inner_attn = attention.inner_attn
        self.proj = W8A8Int8Linear(attention.proj)

        qkv_qweight, qkv_weight_scale = _quantize_weight_per_output_channel_int8(attention.qkv.weight)
        self.register_buffer("qkv_qweight", qkv_qweight.contiguous())
        self.register_buffer("qkv_weight_scale", qkv_weight_scale.flatten().contiguous())
        if attention.qkv.bias is None:
            self.qkv_bias = None
        else:
            self.qkv_bias = nn.Parameter(attention.qkv.bias.detach().clone(), requires_grad=False)

    def forward_quantized(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        input_shape: Tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, seq_len = input_shape
        qkv = sgl_int8_scaled_mm(
            x_q,
            self.qkv_qweight.t(),
            x_scale.flatten().contiguous(),
            self.qkv_weight_scale if self.qkv_weight_scale.dtype == torch.float32 else self.qkv_weight_scale.float(),
            output_dtype,
            self.qkv_bias,
        ).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        if self.use_flash_attn:
            context, _ = self.inner_attn(qkv, key_padding_mask=None, need_weights=False, causal=False)
            hidden_states = context.reshape(batch_size, seq_len, self.embed_dim)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            hidden_states = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        hidden_states = self.proj(hidden_states)
        return self.proj_drop(hidden_states)


class W8A8InternVisionMLP(nn.Module):
    """InternViT MLP with fused LayerNorm-provided activation quantization for fc1."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        if sgl_int8_scaled_mm is None:
            raise ImportError("w8a8_int8 requires sgl_kernel.int8_scaled_mm") from _SGL_KERNEL_IMPORT_ERROR
        self.hidden_size = mlp.fc1.in_features
        self.intermediate_size = mlp.fc1.out_features
        self.act = mlp.act
        self.fc2 = W8A8Int8Linear(mlp.fc2)

        fc1_qweight, fc1_weight_scale = _quantize_weight_per_output_channel_int8(mlp.fc1.weight)
        self.register_buffer("fc1_qweight", fc1_qweight.contiguous())
        self.register_buffer("fc1_weight_scale", fc1_weight_scale.flatten().contiguous())
        if mlp.fc1.bias is None:
            self.fc1_bias = None
        else:
            self.fc1_bias = nn.Parameter(mlp.fc1.bias.detach().clone(), requires_grad=False)

    def forward_quantized(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        input_shape: Tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        hidden_states = sgl_int8_scaled_mm(
            x_q,
            self.fc1_qweight.t(),
            x_scale.flatten().contiguous(),
            self.fc1_weight_scale if self.fc1_weight_scale.dtype == torch.float32 else self.fc1_weight_scale.float(),
            output_dtype,
            self.fc1_bias,
        )
        hidden_states = self.act(hidden_states.reshape(*input_shape, self.intermediate_size))
        return self.fc2(hidden_states)


class W8A8InternVisionEncoderLayerLayerNormQuant(nn.Module):
    """InternViT encoder layer that fuses LayerNorm producer with W8A8 activation quantization."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.embed_dim = layer.embed_dim
        self.intermediate_size = layer.intermediate_size
        self.norm_type = layer.norm_type
        self.attn = W8A8InternVisionAttention(layer.attn)
        self.mlp = W8A8InternVisionMLP(layer.mlp)
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.ls1 = layer.ls1
        self.ls2 = layer.ls2
        self.drop_path1 = layer.drop_path1
        self.drop_path2 = layer.drop_path2

    @staticmethod
    def _layernorm_eps(norm: nn.Module) -> float:
        if hasattr(norm, "eps"):
            return float(norm.eps)
        return 1e-6

    def _layernorm_quantize(self, hidden_states: torch.Tensor, norm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_2d = hidden_states.reshape(-1, self.embed_dim).contiguous()
        if not isinstance(norm, nn.LayerNorm):
            raise TypeError("vision LayerNorm+Quant mode requires nn.LayerNorm in InternViT encoder")
        if norm.bias is None:
            raise TypeError("vision LayerNorm+Quant mode requires LayerNorm bias")
        return fused_layernorm_quantize_activation_per_token_int8(
            hidden_2d,
            norm.weight,
            norm.bias,
            layernorm_eps=self._layernorm_eps(norm),
            quant_eps=1e-6,
        )

    def forward(self, hidden_states: torch.Tensor):
        input_shape = hidden_states.shape[:-1]
        x_q, x_scale = self._layernorm_quantize(hidden_states, self.norm1)
        attn_out = self.attn.forward_quantized(
            x_q=x_q,
            x_scale=x_scale,
            input_shape=input_shape,
            output_dtype=hidden_states.dtype,
        )
        hidden_states = hidden_states + self.drop_path1(attn_out * self.ls1)

        x_q, x_scale = self._layernorm_quantize(hidden_states, self.norm2)
        mlp_out = self.mlp.forward_quantized(
            x_q=x_q,
            x_scale=x_scale,
            input_shape=input_shape,
            output_dtype=hidden_states.dtype,
        )
        hidden_states = hidden_states + self.drop_path2(mlp_out * self.ls2)
        return hidden_states


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


def apply_llm_w8a8_int8_up_gate_concat_quant(
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
                setattr(parent, name, W8A8Qwen2UpGateConcatMLP(child))
                replaced_linears += 2
            else:
                convert(child, child_prefix)

    convert(module)
    return QuantizationSummary(mode="w8a8_int8_up_gate_concat", replaced_linears=replaced_linears, replaced_convs=0)


def apply_llm_w8a8_int8_up_gate_concat_qkv_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
) -> QuantizationSummary:
    replaced_linears = 0

    def is_qwen2_mlp(child: nn.Module) -> bool:
        return all(hasattr(child, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")) and isinstance(
            child.gate_proj, nn.Linear
        ) and isinstance(child.up_proj, nn.Linear)

    def is_qwen2_attention(child: nn.Module) -> bool:
        return all(
            hasattr(child, attr)
            for attr in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "head_dim",
                "config",
                "layer_idx",
            )
        ) and isinstance(child.q_proj, nn.Linear) and isinstance(child.k_proj, nn.Linear) and isinstance(
            child.v_proj, nn.Linear
        )

    def convert(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced_linears
        for name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if any(child_prefix.endswith(suffix) for suffix in skip_name_suffixes):
                continue
            if is_qwen2_mlp(child):
                setattr(parent, name, W8A8Qwen2UpGateConcatMLP(child))
                replaced_linears += 2
            elif is_qwen2_attention(child):
                setattr(parent, name, W8A8Qwen2QKVAttention(child))
                replaced_linears += 3
            else:
                convert(child, child_prefix)

    convert(module)
    return QuantizationSummary(mode="w8a8_int8_up_gate_concat_qkv", replaced_linears=replaced_linears, replaced_convs=0)


def apply_llm_w8a8_int8_rmsnorm_up_gate_qkv_quant(
    module: nn.Module,
    skip_name_suffixes: Tuple[str, ...] = ("lm_head",),
    static_act_scale: float | None = None,
) -> QuantizationSummary:
    replaced_linears = 0

    def is_qwen2_decoder_layer(child: nn.Module) -> bool:
        return all(
            hasattr(child, attr)
            for attr in (
                "self_attn",
                "mlp",
                "input_layernorm",
                "post_attention_layernorm",
                "hidden_size",
            )
        )

    def convert(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced_linears
        for name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if any(child_prefix.endswith(suffix) for suffix in skip_name_suffixes):
                continue
            if is_qwen2_decoder_layer(child):
                setattr(parent, name, W8A8Qwen2DecoderLayerRMSQuant(child, static_act_scale=static_act_scale))
                replaced_linears += 5
            else:
                convert(child, child_prefix)

    convert(module)
    return QuantizationSummary(
        mode="w8a8_int8_rmsnorm_static_up_gate_qkv" if static_act_scale is not None else "w8a8_int8_rmsnorm_up_gate_qkv",
        replaced_linears=replaced_linears,
        replaced_convs=0,
    )


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


def apply_vision_w8a8_int8_layernorm_qkv_fc1_quant(module: nn.Module) -> QuantizationSummary:
    replaced_linears = 0

    def is_intern_vision_encoder_layer(child: nn.Module) -> bool:
        return all(
            hasattr(child, attr)
            for attr in (
                "attn",
                "mlp",
                "norm1",
                "norm2",
                "ls1",
                "ls2",
                "drop_path1",
                "drop_path2",
                "embed_dim",
                "intermediate_size",
                "norm_type",
            )
        ) and isinstance(child.norm1, nn.LayerNorm) and isinstance(child.norm2, nn.LayerNorm)

    def convert(parent: nn.Module) -> None:
        nonlocal replaced_linears
        for name, child in list(parent.named_children()):
            if is_intern_vision_encoder_layer(child):
                setattr(parent, name, W8A8InternVisionEncoderLayerLayerNormQuant(child))
                replaced_linears += 4
            else:
                convert(child)

    convert(module)
    return QuantizationSummary(
        mode="w8a8_int8_layernorm_qkv_fc1",
        replaced_linears=replaced_linears,
        replaced_convs=0,
    )
