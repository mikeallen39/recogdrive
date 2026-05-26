import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, rotate_half


def _contiguous_rotary_embeddings(rotary_embedder: nn.Module, x: torch.Tensor, seq_len: int):
    if not hasattr(rotary_embedder, "cos_cached") or not hasattr(rotary_embedder, "sin_cached"):
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return rotary_embedder(x, position_ids)

    if seq_len > getattr(rotary_embedder, "max_seq_len_cached", 0):
        rotary_embedder._set_cos_sin_cache(seq_len=seq_len, device=x.device)

    cos = rotary_embedder.cos_cached[:, :, :seq_len, :]
    sin = rotary_embedder.sin_cached[:, :, :seq_len, :]
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    """
    A versatile and highly configurable attention module.

    Supports self-attention, cross-attention, QK Normalization, and RoPE,
    while prioritizing fused attention backends like FlashAttention.
    """
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        qk_norm: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_dim = dim_head
        self.scale = dim_head ** -0.5

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        context_dim = cross_attention_dim or query_dim

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_embedder: Optional[nn.Module] = None,
        encoder_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, N_q, _ = hidden_states.shape
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        is_self_attention = encoder_hidden_states is None and encoder_kv is None

        q = self.to_q(hidden_states).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        if encoder_kv is None:
            k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_norm(k)
        else:
            k, v = encoder_kv

        q = self.q_norm(q)
        
        if rotary_embedder is not None:
            cos, sin = _contiguous_rotary_embeddings(rotary_embedder, hidden_states, N_q)
            q = torch.addcmul(q * cos, rotate_half(q), sin)
            
            if is_self_attention:
                k = torch.addcmul(k * cos, rotate_half(k), sin)

        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.to_out[-1].p if self.training else 0.0,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.to_out[-1].p, training=self.training)
            x = torch.matmul(attn_probs, v)

        x = x.transpose(1, 2).reshape(B, N_q, -1)
        return self.to_out(x)

    def project_encoder_kv(self, encoder_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute cross-attention K/V for repeated diffusion denoising steps."""
        B = encoder_hidden_states.shape[0]
        k = self.to_k(encoder_hidden_states).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(encoder_hidden_states).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        return self.k_norm(k), v
