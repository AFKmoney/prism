"""Baseline models for fair comparison with PRISM.

All baselines expose the same minimal interface as ``Prism``::

    forward(input_ids) -> ModelOutput(logits, aux_loss=0)
    num_parameters() -> int

so the training harness can treat them interchangeably. They are sized to
match PRISM's parameter count as closely as possible for fair comparison.

Three models are provided:

* ``TransformerBaseline`` — a standard decoder-only Transformer with causal
  self-attention. The dominant architecture on the market; PRISM's main foil.
* ``SSMBaseline`` — a single-rate state-space / recurrent model (one Δ for the
  whole layer, input-dependent). Stands in for the Mamba family.
* ``PRISM`` itself is imported from prism.model.

The fair-comparison principle: same d_model, same depth, vocab, and (roughly)
the same parameter budget. We report exact param counts in the harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from prism.config import PrismConfig
from prism.norm import RMSNorm


@dataclass
class ModelOutput:
    """Minimal output shared by all models in the harness."""

    logits: torch.Tensor
    aux_loss: torch.Tensor
    aux_breakdown: dict


class _BaseLM(nn.Module):
    """Shared embedding / head scaffolding for baseline LMs."""

    def __init__(self, config: PrismConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embed.weight, std=config.init_std)

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        # Tied head for fair comparison with PRISM's default.
        return F.linear(h, self.embed.weight)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# Transformer decoder (causal self-attention + MLP). The market standard.
# =========================================================================


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d_head)
        q, k, v = qkv.unbind(dim=2)                 # each (B,T,h,dh)
        # scaled dot-product attention with causal mask.
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        attn = attn.transpose(1, 2).reshape(B, T, C)
        return self.out(attn)


class _TransformerBlock(nn.Module):
    def __init__(self, config: PrismConfig, num_heads: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = _CausalSelfAttention(config.d_model, num_heads)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        hidden = config.neural_hidden
        self.w_gate = nn.Linear(config.d_model, hidden, bias=False)
        self.w_up = nn.Linear(config.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        h = F.silu(self.w_gate(self.mlp_norm(x))) * self.w_up(self.mlp_norm(x))
        x = x + self.w_down(h)
        return x


class TransformerBaseline(_BaseLM):
    """Standard decoder-only Transformer. The market default."""

    def __init__(self, config: PrismConfig, num_heads: int = 4) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(config, num_heads) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, input_ids: torch.Tensor, mem=None) -> ModelOutput:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        logits = self._head(self.final_norm(x))
        zero = torch.zeros((), device=input_ids.device, dtype=x.dtype)
        return ModelOutput(logits=logits, aux_loss=zero, aux_breakdown={})


# =========================================================================
# Single-rate SSM (input-dependent Δ, one rate per layer). Mamba-family stand-in.
# =========================================================================


class _SSMLayer(nn.Module):
    """A simplified selective SSM with a single rate per layer.

    Update (per token)::

        Δ = softplus(w_Δ · x + b_Δ)
        A = -softplus(A_param)              # negative => stable decay
        h(t) = exp(Δ·A) h(t-1) + Δ · B(x)
        y(t) = C(x) · h(t)

    where B and C are input-dependent projections (the 'selective' part).
    This captures the *single dynamic rate* spirit of Mamba, in contrast with
    PRISM's *structured multi-rate* decomposition.
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_state = d_state
        # Input-dependent B and step Δ (the 'selective' part).
        self.b_proj = nn.Linear(d_model, d_state, bias=False)
        self.delta_proj = nn.Linear(d_model, 1, bias=True)
        # Learnable log-A (kept negative via softplus) for stable decay.
        self.A_log = nn.Parameter(torch.randn(d_state) * 0.1)
        # State readout projection.
        self.out = nn.Linear(d_state, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        N = self.d_state
        A = -F.softplus(self.A_log)                  # (N,), negative => stable

        delta = F.softplus(self.delta_proj(x))       # (B,T,1) input-dependent step
        Bb = self.b_proj(x)                          # (B,T,N) input-dependent B

        # Discretize (zero-order hold simplified): a_bar = exp(Δ·A).
        a_bar = torch.exp(delta * A)                 # (B,T,N)

        # Recurrent scan, reading out the full state h at each step.
        h = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = a_bar[:, t] * h + delta[:, t] * Bb[:, t]
            outs.append(self.out(h))                 # (B, d_model)
        return torch.stack(outs, dim=1)              # (B, T, d_model)


class _SSMBlock(nn.Module):
    def __init__(self, config: PrismConfig, d_state: int) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ssm = _SSMLayer(config.d_model, d_state)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        hidden = config.neural_hidden
        self.w_gate = nn.Linear(config.d_model, hidden, bias=False)
        self.w_up = nn.Linear(config.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm(x))
        h = F.silu(self.w_gate(self.mlp_norm(x))) * self.w_up(self.mlp_norm(x))
        x = x + self.w_down(h)
        return x


class SSMBaseline(_BaseLM):
    """Single-rate selective SSM stack (Mamba-family stand-in)."""

    def __init__(self, config: PrismConfig, d_state: int = 16) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [_SSMBlock(config, d_state) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, input_ids: torch.Tensor, mem=None) -> ModelOutput:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        logits = self._head(self.final_norm(x))
        zero = torch.zeros((), device=input_ids.device, dtype=x.dtype)
        return ModelOutput(logits=logits, aux_loss=zero, aux_breakdown={})


# =========================================================================
# Registry
# =========================================================================


def build_model(name: str, config: PrismConfig) -> nn.Module:
    """Factory keyed by model name. Used by the training harness."""
    if name == "prism":
        from prism.model import Prism

        return Prism(config)
    if name == "transformer":
        return TransformerBaseline(config)
    if name == "ssm":
        return SSMBaseline(config)
    raise ValueError(f"unknown model: {name!r}")
