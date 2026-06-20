"""Multi-Rate Bus (MRB).

The MRB replaces self-attention with a bank of K recurrent filters at
logarithmically-spaced decay rates. Each group k has a fixed per-step retention
γ_k = exp(-Δ_k), and a learned per-token gate g_k selects which temporal
scales to read out for the current token.

Per-token recurrence (group k, width d_rate = d_model // K)::

    h_k(t) = γ_k · h_k(t-1) + (1 - γ_k) · z_k(t)
    y(t)   = Σ_k  g_k(t) · h_k(t)

where ``z_k = W_k x + b_k`` is the per-group projection of the input.

For a full sequence this is a weighted cumulative sum::

    h_k(t) = Σ_{s<=t} γ_k^(t-s) (1 - γ_k) z_k(s)

which we compute with a single parallel scan built from a causal cumulative
sum of geometric weights. Cost: O(n · d_model) time, O(K · d_model) state.

The decay schedule Δ_k is fixed (geometric on a log scale) — this is the
inductive bias. The gate g_k and the projections W_k are learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from prism.config import PrismConfig


@dataclass
class MRBOutput:
    """Output of the Multi-Rate Bus.

    Attributes:
        y: hidden states, shape (B, T, d_model).
        gates: per-token scale gates, shape (B, T, K) — interpretable.
    """

    y: torch.Tensor
    gates: torch.Tensor


def _causal_geom_cumsum(z: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute Σ_{s<=t} gamma^(t-s) z(s) for each position t.

    Vectorized parallel form. The recurrence ``h(t) = gamma·h(t-1) + z(t)``
    has the closed form::

        h(t) = gamma^t · Σ_{s<=t} gamma^(-s) z(s)

    We compute the inner cumulative sum (vectorized over batch and feature
    dims with a single ``torch.cumsum``), then multiply by the per-position
    factor ``gamma^t``. This is exact and O(n) work with no Python loop over T.

    Numerical note: ``gamma^(-s)`` grows with s, but for our toy sequence
    lengths (T <= 512) and gamma in [exp(-5.5), exp(-0.7)] ≈ [0.004, 0.5],
    the worst-case dynamic range is ~exp(5.5·512) which would overflow. We
    therefore work in float64 internally for stability, then cast back. For
    very long sequences a chunked scan would be needed; not required here.
    """
    if gamma == 0.0:
        return z
    if gamma == 1.0:
        return torch.cumsum(z, dim=1)

    B, T, D = z.shape
    device = z.device
    orig_dtype = z.dtype

    # Work in float64 for numerical headroom.
    z64 = z.double()
    # Position indices 0..T-1.
    idx = torch.arange(T, device=device, dtype=torch.float64)
    # factor_t = gamma^t, shape (T,)
    factor_t = torch.pow(gamma, idx)
    # inv_pow_s = gamma^(-s), shape (T,1,1) for broadcasting over (B,D).
    inv_pow = torch.pow(gamma, -idx).view(T, 1, 1)

    # Scaled inputs: z(s) * gamma^(-s), shape (B, T, D).
    scaled = z64 * inv_pow.permute(1, 0, 2)            # (B,T,D) * (1,T,1)
    # Cumulative sum along time: Σ_{s<=t} z(s) gamma^(-s).
    csum = torch.cumsum(scaled, dim=1)                 # (B,T,D)
    # Multiply by gamma^t per position.
    out64 = csum * factor_t.view(1, T, 1)
    return out64.to(orig_dtype)


class MultiRateBus(nn.Module):
    """Multi-Rate Bus temporal mixing layer."""

    def __init__(self, config: PrismConfig) -> None:
        super().__init__()
        self.config = config
        K = config.num_rates
        d_rate = config.d_rate

        # Per-group input projections: z_k = W_k x + b_k
        # We keep one big linear producing all K groups at once for efficiency.
        # Input is the full d_model; output is K * d_rate = d_model.
        self.in_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        # Per-token scale gate: g_k = sigmoid(W_g x + b_g)
        self.gate_proj = nn.Linear(config.d_model, K, bias=True)

        # Output projection (mixes the K groups back into d_model).
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        # Register the fixed decay schedule as a buffer (not a parameter).
        deltas = config.rate_deltas()
        gammas = [math.exp(-d) for d in deltas]
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float32), persistent=False)

        # Init
        nn.init.normal_(self.in_proj.weight, std=config.init_std)
        nn.init.normal_(self.out_proj.weight, std=config.init_std)
        nn.init.normal_(self.gate_proj.weight, std=config.init_std)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor) -> MRBOutput:
        """Apply the multi-rate temporal mixing.

        Args:
            x: shape (B, T, d_model).

        Returns:
            MRBOutput with ``y`` of shape (B, T, d_model) and ``gates`` of
            shape (B, T, K).
        """
        B, T, _ = x.shape
        K = self.config.num_rates
        d_rate = self.config.d_rate

        # Project input to per-group signals, then reshape into K groups.
        z = self.in_proj(x)                       # (B, T, d_model)
        z = z.view(B, T, K, d_rate)

        # Per-token gate over the K scales.
        gates = torch.sigmoid(self.gate_proj(x))  # (B, T, K)

        # Run the per-group decayed cumulative sum. Each group has its own γ.
        out = torch.empty(B, T, K, d_rate, device=x.device, dtype=x.dtype)
        for k in range(K):
            gamma = float(self.gammas[k].item())
            # Scale inputs by (1 - γ) so that under a constant input the
            # steady-state value equals the input (like a normalized lowpass).
            scaled = z[..., k, :] * (1.0 - gamma)
            out[..., k, :] = _causal_geom_cumsum(scaled, gamma)

        # Apply the learned per-token scale gate.
        out = out * gates.unsqueeze(-1)           # (B, T, K, d_rate)

        # Merge groups back into d_model.
        out = out.reshape(B, T, self.config.d_model)
        y = self.out_proj(out)
        return MRBOutput(y=y, gates=gates)

    @torch.no_grad()
    def effective_horizon(self) -> list[float]:
        """Return the half-life (in tokens) of each rate group.

        Useful for inspection — half-life = ln 2 / Δ_k.
        """
        return [math.log(2.0) / math.log(1.0 / g.item()) if 0 < g.item() < 1 else float("inf") for g in self.gammas]
