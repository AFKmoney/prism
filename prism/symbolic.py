"""Differentiable Symbolic Primitive Library.

The Symbolic expert soft-selects and composes a closed library of typed,
differentiable primitives. Each primitive takes one or two tensor arguments
(derived from the input feature ``x``) and returns a tensor of the same shape
as ``x`` so the result can re-enter the residual stream.

This is *neuro-symbolic in the weights*: there is no external solver. The
primitives are pure tensor operations, fully differentiable, and the model
learns via straight-through estimation which primitive to apply at each token.

Why this is different from a plain MLP
-------------------------------------
A standard MLP applies one fixed nonlinear transformation. The Symbolic expert
applies a *soft mixture over structurally different operations* (compare, gate,
select, shift, threshold, count). The router can choose, per token, whether the
right move is "compare two halves of the vector" vs "count active dimensions"
vs "soft-threshold" — operations with distinct algebraic semantics that an MLP
cannot represent in a single layer.
"""

from __future__ import annotations

import torch
from torch import nn

from prism.config import PrismConfig


# Number of arguments each primitive consumes from the argument head.
# 1 = unary (operates on x only), 2 = binary (operates on x and a derived arg).
PRIMITIVE_ARITY = {
    "compare": 2,
    "gate": 3,
    "select": 2,
    "shift": 1,
    "threshold": 2,
    "count": 1,
}

PRIMITIVE_NAMES = list(PRIMITIVE_ARITY.keys())


def _split_halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the last dim of x into two equal halves (padding if odd)."""
    d = x.shape[-1]
    if d % 2 == 1:
        x = nn.functional.pad(x, (0, 1))
        d += 1
    return x[..., : d // 2], x[..., d // 2 :]


def prim_compare(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between the two halves of x, broadcast back.

    Returns a tensor shaped like x where every position holds the scalar
    similarity (so downstream layers can read it). This makes 'compare' a
    detector of internal consistency / repetition — useful for induction.
    """
    h1, h2 = _split_halves(x)
    sim = nn.functional.cosine_similarity(h1, h2, dim=-1)  # (...,)
    return sim.unsqueeze(-1).expand_as(x).contiguous()


def prim_gate(x: torch.Tensor, c: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable conditional: c·x + (1-c)·b, with c squashed into [0,1].

    Lets the model blend the input with a learned alternative based on a
    learned condition — an explicit IF/ELSE inside the network.
    """
    c = torch.sigmoid(c)
    return c * x + (1.0 - c) * b


def prim_select(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Soft masked lookup: Σ m_i x_i broadcast across the feature dim.

    Treats m as a soft mask over the feature dimension (after softmax), then
    returns the weighted sum tiled back to x's shape. Acts as a soft gather.
    """
    d = x.shape[-1]
    m = m[..., :d]
    if m.shape[-1] < d:
        m = nn.functional.pad(m, (0, d - m.shape[-1]))
    w = torch.softmax(m, dim=-1)
    s = (x * w).sum(-1, keepdim=True)
    return s.expand_as(x).contiguous()


def prim_shift(x: torch.Tensor) -> torch.Tensor:
    """Rotate the feature dim by one position (cyclic). A positional transform."""
    return torch.cat([x[..., -1:], x[..., :-1]], dim=-1)


def prim_threshold(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Soft threshold: x * sigmoid(k(x - t)) with learned scale k=5.

    A smooth step function. Lets the model suppress values below a learned
    threshold — the discrete 'if x > t' made differentiable.
    """
    d = x.shape[-1]
    t = t[..., :d]
    if t.shape[-1] < d:
        t = nn.functional.pad(t, (0, d - t.shape[-1]))
    k = 5.0
    gate = torch.sigmoid(k * (x - t))
    return x * gate


def prim_count(x: torch.Tensor) -> torch.Tensor:
    """Count active dimensions (sum of sigmoid activations), broadcast back.

    Returns the scalar 'how many features are on', tiled to x's shape.
    """
    d = x.shape[-1]
    active = torch.sigmoid(x).sum(-1, keepdim=True) / max(d, 1)  # normalized
    return active.expand_as(x).contiguous()


PRIMITIVE_FUNCS = {
    "compare": prim_compare,
    "gate": prim_gate,
    "select": prim_select,
    "shift": prim_shift,
    "threshold": prim_threshold,
    "count": prim_count,
}


class SymbolicLibrary(nn.Module):
    """Differentiable primitive library with soft selection.

    Per token, produces:
      * a distribution ``p`` over primitives (learned),
      * argument tensors derived from x,
      * output = Σ_i p_i · prim_i(args).

    Hardening in the forward pass uses straight-through estimation so the
    selection is one-hot at inference but gradients still flow through p.
    """

    def __init__(self, config: PrismConfig) -> None:
        super().__init__()
        self.config = config
        self.num_primitives = len(PRIMITIVE_NAMES)
        assert self.num_primitives == config.symbolic_num_primitives, (
            f"symbolic_num_primitives={config.symbolic_num_primitives} but the "
            f"library has {self.num_primitives} primitives. Adjust the config."
        )

        d = config.d_model

        # Primitive-selection logits.
        self.select_proj = nn.Linear(d, self.num_primitives, bias=True)

        # Argument heads. We produce up to 3 argument tensors (gate is ternary),
        # each of width d_model, so the primitives have material to work with.
        # For efficiency we project once into 3*d and split.
        self.arg_proj = nn.Linear(d, 3 * d, bias=True)

        # Output projection on the *weighted* primitive sum (not the concat).
        # This is both cheaper (d->d instead of P*d->d) and lets the selected
        # primitive dominate naturally.
        self.out_proj = nn.Linear(d, d, bias=False)

        # Init
        nn.init.normal_(self.select_proj.weight, std=config.init_std)
        nn.init.normal_(self.arg_proj.weight, std=config.init_std)
        nn.init.normal_(self.out_proj.weight, std=config.init_std)
        nn.init.zeros_(self.select_proj.bias)
        nn.init.zeros_(self.arg_proj.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the soft-composed program.

        Args:
            x: shape (..., d_model).

        Returns:
            (output, prim_weights) where output is (..., d_model) and
            prim_weights is (..., num_primitives) — interpretable.
        """
        *lead, d = x.shape
        P = self.num_primitives

        # Primitive selection logits -> soft distribution.
        logits = self.select_proj(x)                      # (..., P)
        p_soft = torch.softmax(logits, dim=-1)            # (..., P)

        # Straight-through hardening: forward uses one-hot of argmax, backward
        # uses the soft distribution. This keeps the model's *behaviour*
        # symbolic-ish at inference while remaining trainable.
        p_hard = nn.functional.one_hot(
            logits.argmax(dim=-1), num_classes=P
        ).to(dtype=p_soft.dtype)
        p = p_hard + p_soft - p_soft.detach()             # ST estimator

        # Arguments.
        args = self.arg_proj(x)                           # (..., 3d)
        a1, a2, a3 = args.chunk(3, dim=-1)                # each (..., d)

        # Compute each primitive's output, weight by p, sum, then project.
        # The weighted sum means the argmax (one-hot via ST) primitive
        # dominates in the forward pass, while all contribute a little via
        # the epsilon-soft term handled by the caller.
        prim_outs = []
        for name in PRIMITIVE_NAMES:
            arity = PRIMITIVE_ARITY[name]
            fn = PRIMITIVE_FUNCS[name]
            if arity == 1:
                out = fn(x)
            elif arity == 2:
                out = fn(x, a1)
            elif arity == 3:
                out = fn(x, a1, a2)
            else:  # pragma: no cover - defensive
                raise AssertionError(f"unsupported arity {arity}")
            prim_outs.append(out)
        stacked = torch.stack(prim_outs, dim=-2)          # (..., P, d)
        weighted_sum = (stacked * p.unsqueeze(-1)).sum(dim=-2)  # (..., d)
        out = self.out_proj(weighted_sum)                 # (..., d)
        return out, p_soft
