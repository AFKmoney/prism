"""Unit tests for the Multi-Rate Bus."""

from __future__ import annotations

import math

import torch

from prism.config import PrismConfig
from prism.mrb import MultiRateBus, _causal_geom_cumsum


def _cfg(**kw) -> PrismConfig:
    base = dict(d_model=16, num_rates=4)
    base.update(kw)
    return PrismConfig(**base)


def test_causal_geom_cumsum_constant_input_steady_state():
    # Under a constant input c scaled by (1-gamma), the steady state of
    #   h(t) = gamma h(t-1) + (1-gamma) c  is c.
    gamma = 0.5
    z = torch.full((1, 20, 3), (1 - gamma) * 7.0)
    h = _causal_geom_cumsum(z, gamma)
    # After many steps, h(t) -> 7.0
    assert torch.allclose(h[:, -1], torch.full((1, 3), 7.0), atol=1e-5)


def test_causal_geom_cumsum_causality():
    # The output at t must not depend on inputs after t.
    z1 = torch.randn(2, 8, 4)
    z2 = z1.clone()
    z2[:, 5:, :] = 99.0  # corrupt the future
    gamma = 0.7
    h1 = _causal_geom_cumsum(z1, gamma)
    h2 = _causal_geom_cumsum(z2, gamma)
    assert torch.allclose(h1[:, :5], h2[:, :5])


def test_mrb_output_shape():
    cfg = _cfg()
    mrb = MultiRateBus(cfg)
    x = torch.randn(3, 10, cfg.d_model)
    out = mrb(x)
    assert out.y.shape == (3, 10, cfg.d_model)
    assert out.gates.shape == (3, 10, cfg.num_rates)


def test_mrb_gates_in_unit_interval():
    cfg = _cfg()
    mrb = MultiRateBus(cfg)
    x = torch.randn(2, 5, cfg.d_model)
    out = mrb(x)
    assert (out.gates >= 0).all() and (out.gates <= 1).all()


def test_mrb_rate_deltas_log_spaced():
    cfg = _cfg(num_rates=4)
    deltas = cfg.rate_deltas()
    ratios = [deltas[i + 1] / deltas[i] for i in range(len(deltas) - 1)]
    # Geometric (log-spaced) -> constant ratio.
    for r in ratios:
        assert math.isclose(r, ratios[0], rel_tol=1e-6)


def test_mrb_gradients_flow():
    cfg = _cfg()
    mrb = MultiRateBus(cfg)
    x = torch.randn(2, 6, cfg.d_model, requires_grad=True)
    out = mrb(x)
    out.y.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_mrb_causal_no_future_leakage():
    # Perturbing a token must not change outputs at earlier positions.
    cfg = _cfg()
    mrb = MultiRateBus(cfg)
    mrb.eval()
    x = torch.randn(1, 8, cfg.d_model)
    with torch.no_grad():
        y1 = mrb(x).y
        x2 = x.clone()
        x2[:, 4] += 1.0
        y2 = mrb(x2).y
    assert torch.allclose(y1[:, :4], y2[:, :4])


def test_mrb_effective_horizon_monotonic_decreasing():
    # half-life = ln(2) / Delta. Larger Delta -> faster decay -> SHORTER half-life.
    # So horizons are monotonically *decreasing* across rate groups k=0..K-1
    # (group 0 has the longest memory horizon).
    cfg = _cfg(num_rates=4)
    mrb = MultiRateBus(cfg)
    horizons = mrb.effective_horizon()
    for a, b in zip(horizons, horizons[1:]):
        assert a >= b - 1e-9
