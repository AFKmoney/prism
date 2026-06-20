"""Unit tests for the PRISM model end-to-end."""

from __future__ import annotations

import torch

from prism.config import PrismConfig, MemoryConfig
from prism.model import Prism


def _cfg(**kw) -> PrismConfig:
    base = dict(vocab_size=32, d_model=16, num_layers=3, num_rates=4,
                memory=MemoryConfig(d_mem=8, num_slots=4))
    base.update(kw)
    return PrismConfig(**base)


def test_forward_shape():
    cfg = _cfg()
    m = Prism(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    out = m(ids)
    assert out.logits.shape == (2, 10, cfg.vocab_size)
    assert out.aux_loss.shape == ()


def test_all_parameters_get_gradient():
    cfg = _cfg()
    m = Prism(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    out = m(ids)
    target = torch.randint(0, cfg.vocab_size, (2, 10))
    loss = torch.nn.functional.cross_entropy(
        out.logits.reshape(-1, cfg.vocab_size), target.reshape(-1)
    ) + out.aux_loss
    loss.backward()
    no_grad = [n for n, p in m.named_parameters() if p.grad is None]
    assert no_grad == [], f"params without grad: {no_grad}"


def test_no_nan_gradient():
    cfg = _cfg()
    m = Prism(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    out = m(ids)
    (out.logits.sum() + out.aux_loss).backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"{n}: NaN gradient"
            assert not torch.isinf(p.grad).any(), f"{n}: inf gradient"


def test_carry_memory_across_calls():
    # final_mem of one forward can seed the next.
    cfg = _cfg()
    m = Prism(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    out1 = m(ids)
    assert out1.final_mem.tape.shape == (2, cfg.memory.num_slots, cfg.memory.d_mem)
    out2 = m(ids, mem=out1.final_mem)
    assert out2.logits.shape == (2, 5, cfg.vocab_size)


def test_last_block_has_no_memory_expert():
    cfg = _cfg(num_layers=3)
    m = Prism(cfg)
    last_kinds = [e.expert_type for e in m.blocks[-1].router.experts]
    assert "memory" not in last_kinds
    non_last_kinds = [e.expert_type for e in m.blocks[0].router.experts]
    assert "memory" in non_last_kinds


def test_param_count_positive():
    cfg = _cfg()
    m = Prism(cfg)
    assert m.num_parameters() > 0


def test_deterministic_in_eval_mode():
    cfg = _cfg()
    m = Prism(cfg)
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        a = m(ids).logits
        b = m(ids).logits
    assert torch.allclose(a, b)
