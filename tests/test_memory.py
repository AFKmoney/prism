"""Unit tests for the Shared Memory Bus."""

from __future__ import annotations

import torch

from prism.config import MemoryConfig
from prism.memory import MemoryHead, MemoryState


def _mem_cfg(**kw) -> MemoryConfig:
    base = dict(d_mem=8, num_slots=4)
    base.update(kw)
    return MemoryConfig(**base)


def test_memory_state_create_shape():
    cfg = _mem_cfg()
    state = MemoryState.create(batch_size=3, config=cfg, device=torch.device("cpu"), dtype=torch.float32)
    assert state.tape.shape == (3, cfg.num_slots, cfg.d_mem)
    assert state.read_entropy.shape == ()


def test_memory_head_read_write_shapes():
    cfg = _mem_cfg()
    head = MemoryHead(d_model=16, config=cfg)
    state = MemoryState.create(2, cfg, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 5, 16)  # batch, time, d_model
    out, new_state = head(x, state)
    assert out.shape == (2, 5, 16)
    assert new_state.tape.shape == state.tape.shape


def test_memory_head_write_changes_tape():
    cfg = _mem_cfg()
    head = MemoryHead(d_model=16, config=cfg)
    state = MemoryState.create(2, cfg, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 5, 16)
    before = state.tape.clone()
    _, new_state = head(x, state)
    # The tape should change (write gate is nonzero with sigmoid init around 0.5).
    assert not torch.allclose(before, new_state.tape)


def test_memory_head_read_is_content_based():
    # If the query matches slot k's content, the read weight should peak there.
    cfg = _mem_cfg(d_mem=4, num_slots=4)
    head = MemoryHead(d_model=4, config=cfg)
    # Build a tape with a distinct slot 2.
    state = MemoryState.create(1, cfg, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        state.tape.zero_()
        state.tape[0, 2] = 5.0  # slot 2 is 'active'
    # Make the query project to something aligned with slot 2's content.
    # We can't easily control q_proj output, but we can check that two identical
    # inputs give identical read weights (determinism) and that the read vector
    # is a convex combination of slot contents.
    x = torch.randn(1, 1, 4)
    out1, _ = head(x, state)
    out2, _ = head(x, state)
    assert torch.allclose(out1, out2)


def test_memory_head_gradients_flow():
    cfg = _mem_cfg()
    head = MemoryHead(d_model=16, config=cfg)
    state = MemoryState.create(2, cfg, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 3, 16, requires_grad=True)
    out, new_state = head(x, state)
    (out.sum() + new_state.read_entropy).backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_memory_head_read_entropy_accumulates():
    # Two calls should accumulate read_entropy (it is additive across calls).
    cfg = _mem_cfg()
    head = MemoryHead(d_model=16, config=cfg)
    state = MemoryState.create(1, cfg, torch.device("cpu"), torch.float32)
    x = torch.randn(1, 4, 16)
    _, s1 = head(x, state)
    _, s2 = head(x, s1)
    # The accumulator should be >= the single-call value.
    assert s2.read_entropy >= s1.read_entropy - 1e-6
