"""PRISM — Polymorphic Recurrent Intelligence with Shared Memory.

A novel language model architecture that unifies four paradigms under one
abstraction:

* **Sub-quadratic continuous backbone** — the Multi-Rate Bus (MRB), a bank of
  recurrent filters at logarithmically-spaced decay rates with a learned
  per-token scale gate.
* **Polymorphic MoE** — a router selects which *kind* of computation a token
  undergoes: Neural (MLP), Memory (read/write head), or Symbolic (a library of
  differentiable primitives).
* **Shared differentiable memory bus** — a single memory tape flows through all
  layers and time steps, acting as the Global Workspace through which the
  heterogeneous experts communicate.
* **Differentiable symbolic reasoning** — typed primitives are soft-selected
  and composed end-to-end inside the MoE router.

See ``docs/ARCHITECTURE.md`` and the design spec at
``../docs/superpowers/specs/2026-06-19-prism-design.md`` for full details.
"""

from prism.config import PrismConfig, MemoryConfig
from prism.model import Prism

__all__ = ["PrismConfig", "MemoryConfig", "Prism"]
__version__ = "0.1.0"
