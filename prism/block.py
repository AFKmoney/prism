"""PRISM Block.

One block = MRB temporal mixing + polymorphic routing, with residuals and
RMSNorm. The shared memory tape flows through every block unchanged except
for the updates the Memory expert writes.
"""

from __future__ import annotations

import torch
from torch import nn

from prism.config import PrismConfig
from prism.experts import ExpertStats
from prism.memory import MemoryState
from prism.mrb import MultiRateBus
from prism.norm import RMSNorm
from prism.router import PolymorphicRouter


class PrismBlock(nn.Module):
    def __init__(self, config: PrismConfig, expert_types: list[str] | None = None) -> None:
        super().__init__()
        self.config = config
        self.mrb = MultiRateBus(config)
        self.mrb_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        # Allow per-block expert pools. The last block typically omits 'memory'
        # because its writes would never be read by a downstream block.
        block_config = config
        if expert_types is not None:
            from dataclasses import replace

            block_config = replace(config, expert_types=tuple(expert_types))
        self.router = PolymorphicRouter(block_config)
        self.expert_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(
        self, x: torch.Tensor, mem: MemoryState
    ) -> tuple[torch.Tensor, MemoryState, ExpertStats, torch.Tensor]:
        """Args:
            x: (B, T, d_model)
            mem: shared memory state

        Returns:
            out: (B, T, d_model)
            new_mem: updated memory state
            stats: merged expert stats
            aux_loss: scalar load-balancing loss for this block
        """
        # MRB temporal mixing with residual (norm the input to the bus first).
        mrb_out = self.mrb(self.mrb_norm(x))
        x = x + mrb_out.y

        # Polymorphic routing.
        h = self.expert_norm(x)
        expert_out, new_mem, stats, aux_loss = self.router(h, mem)
        x = x + expert_out

        return x, new_mem, stats, aux_loss
