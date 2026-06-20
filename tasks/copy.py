"""Task: Copy.

The model must reproduce a random token sequence after a separator.

Layout (length = 2k + 1)::

    positions:   0 .. k-1    k        k+1 .. 2k
    tokens:      a_1..a_k    SEP      PAD PAD ... PAD

Label convention (shared across all tasks):
    * ``targets[t]`` is the token the model should emit at position ``t`` given
      tokens ``0..t-1``. The model's logit at position ``t-1`` predicts
      ``targets[t]``.
    * ``loss_mask[t]`` is 1 where the loss counts, 0 where it is ignored.

For copy, the model should output ``a_1`` at the SEP position (having seen all
of a_1..a_k), then ``a_2`` at the next position, etc. So::

    targets[ k ]      = a_1     (loss_mask = 1)
    targets[ k+1 ]    = a_2     (loss_mask = 1)
    ...
    targets[ 2k ]     = a_k     (loss_mask = 1)

Loss is computed only on the k copy positions.
"""

from __future__ import annotations

import torch

# Reserved token ids.
PAD = 0
SEP = 1
DATA_OFFSET = 2  # data tokens live in [2, vocab_size)


def generate_batch(
    batch_size: int,
    seq_half: int,
    vocab_size: int,
    device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one batch of copy examples.

    Returns:
        input_ids: (B, 2k+1)
        targets:   (B, 2k+1) with PAD where ignored
        loss_mask: (B, 2k+1) float, 1.0 on the k copy positions
    """
    assert vocab_size > DATA_OFFSET
    data_vocab = vocab_size - DATA_OFFSET
    data = torch.randint(0, data_vocab, (batch_size, seq_half), device=device, generator=generator) + DATA_OFFSET
    sep = torch.full((batch_size, 1), SEP, device=device, dtype=torch.long)
    pad = torch.full((batch_size, seq_half), PAD, device=device, dtype=torch.long)

    input_ids = torch.cat([data, sep, pad], dim=1)            # (B, 2k+1)

    targets = torch.full_like(input_ids, PAD)
    # At SEP position (k), emit a_1; then a_2 ... a_k.
    targets[:, seq_half : seq_half + seq_half] = data

    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_mask[:, seq_half : seq_half + seq_half] = 1.0
    return input_ids, targets, loss_mask


def description(seq_half: int) -> str:
    return f"copy(k={seq_half}): reproduce a random {seq_half}-token sequence after a separator."
