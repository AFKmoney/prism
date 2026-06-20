"""Task: Induction (associative lookup).

A sequence of (key, value) pairs is presented, then a separator, then a query
key that appeared earlier. The model must output the value that was paired with
that key.

Layout::

    tokens: [k1 v1 k2 v2 ... kn vn] [SEP] [q] [PAD]
                                            ^
                                  model must emit v(q) here

This probes the Symbolic expert's ``select``/``compare`` primitives plus the
Memory expert's retrieval: the model must learn the lookup operation.

Label convention (shared): ``targets[t]`` is what the model emits at position
``t`` given ``0..t-1``; ``loss_mask[t]`` marks where loss counts.
"""

from __future__ import annotations

import torch

PAD = 0
SEP = 1
DATA_OFFSET = 2


def generate_batch(
    batch_size: int,
    num_pairs: int,
    vocab_size: int,
    device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one batch of induction examples.

    Args:
        batch_size: B.
        num_pairs: n, number of (key, value) pairs shown before the query.
        vocab_size: total vocab (keys and values drawn from [DATA_OFFSET, vocab)).

    Returns:
        input_ids: (B, 2n + 2)  — pairs + SEP + query
        targets:   (B, 2n + 2)
        loss_mask: (B, 2n + 2)  — 1 only at the final position
    """
    assert vocab_size > DATA_OFFSET
    data_vocab = vocab_size - DATA_OFFSET
    # Keys and values share the same data vocab; keys are distinct within an example
    # so the lookup is well-defined.
    pairs = torch.empty(batch_size, 2 * num_pairs, device=device, dtype=torch.long)
    keys = torch.empty(batch_size, num_pairs, device=device, dtype=torch.long)
    values = torch.empty(batch_size, num_pairs, device=device, dtype=torch.long)
    for b in range(batch_size):
        k = torch.randperm(data_vocab, generator=generator)[:num_pairs] + DATA_OFFSET
        v = torch.randint(0, data_vocab, (num_pairs,), generator=generator) + DATA_OFFSET
        keys[b] = k
        values[b] = v
        pairs[b, 0::2] = k
        pairs[b, 1::2] = v

    sep = torch.full((batch_size, 1), SEP, device=device, dtype=torch.long)
    # Query: pick a random one of the n keys (different per example).
    query_idx = torch.randint(0, num_pairs, (batch_size,), device=device, generator=generator)
    query = keys[torch.arange(batch_size), query_idx].unsqueeze(1)            # (B,1)
    answer = values[torch.arange(batch_size), query_idx].unsqueeze(1)         # (B,1)
    pad = torch.full((batch_size, 1), PAD, device=device, dtype=torch.long)

    # Sequence: [pairs...] [SEP] [query] [PAD]
    # The model emits the answer at the PAD position (last), having seen the query.
    input_ids = torch.cat([pairs, sep, query, pad], dim=1)                    # (B, 2n+2)

    targets = torch.full_like(input_ids, PAD)
    targets[:, -1] = answer.squeeze(1)                                        # answer at last pos

    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_mask[:, -1] = 1.0
    return input_ids, targets, loss_mask


def description(num_pairs: int) -> str:
    return f"induction(n={num_pairs}): given {num_pairs} (key,value) pairs + a query key, emit the paired value."
