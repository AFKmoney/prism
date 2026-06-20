"""Task: Mini language modelling.

Character-level language modelling on a tiny in-memory corpus. We bundle a
small synthetic corpus (so the task is self-contained and reproducible without
network access) plus support for an optional external text file.

Tests the whole PRISM stack end-to-end on a realistic (if tiny) LM objective.
"""

from __future__ import annotations

from pathlib import Path

import torch

# A small self-contained corpus. Deterministic, public-domain-style text so the
# task is reproducible without downloads. The point is *learnability*, not
# literary quality.
DEFAULT_CORPUS = """\
the cat sat on the mat.
the dog sat on the log.
the cat ran after the dog.
the dog ran after the cat.
a bird flew over the house.
the house was small and warm.
she opened the door and walked in.
the rain fell softly on the roof.
he read a book by the window.
the sun rose over the quiet hills.
birds sang in the early morning.
the river flowed past the old mill.
children played in the green field.
the wind moved through the tall trees.
a fire burned in the stone hearth.
the night was cold and full of stars.
she wrote a letter to her friend.
the boat drifted on the still lake.
mountains stood against the blue sky.
the city never seemed to sleep.
time passed slowly in the small town.
the garden was full of red flowers.
he climbed the hill to watch the sunset.
waves crashed against the rocky shore.
a path led through the dark forest.
the clock struck twelve in the tower.
music filled the room with warmth.
the old bridge crossed the narrow river.
stars gleamed above the sleeping village.
the baker baked bread before dawn.
""".strip()


def build_byte_vocab() -> tuple[dict, int]:
    """Byte-level vocab: 256 bytes + PAD. vocab_size = 257."""
    vocab = {bytes([i]): i + 1 for i in range(256)}  # 0 reserved for PAD
    vocab[b"<pad>"] = 0
    return vocab, 257


def encode_text(text: str) -> list[int]:
    """Encode a string to byte ids (1-indexed, 0 = PAD)."""
    return [b + 1 for b in text.encode("utf-8")]


def load_corpus(path: str | None = None) -> str:
    """Load the corpus from a file, or fall back to the bundled DEFAULT_CORPUS."""
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_CORPUS


def make_dataset(
    text: str,
    seq_len: int,
    device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Encode the corpus and slice it into overlapping chunks of length seq_len+1.

    Returns a 1D LongTensor of token ids. The trainer samples windows from it.
    """
    ids = encode_text(text)
    return torch.tensor(ids, dtype=torch.long, device=device)


def sample_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of contiguous windows for next-token prediction.

    Standard causal LM: input[t] predicts target[t+1]. We return input_ids of
    length seq_len and targets shifted by 1, with full loss_mask=1.
    """
    n = data.shape[0]
    max_start = n - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"corpus too small ({n} tokens) for seq_len={seq_len}")
    starts = torch.randint(0, max_start, (batch_size,), device=device, generator=generator)
    input_ids = torch.stack([data[s : s + seq_len] for s in starts])           # (B, seq_len)
    targets = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])     # (B, seq_len)
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    return input_ids, targets, loss_mask


def description(seq_len: int) -> str:
    return f"mini_lm(seq_len={seq_len}): character-level LM on a small bundled corpus."
