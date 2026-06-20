"""Unified training harness for PRISM and baselines.

Usage::

    py -3.13 -m prism.train --model prism --task copy --steps 1000
    py -3.13 -m prism.train --model transformer --task induction --steps 500
    py -3.13 -m prism.train --compare --steps 800      # run all models on a task

The harness is deliberately simple and dependency-free: it prints a compact
one-line-per-N-steps progress table to stdout and writes a final JSON summary.
No wandb, no accelerators — CPU-only and fully reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from prism.baselines import build_model
from prism.config import MemoryConfig, PrismConfig
from prism.model import Prism


# ---------------------------------------------------------------------------
# Task adapters: each returns a callable (step -> (input_ids, targets, mask))
# and a description string. The config's vocab_size is overridden per task.
# ---------------------------------------------------------------------------


def make_task(task: str, batch_size: int, device, seed: int):
    """Return (sample_fn, description, effective_vocab_size)."""
    g = torch.Generator(device=device).manual_seed(seed)
    if task == "copy":
        seq_half = 8
        vocab = 18  # PAD, SEP, + 16 data tokens

        def sample():
            from tasks import copy

            return copy.generate_batch(batch_size, seq_half, vocab, device, g)

        from tasks import copy as _c

        return sample, _c.description(seq_half), vocab
    if task == "induction":
        # Kept small enough to be learnable in a CPU budget: in-context lookup
        # is hard, so 3 pairs over a 12-token vocab is the sweet spot where the
        # task is non-trivial but learnable in a few hundred steps.
        num_pairs = 3
        vocab = 12

        def sample():
            from tasks import induction

            return induction.generate_batch(batch_size, num_pairs, vocab, device, g)

        from tasks import induction as _i

        return sample, _i.description(num_pairs), vocab
    if task == "mini_lm":
        from tasks import mini_lm

        seq_len = 48
        _, vocab = mini_lm.build_byte_vocab()
        corpus = mini_lm.load_corpus(None)
        data = mini_lm.make_dataset(corpus, seq_len, device)

        def sample():
            return mini_lm.sample_batch(data, batch_size, seq_len, device, g)

        return sample, mini_lm.description(seq_len), vocab
    raise ValueError(f"unknown task: {task!r}")


# ---------------------------------------------------------------------------
# Core train loop
# ---------------------------------------------------------------------------


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over masked positions only."""
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none", ignore_index=0)
    loss = loss.reshape(B, T)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(-1)
    correct = (preds == targets).float() * mask
    return float(correct.sum() / mask.sum().clamp(min=1.0))


def train_one(
    model_name: str,
    task: str,
    steps: int,
    lr: float,
    batch_size: int,
    log_every: int,
    seed: int,
    device: str = "cpu",
) -> dict:
    """Train a single model on a single task. Returns a metrics dict."""
    torch.manual_seed(seed)
    sample, desc, vocab = make_task(task, batch_size, device, seed)

    # Build a config sized for CPU; vocab overridden by the task.
    cfg = PrismConfig(
        vocab_size=vocab,
        d_model=64,
        num_layers=3,
        num_rates=4,
        memory=MemoryConfig(d_mem=64, num_slots=16),
        router_load_balance_weight=0.01,
    )
    model = build_model(model_name, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Linear warmup (10% of steps) then cosine decay. The pure-cosine-from-step-0
    # schedule was too aggressive early and starved the router of gradient signal;
    # warmup lets the load-balancing loss stabilize routing first.
    warmup_steps = max(1, steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    n_params = model.num_parameters()
    history = []
    t0 = time.time()
    print(f"\n=== {model_name} on {task} === ({n_params} params)")
    print(f"    {desc}")

    for step in range(steps):
        input_ids, targets, mask = sample()
        if hasattr(model, "forward") and model_name == "prism":
            out = model(input_ids)
            logits, aux = out.logits, out.aux_loss
        else:
            out = model(input_ids)
            logits, aux = out.logits, out.aux_loss
        loss = masked_cross_entropy(logits, targets, mask) + aux
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                acc = accuracy(logits, targets, mask)
                ppl = math.exp(min(float(masked_cross_entropy(logits, targets, mask)), 20.0))
            rec = {
                "step": step,
                "loss": float(loss.detach()),
                "ce": float(masked_cross_entropy(logits, targets, mask).detach()),
                "acc": acc,
                "ppl": ppl,
                "lr": float(opt.param_groups[0]["lr"]),
            }
            history.append(rec)
            print(f"    step {step:5d} | ce {rec['ce']:.4f} | acc {acc:.3f} | ppl {ppl:7.2f} | lr {rec['lr']:.4f}")

    elapsed = time.time() - t0
    final = history[-1]
    summary = {
        "model": model_name,
        "task": task,
        "params": n_params,
        "steps": steps,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_s": round(steps * batch_size * 33 / max(elapsed, 1e-9), 1),  # approx
        "final_ce": final["ce"],
        "final_acc": final["acc"],
        "final_ppl": final["ppl"],
    }
    print(f"    done in {elapsed:.1f}s — final ce {final['ce']:.4f} acc {final['acc']:.3f}")
    return {"summary": summary, "history": history}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PRISM training harness")
    p.add_argument("--model", choices=["prism", "transformer", "ssm"], default="prism")
    p.add_argument("--task", choices=["copy", "induction", "mini_lm"], default="copy")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", action="store_true", help="run all three models on the task")
    p.add_argument("--out", type=str, default=None, help="write JSON summary to this path")
    args = p.parse_args(argv)

    models = ["prism", "transformer", "ssm"] if args.compare else [args.model]
    results = {}
    for m in models:
        results[m] = train_one(m, args.task, args.steps, args.lr, args.batch_size, args.log_every, args.seed)

    if args.compare:
        print("\n=== COMPARISON ===")
        print(f"{'model':<14} {'params':>10} {'ce':>8} {'acc':>8} {'ppl':>9} {'time':>8}")
        for m, r in results.items():
            s = r["summary"]
            print(f"{m:<14} {s['params']:>10} {s['final_ce']:>8.4f} {s['final_acc']:>8.3f} {s['final_ppl']:>9.2f} {s['elapsed_s']:>7.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nsummary written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
