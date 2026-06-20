# PRISM — Architecture Reference

This document specifies PRISM at the level needed to reimplement it from
scratch. For the design rationale and alternatives considered, see the spec at
`../../docs/superpowers/specs/2026-06-19-prism-design.md`.

## 1. The unifying idea

Most language models apply one fixed computation per layer. PRISM routes each
token to one of three **structurally different** computations, all sharing one
differentiable memory tape:

| Expert | Computation | When a token needs it |
|---|---|---|
| **Neural** (N) | SwiGLU MLP transform | "transform this feature" |
| **Memory** (M) | read/write the shared tape | "store this" / "recall this" |
| **Symbolic** (S) | compose differentiable primitives | "compare/select/lookup" |

The router learns *which kind of operation* each token requires. This is a
heterogeneous MoE — unlike standard MoEs where every expert is an identical MLP.

## 2. Multi-Rate Bus (MRB)

Replaces self-attention. Sub-quadratic: O(n·d) time, O(K·d) state.

### Decay schedule (fixed inductive bias)

K groups, each with a retention γ_k derived from a geometrically-spaced decay:

```
Δ_k = exp(log(Δ_0) + (log(Δ_max) - log(Δ_0)) · k/(K-1))   # log-spaced
γ_k = exp(-Δ_k)                                             # retention
```

Defaults: Δ_0 = ln 2 (half-life 1 step), Δ_max = 8·ln 2 (half-life 8 steps).
Group 0 forgets fast (local context); group K-1 retains long (global context).

### Per-token update (group k, width d_rate = d_model // K)

```
z_k(t) = W_k · x(t) + b_k                       # input projection
h_k(t) = γ_k · h_k(t-1) + (1 - γ_k) · z_k(t)    # decayed accumulation
y(t)   = Σ_k  g_k(t) · h_k(t)                    # gated readout
g_k(t) = sigmoid(W_g · x(t) + b_g)_k             # learned scale gate
```

### Vectorized form

For a sequence, the recurrence `h_k(t) = γ·h_k(t-1) + z_k(t)` has the closed
form:

```
h_k(t) = γ_k^t · Σ_{s≤t} γ_k^{-s} · z_k(s)
```

computed as: scale inputs by `γ^{-s}`, cumsum along time, multiply by `γ^t`.
Done in float64 for numerical headroom. No Python loop over time.

### The gate is the interpretable part

`g_k(t)` tells you, for token `t`, how much it relies on the k-th temporal
scale. A token that mostly reads group K-1 is using long-range memory; one
reading group 0 is local. This is directly inspectable after training.

### Relation to prior work

| Model | Temporal mechanism |
|---|---|
| Transformer | global attention, O(n²) |
| Mamba | single input-dependent Δ (one dynamic rate) |
| RetNet | single γ, exponential mask |
| RWKV | time/channel mix, fixed per-layer rate |
| **PRISM MRB** | **structured K-rate decomposition + learned per-token scale gate** |

## 3. Shared Memory Bus

A single tape `S × d_mem` flows through every layer and time step. It is the
Global Workspace — how heterogeneous experts communicate.

### Read (content-based, by the Memory expert)

```
q    = W_q · x                          # query per position
score[b,m,s] = q[b,m] · tape[b,s] / √d_mem
w    = softmax(score, over s)           # (B, M, S) read weights
read = Σ_s w[b,m,s] · tape[b,s]         # retrieved vector
```

### Write (NTM-style gated erase+add)

```
v     = W_v · x                         # value to write
w_g   = sigmoid(W_wg · x)               # write strength (B, M, 1)
e_g   = sigmoid(W_eg · x)               # erase mask  (B, M, d_mem)
# aggregated over positions (mean) to keep the tape bounded:
tape  ← tape · (1 - mean(w_g·e_g)) + mean(w_g·v)
```

### Why a shared tape

The tape is mutated in place across the L blocks within one forward pass, and
carried across time steps via `final_mem`. Layer 3 can read what layer 1 wrote.
This makes memory a *communication channel* between expert kinds, not just a
per-layer buffer.

### Initialization

The tape starts at **zeros** (working memory is empty until written). This makes
inference deterministic and is semantically correct. Initial random content was
tried and rejected — it added noise without benefit.

### Regularizer

Read-distribution entropy is tracked and encouraged (`-mean(entropy)` added to
the loss with weight 0.01) to prevent collapse onto a single slot.

## 4. Polymorphic Router

Top-1 routing with epsilon-soft straight-through estimation.

### Routing input

The gate sees `[x ‖ mean(tape)]` — the token feature **and** a summary of
memory state — so it can decide "I need to retrieve" vs "I need to transform".

### Selection

```
logits = W_gate · [x ‖ mem_summary]            # (..., num_experts)
p_soft = softmax(logits)
top    = argmax(logits)
onehot = one_hot(top)
p_st   = onehot + p_soft - p_soft.detach()     # straight-through
p      = (1 - ε) · p_st + ε · p_soft           # epsilon-soft blend
```

### Why epsilon-soft

Pure hard top-1 routing produces **dead experts** (an expert never selected
gets no gradient, stays dead forever). Blending in a small ε of the soft
distribution (default 0.05) guarantees every expert receives gradient every
step. ε can be annealed to 0 for pure symbolic behaviour at inference.

### Output combination

All experts run; their outputs are weighted by `p` and summed. The Memory
expert's resulting tape is blended into the shared tape by the fraction of
tokens that selected it.

### Load-balancing loss (Switch Transformer style)

```
f_i = (tokens routed to expert i) / (total tokens)
P_i = mean routing probability for expert i
aux = num_experts · Σ_i f_i · P_i               # minimized when uniform
```

## 5. Symbolic Expert

A closed library of 6 differentiable, typed primitives. Per token, a
distribution `p` over primitives is learned; the output is the `p`-weighted sum
of primitive outputs, then projected.

### Primitive library

| Primitive | Arity | Operation | Differentiable semantics |
|---|---|---|---|
| `compare` | 2 | cosine sim of two halves of x, broadcast | internal-consistency detector |
| `gate` | 3 | `σ(c)·x + (1-σ(c))·b` | differentiable IF/ELSE |
| `select` | 2 | `Σ softmax(m)_i · x_i` broadcast | soft gather |
| `shift` | 1 | cyclic rotate of feature dim | positional transform |
| `threshold` | 2 | `x · σ(5(x-t))` | smooth step (differentiable `>`) |
| `count` | 1 | `mean(σ(x))` broadcast | count active dims |

### Why this is neuro-symbolic *in the weights*

There is no external solver. The primitives are pure tensor ops, fully
differentiable. The selection is hardened via straight-through in the forward
pass (one primitive dominates) while gradients flow through the soft
distribution. The model *learns when to reason* — on the induction task, the
last block routes ~73% to the symbolic expert.

## 6. Block and model assembly

### One block

```
x → MRB(RMSNorm(x)) → +residual → Router(RMSNorm) → +expert_out → +residual
                                          ↕ read/write
                                     Shared Memory Bus
```

### Full model

```
input_ids → Embedding → [Block₁ … Block_L] → RMSNorm → Linear(tied) → logits
                              ↕
                    one Memory tape carried through all blocks
```

### Last block drops the memory expert

The final block's memory writes would go into a tape no downstream block reads,
so those write weights would receive no gradient (dead parameters). Dropping
the memory expert from the last block keeps **every parameter live**. This is a
small but important detail — verified by `test_all_parameters_get_gradient`.

## 7. Configuration

All hyperparameters in `prism/config.py` as dataclasses (`PrismConfig`,
`MemoryConfig`). Toy defaults are CPU-friendly (~186k–201k params).

Key knobs:
- `d_model`, `num_layers`, `num_rates`: backbone size.
- `expert_types`: which expert kinds live in each router. Set to
  `("neural", "memory")` for HERA-lite (the documented fallback if symbolic
  proves unstable).
- `memory.d_mem`, `memory.num_slots`: tape geometry.
- `router.epsilon`: soft-routing blend (anneal toward 0 for pure top-1).

## 8. Testing strategy

33 unit tests in `tests/`, covering:
- **Shapes**: every component produces the declared output shape.
- **Gradient flow**: every parameter receives a non-NaN gradient.
- **Causality**: MRB output at position t doesn't depend on tokens after t.
- **Determinism**: identical inputs give identical outputs in eval mode.
- **Numerics**: MRB steady-state, load-balancing loss non-negativity.
- **Semantics**: neural expert doesn't touch memory; router doesn't collapse.

The test suite is the contract: if a refactor breaks any test, the refactor is
wrong, not the test.

## 9. Known limitations

- **Toy scale only.** Training a competitive PRISM needs GPUs and data —
  out of scope for this prototype (by design; see spec §1).
- **Copy and dense LM are not PRISM's strength.** SSMs win there. PRISM's value
  is on compositional/lookup reasoning (see RESULTS.md).
- **Sequential MRB scan in float64.** Fine for T ≤ 512; for long sequences a
  chunked scan would be needed.
- **Symbolic ST can be slow to converge.** Anneal ε to 0 late in training for
  cleaner symbolic behaviour.
