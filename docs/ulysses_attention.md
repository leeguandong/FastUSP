# Ulysses Attention: Head-Parallel Distributed Attention

## Problem Statement

In standard Transformer models, the self-attention mechanism requires access to the **full sequence** to compute attention scores. When training or running inference on large models across multiple GPUs, the input data is typically sharded along the sequence dimension — each GPU holds only a fragment of the full sequence. This creates a fundamental conflict: attention needs global context, but each GPU only has local data.

A naive approach would be to gather the full sequence on every GPU before computing attention, but this defeats the purpose of distribution and introduces massive communication overhead.

## Core Insight

Multi-Head Attention (MHA) computes each head **independently**. The output of head `h` depends only on `Q_h`, `K_h`, and `V_h` — it never interacts with other heads during the attention computation itself. This independence is the key insight behind Ulysses Attention:

> **Instead of gathering sequences, redistribute the data so that each GPU holds all sequence tokens but only a subset of attention heads.**

This transforms the problem from **sequence parallelism** to **head parallelism** using a single All-to-All collective communication.

## Data Transformation

Suppose we have `N` GPUs, `H` attention heads, sequence length `S`, and head dimension `D`.

**Before All-to-All** — data is sharded along the sequence dimension:

```
Each GPU holds: [B, H, S/N, D]
  - Full heads (all H)
  - Partial sequence (S/N tokens)
```

**After All-to-All** — data is sharded along the head dimension:

```
Each GPU holds: [B, H/N, S, D]
  - Partial heads (H/N heads)
  - Full sequence (all S tokens)
```

### 4-GPU Example (H=8, S=1024)

```
Initial Distribution (sequence-sharded):
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0: Heads [0..7], Tokens [0..255]     → shape [B, 8, 256, D]│
│ GPU 1: Heads [0..7], Tokens [256..511]   → shape [B, 8, 256, D]│
│ GPU 2: Heads [0..7], Tokens [512..767]   → shape [B, 8, 256, D]│
│ GPU 3: Heads [0..7], Tokens [768..1023]  → shape [B, 8, 256, D]│
└─────────────────────────────────────────────────────────────────┘
                            │
                       All-to-All
                            │
                            ▼
Target Distribution (head-sharded):
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0: Heads [0..1], Tokens [0..1023]    → shape [B, 2, 1024, D]│
│ GPU 1: Heads [2..3], Tokens [0..1023]    → shape [B, 2, 1024, D]│
│ GPU 2: Heads [4..5], Tokens [0..1023]    → shape [B, 2, 1024, D]│
│ GPU 3: Heads [6..7], Tokens [0..1023]    → shape [B, 2, 1024, D]│
└─────────────────────────────────────────────────────────────────┘
```

Each GPU now has the **complete sequence** for its assigned heads and can compute attention locally without any further communication.

## Algorithm

```
Input:  Q_local, K_local, V_local   — each of shape [B, H, S/N, D] on GPU i

Step 1: Input All-to-All
    Q_heads = AlltoAll(Q_local)      — shape becomes [B, H/N, S, D]
    K_heads = AlltoAll(K_local)      — shape becomes [B, H/N, S, D]
    V_heads = AlltoAll(V_local)      — shape becomes [B, H/N, S, D]

Step 2: Local Attention (standard scaled dot-product attention)
    For each head h in the local subset:
        A_h = softmax(Q_h × K_h^T / √D)
        O_h = A_h × V_h
    O_heads = stack(O_h)             — shape [B, H/N, S, D]

Step 3: Output All-to-All
    O_local = AlltoAll(O_heads)      — shape becomes [B, H, S/N, D]

Output: O_local                      — same layout as input
```

The entire algorithm consists of two All-to-All communication steps sandwiching a standard local attention computation. No special numerical tricks are needed — the attention kernel is identical to single-GPU attention.

## Mathematical Correctness

Ulysses Attention produces results **identical** to standard multi-head attention. Here is why:

1. **Head Independence**: In MHA, the output for head `h` is:

   ```
   O_h = softmax(Q_h × K_h^T / √D) × V_h
   ```

   This computation involves only `Q_h`, `K_h`, `V_h` — no cross-head interaction.

2. **All-to-All is pure data redistribution**: It does not modify, approximate, or drop any values. It is a lossless permutation of data across GPUs.

3. **Combining these two facts**: Each GPU computes exact attention for its subset of heads over the full sequence. The final reverse All-to-All reassembles the output in the original sequence-sharded layout. The end result is bit-for-bit identical to single-GPU attention.

There is no approximation, no numerical error beyond floating-point non-determinism from reordering operations.

## Complexity Analysis

| Metric | Formula | Notes |
|--------|---------|-------|
| Compute per GPU | O(B × (H/N) × S² × D) | Each GPU handles H/N heads over full sequence |
| Communication volume | O(4 × B × H × S × D / N) | All-to-All for Q, K, V (input) + O (output) |
| Communication rounds | 2 | One input All-to-All + one output All-to-All |
| Memory per GPU | O(B × (H/N) × S² + B × H × S × D / N) | Attention matrix + activations |

The communication cost scales as **O(1/N)** per GPU — adding more GPUs reduces per-GPU communication proportionally. The total number of communication rounds is always exactly **2**, regardless of GPU count.

## Constraint

Ulysses Attention requires:

```
H % N == 0
```

The number of attention heads `H` must be evenly divisible by the number of GPUs `N`. This is because heads are distributed equally across GPUs. If this condition is not met, Ulysses Attention cannot be used directly.

In practice, most Transformer models have head counts that are powers of 2 or multiples of common GPU counts (2, 4, 8), so this constraint is rarely a problem.

## Comparison: Ulysses Attention vs Ring Attention

| Aspect | Ulysses Attention | Ring Attention |
|--------|-------------------|----------------|
| Parallelism type | Head parallelism | Sequence parallelism |
| Communication pattern | All-to-All | Point-to-point ring pass |
| Communication rounds | 2 (fixed) | N-1 (scales with GPU count) |
| Communication volume | O(B×H×S×D/N) per round | O(B×H×S×D/N) per round |
| Overlap potential | Limited (All-to-All is blocking) | High (compute overlaps with KV transfer) |
| Constraint | H % N == 0 | None |
| Sequence length scaling | Full sequence on each GPU | Sequence split across GPUs |
| Memory efficiency | Each GPU stores full S² attention for H/N heads | Each GPU stores (S/N)×S attention blocks |
| Mathematical exactness | Exact | Exact (with online softmax) |
| Best suited for | Moderate GPU counts where H ≥ N | Very long sequences, large GPU counts |
| Implementation complexity | Simple (two All-to-All calls) | Moderate (ring topology + online softmax) |

Ulysses Attention excels when the number of heads is large relative to the GPU count and when low communication latency is important (only 2 rounds). Ring Attention is preferable when sequences are extremely long and memory per GPU is the bottleneck, or when the head count is not divisible by the GPU count.
