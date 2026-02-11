# Unified Sequence Parallelism (USP): Combining Ulysses and Ring Attention

## Core Goal

Ulysses Attention and Ring Attention each have distinct strengths and limitations:

- **Ulysses** is communication-efficient (only 2 rounds) but requires `H % N == 0` and keeps the full sequence on each GPU.
- **Ring** scales to arbitrary GPU counts and very long sequences but requires `N-1` communication rounds.

**Unified Sequence Parallelism (USP)** combines both strategies in a **2D mesh** to get the best of both worlds. GPUs are organized into a grid where one dimension uses Ulysses (head parallelism) and the other uses Ring (sequence parallelism). This allows scaling beyond the head count limit while maintaining Ulysses's communication efficiency where possible.

## 2D Mesh Design

Given `N` total GPUs, USP factors them into a 2D mesh of shape `(R, U)` where `N = R × U`:

```
N GPUs = R (Ring dimension) × U (Ulysses dimension)

Example: 8 GPUs with mesh=(2, 4)
         Ring groups of size 2, Ulysses groups of size 4
```

- **Ulysses dimension (U)**: GPUs within the same Ulysses group perform All-to-All head redistribution. Requires `H % U == 0`.
- **Ring dimension (R)**: GPUs within the same Ring group pass KV chunks in a ring. No head count constraint.

## State Machine Design

USP implements the combined strategy through a **recursive `__torch_function__` interception** mechanism. When a model calls `scaled_dot_product_attention`, USP intercepts the call and processes it through a layered state machine:

```
Model calls: F.scaled_dot_product_attention(Q, K, V)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Layer 1: Ulysses Interceptor                │
│                                             │
│  1. Input All-to-All (within Ulysses group) │
│     [B, H, S_local, D] → [B, H/U, S', D]  │
│  2. Call attention (intercepted again)  ─────┼──┐
│  3. Output All-to-All (reverse)             │  │
│     [B, H/U, S', D] → [B, H, S_local, D]  │  │
└─────────────────────────────────────────────┘  │
                                                  │
    ┌─────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────┐
│ Layer 2: Ring Interceptor                   │
│                                             │
│  1. Initialize O=0, LSE=-inf               │
│  2. For step = 0 to R-1:                   │
│     a. Call attention (intercepted again) ───┼──┐
│     b. Merge O, LSE with online softmax     │  │
│     c. Ring-pass KV to next GPU             │  │
│  3. Return merged O                         │  │
└─────────────────────────────────────────────┘  │
                                                  │
    ┌─────────────────────────────────────────────┘
    ▼
┌─────────────────────────────────────────────┐
│ Layer 3: Actual Attention Kernel            │
│                                             │
│  FlashAttention / standard SDPA             │
│  Computes: softmax(Q×K^T/√D) × V           │
│  Returns: O_chunk, LSE_chunk                │
└─────────────────────────────────────────────┘
```

### How It Works

1. **Layer 1 — Ulysses Dispatch**: The original attention call is intercepted. USP performs the input All-to-All within the Ulysses group, transforming `[B, H, S/N, D]` → `[B, H/U, S/R, D]` (heads redistributed, sequence still partitioned by Ring dimension).

2. **Layer 2 — Ring Orchestration**: The attention call (now with redistributed heads) is intercepted again. USP initiates the ring pass protocol within the Ring group, iterating through KV chunks from peer GPUs.

3. **Layer 3 — Kernel Execution**: The actual attention kernel (e.g., FlashAttention) executes on the local data. This is the real computation — no more interception.

The results propagate back up: Ring merges partial outputs using online softmax, then Ulysses performs the output All-to-All to restore the original data layout.

## 4-GPU Example: mesh=(2, 2)

### GPU Topology

```
4 GPUs arranged in a 2×2 mesh:

              Ulysses dim (U=2)
              ◄──────────────►
         ┌──────────┬──────────┐
    ▲    │  GPU 0   │  GPU 1   │   Ulysses Group 0: {GPU 0, GPU 1}
    │    │          │          │   Ulysses Group 1: {GPU 2, GPU 3}
  Ring   ├──────────┼──────────┤
  dim    │  GPU 2   │  GPU 3   │   Ring Group 0: {GPU 0, GPU 2}
  (R=2)  │          │          │   Ring Group 1: {GPU 1, GPU 3}
    ▼    └──────────┴──────────┘

Ulysses groups (horizontal): All-to-All within {0,1} and {2,3}
Ring groups (vertical):      Ring pass within {0,2} and {1,3}
```

### Data Flow

```
Given: H=8 heads, S=1024 tokens, 4 GPUs, mesh=(R=2, U=2)

Initial state (sequence-sharded, each GPU has S/4 = 256 tokens):
  GPU 0: [B, 8, 256, D]  (tokens 0-255)
  GPU 1: [B, 8, 256, D]  (tokens 256-511)
  GPU 2: [B, 8, 256, D]  (tokens 512-767)
  GPU 3: [B, 8, 256, D]  (tokens 768-1023)

Step 1 — Ulysses All-to-All (within Ulysses groups):
  Ulysses Group {0,1}: redistribute heads
    GPU 0: [B, 4, 512, D]  (heads 0-3, tokens 0-511)
    GPU 1: [B, 4, 512, D]  (heads 4-7, tokens 0-511)
  Ulysses Group {2,3}: redistribute heads
    GPU 2: [B, 4, 512, D]  (heads 0-3, tokens 512-1023)
    GPU 3: [B, 4, 512, D]  (heads 4-7, tokens 512-1023)

Step 2 — Ring Pass (within Ring groups):
  Ring Group {0,2}: GPU 0 and GPU 2 exchange KV chunks
    GPU 0 computes attention over heads 0-3, tokens 0-1023 (via ring)
    GPU 2 computes attention over heads 0-3, tokens 0-1023 (via ring)
  Ring Group {1,3}: GPU 1 and GPU 3 exchange KV chunks
    GPU 1 computes attention over heads 4-7, tokens 0-1023 (via ring)
    GPU 3 computes attention over heads 4-7, tokens 0-1023 (via ring)

Step 3 — Reverse Ulysses All-to-All:
  Restore original sequence-sharded layout
  GPU 0: [B, 8, 256, D]  (all heads, tokens 0-255)
  GPU 1: [B, 8, 256, D]  (all heads, tokens 256-511)
  GPU 2: [B, 8, 256, D]  (all heads, tokens 512-767)
  GPU 3: [B, 8, 256, D]  (all heads, tokens 768-1023)
```

## DeviceMesh Configuration Examples

### 2 GPUs — Pure Ulysses

```python
# mesh = (Ring=1, Ulysses=2) — pure Ulysses, no ring communication
mesh = init_context_parallel_mesh("cuda", max_ring_dim_size=1)
# Result: DeviceMesh(shape=(1, 2), names=("ring", "ulysses"))
# GPU 0 and GPU 1 form one Ulysses group
# Requires: H % 2 == 0
```

### 4 GPUs — Hybrid (2 Ring × 2 Ulysses)

```python
# mesh = (Ring=2, Ulysses=2) — hybrid
mesh = init_context_parallel_mesh("cuda", max_ring_dim_size=2)
# Result: DeviceMesh(shape=(2, 2), names=("ring", "ulysses"))
# Ulysses groups: {GPU 0, GPU 1}, {GPU 2, GPU 3}
# Ring groups:    {GPU 0, GPU 2}, {GPU 1, GPU 3}
# Requires: H % 2 == 0
```

### 8 GPUs — Hybrid (2 Ring × 4 Ulysses)

```python
# mesh = (Ring=2, Ulysses=4) — hybrid, more Ulysses parallelism
mesh = init_context_parallel_mesh("cuda", max_ring_dim_size=2)
# Result: DeviceMesh(shape=(2, 4), names=("ring", "ulysses"))
# Ulysses groups: {0,1,2,3}, {4,5,6,7}
# Ring groups:    {0,4}, {1,5}, {2,6}, {3,7}
# Requires: H % 4 == 0

# Alternative: mesh = (Ring=4, Ulysses=2) — more Ring parallelism
mesh = init_context_parallel_mesh("cuda", max_ring_dim_size=4)
# Result: DeviceMesh(shape=(4, 2), names=("ring", "ulysses"))
# Ulysses groups: {0,1}, {2,3}, {4,5}, {6,7}
# Ring groups:    {0,2,4,6}, {1,3,5,7}
# Requires: H % 2 == 0
```

## The `max_ring_dim_size` Parameter

The `max_ring_dim_size` parameter in `init_context_parallel_mesh` controls how GPUs are split between the Ring and Ulysses dimensions:

```python
mesh = init_context_parallel_mesh(device_type, max_ring_dim_size=R)
```

- `max_ring_dim_size` sets the **maximum** number of GPUs in the Ring dimension.
- The Ulysses dimension gets `N / R` GPUs (where `N` is total GPU count).
- Setting `max_ring_dim_size=1` disables Ring entirely, giving pure Ulysses.
- Setting `max_ring_dim_size=N` disables Ulysses entirely, giving pure Ring.

**How to choose the value:**

| Scenario | Recommended `max_ring_dim_size` |
|----------|-------------------------------|
| H >= N and H % N == 0 | 1 (pure Ulysses — fewest communication rounds) |
| H < N | N / H (use Ulysses up to H, Ring for the rest) |
| Very long sequences, memory-bound | Higher values (more Ring = less memory per GPU) |
| Latency-sensitive | Lower values (more Ulysses = fewer rounds) |

**Example**: With 8 GPUs and H=4 heads:
- `max_ring_dim_size=1` would need Ulysses dim = 8, but H=4 < 8, so this fails.
- `max_ring_dim_size=2` gives mesh=(2, 4), requiring H % 4 == 0. Works since 4 % 4 == 0.
- `max_ring_dim_size=4` gives mesh=(4, 2), requiring H % 2 == 0. Also works.
- `max_ring_dim_size=8` gives mesh=(8, 1), pure Ring. Always works but most communication rounds.

## Mathematical Equivalence Guarantee

USP produces results **identical** to standard single-GPU attention. The guarantee follows from the composition of two individually exact methods:

1. **Ulysses is exact**: All-to-All is lossless data redistribution; head-parallel attention is mathematically identical to full attention.
2. **Ring is exact**: Online softmax with LSE tracking produces bit-identical results to full-sequence softmax.
3. **Composition preserves exactness**: Applying Ulysses first (redistributing heads) and then Ring (distributing sequence chunks) is equivalent to a single attention computation over all heads and all tokens. The two transformations operate on orthogonal dimensions (heads vs. sequence positions) and do not interfere.

No approximation is introduced at any stage. The only source of numerical difference is floating-point non-determinism from reordering additions, which is inherent to any parallel computation.

## Comparison: Ulysses vs Ring vs Unified

| Aspect | Ulysses | Ring | Unified (USP) |
|--------|---------|------|----------------|
| Parallelism type | Head | Sequence | Head + Sequence |
| GPU arrangement | Flat | Ring | 2D mesh (Ring × Ulysses) |
| Communication pattern | All-to-All | Point-to-point ring | All-to-All within Ulysses groups + ring within Ring groups |
| Communication rounds | 2 | N-1 | 2 + (R-1) where R = Ring dim size |
| Head count constraint | H % N == 0 | None | H % U == 0 (U = Ulysses dim) |
| Max GPU scaling | Limited by H | Unlimited | Unlimited (Ring dim absorbs excess) |
| Sequence memory per GPU | Full S on each GPU | S/N per GPU | S/R per GPU (within Ulysses group) |
| Overlap potential | Low | High | High (Ring portion overlaps) |
| Implementation complexity | Simple | Moderate | Higher (state machine + 2D mesh) |
| Best for | Few GPUs, many heads | Many GPUs, long sequences | Large-scale with mixed constraints |

### When to Use Each

- **Ulysses alone**: When `H >= N` and you want minimal communication rounds. Ideal for 2-8 GPUs with models that have many attention heads.
- **Ring alone**: When the head count is small or not divisible by GPU count, or when sequences are extremely long and memory is the bottleneck.
- **USP (Unified)**: When scaling beyond what Ulysses can handle alone (more GPUs than heads), or when you want to balance communication efficiency (Ulysses) with memory efficiency (Ring). USP is the general-purpose solution that subsumes both special cases.
