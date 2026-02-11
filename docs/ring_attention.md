# Ring Attention: Sequence-Parallel Distributed Attention

## Problem Statement

As sequence lengths grow (4K, 16K, 64K, or even longer), the self-attention mechanism becomes a critical bottleneck. Attention has **O(S^2)** memory and compute complexity — doubling the sequence length quadruples the cost. At some point, a single GPU simply cannot hold the full attention matrix in memory.

Ring Attention solves this by **partitioning the sequence across GPUs** and computing attention incrementally, passing Key-Value (KV) chunks around a ring topology. Each GPU only ever holds a fraction of the full sequence, keeping memory usage bounded.

## Core Idea

Given `N` GPUs and a sequence of length `S`:

1. **Partition** the sequence into `N` chunks of size `S/N`. GPU `i` holds tokens `[i×S/N .. (i+1)×S/N - 1]`.
2. Each GPU has its own local Q, K, V chunks.
3. GPUs are arranged in a **logical ring**. In each step, KV chunks are passed to the next GPU in the ring.
4. Each GPU computes a **partial attention** between its local Q and the currently held KV chunk, then **merges** the result with previous partial results using the **online softmax** (log-sum-exp) trick.
5. After `N-1` passes, every GPU has seen all KV chunks and holds the correct final attention output for its local Q tokens.

## Ring Topology

```
         ┌───────┐
    ┌───►│ GPU 0 │────┐
    │    │ Q0,K0,V0   │
    │    └───────┘    │
    │                 │  KV pass
    │                 ▼
┌───────┐        ┌───────┐
│ GPU 3 │        │ GPU 1 │
│ Q3,K3,V3       │ Q1,K1,V1
└───────┘        └───────┘
    ▲                 │
    │                 │  KV pass
    │    ┌───────┐    │
    └────│ GPU 2 │◄───┘
         │ Q2,K2,V2
         └───────┘

Each step: GPU i sends its current KV to GPU (i+1) % N
           GPU i receives KV from GPU (i-1) % N
```

## Online Softmax: The LSE Trick

The key challenge in Ring Attention is computing softmax **incrementally**. Standard softmax requires the full set of attention scores, but in Ring Attention, scores arrive in chunks. The **online softmax** algorithm using log-sum-exp (LSE) values solves this.

### The Problem

Standard attention for query `q` over keys `K`:

o = softmax(Q × K^T / √D) × V
  = Σ_j [exp(s_j - m) / Σ_k exp(s_k - m)] × v_j

where s_j = q × k_j^T / √D, and m = max(s_j) for numerical stability.
```

When keys arrive in chunks, we cannot compute the global max `m` or the global sum upfront. The online softmax algorithm maintains a **running log-sum-exp (LSE)** value that allows incremental updates:

```
LSE = log(Σ exp(s_j - m)) + m
```

When a new chunk of scores arrives with its own local max `m_new` and local LSE `lse_new`, we merge:

```
m_combined   = max(m_old, m_new)
lse_combined = log(exp(lse_old - m_combined) + exp(lse_new - m_combined)) + m_combined
```

The output accumulator is rescaled accordingly:

```
scale_old = exp(lse_old - lse_combined)
scale_new = exp(lse_new - lse_combined)
O_combined = scale_old × O_old + scale_new × O_new
```

This merge operation is numerically stable and exact.

### Step-by-Step Algorithm

```
Input:  Q_i, K_i, V_i on GPU i (each of shape [B, H, S/N, D])
        N GPUs arranged in a ring: 0 → 1 → 2 → ... → N-1 → 0

Initialize:
    KV_buffer = (K_i, V_i)          # Start with local KV
    O_i = zeros([B, H, S/N, D])     # Output accumulator
    LSE_i = -inf([B, H, S/N])       # Log-sum-exp accumulator

For step = 0 to N-1:
    # 1. Compute local attention with current KV chunk
    O_chunk, LSE_chunk = attention_with_lse(Q_i, KV_buffer)

    # 2. Merge into running accumulator using online softmax
    m_new       = max(LSE_i, LSE_chunk)
    scale_old   = exp(LSE_i - m_new)
    scale_new   = exp(LSE_chunk - m_new)
    O_i         = scale_old × O_i + scale_new × O_chunk
    LSE_i       = m_new + log(exp(LSE_i - m_new) + exp(LSE_chunk - m_new))

    # 3. Ring pass: send current KV to next GPU, receive from previous
    if step < N-1:
        KV_buffer = ring_send_recv(KV_buffer)
        # Async: overlap this transfer with step's computation

    # After N steps, O_i contains the correct attention output for Q_i

Output: O_i — shape [B, H, S/N, D], exact attention for local query tokens
```

## Mathematical Correctness

Ring Attention produces **exact** results (identical to standard attention). The proof relies on two properties:

### 1. Decomposability of Softmax

The softmax function can be decomposed over partitions of the key set. For keys split into chunks `K_1, K_2, ..., K_N`:

```
softmax([s_1; s_2; ...; s_N]) = [α_1 × softmax(s_1); α_2 × softmax(s_2); ...; α_N × softmax(s_N)]
```

where `α_j = exp(m_j - m_global) / Σ_k exp(m_k - m_global)` are rescaling factors derived from per-chunk maxima.

### 2. Associativity of the Merge Operation

The merge operation (rescale + weighted sum using LSE) is **associative**. This means the order in which KV chunks arrive does not affect the final result. Whether we process chunks in order `[0,1,2,3]` or `[2,0,3,1]`, the output is identical (up to floating-point precision).

Together, these properties guarantee that Ring Attention is mathematically equivalent to computing attention over the full sequence on a single device.

## Communication-Computation Overlap

Ring Attention naturally supports **pipelining** of communication and computation:

```
Timeline for GPU i:

Step 0:  [Compute attn(Q_i, KV_i)]  [Send KV_i → next, Recv KV_(i-1) ← prev]
Step 1:  [Compute attn(Q_i, KV_(i-1))]  [Send KV_(i-1) → next, Recv KV_(i-2) ← prev]
Step 2:  [Compute attn(Q_i, KV_(i-2))]  [Send KV_(i-2) → next, Recv KV_(i-3) ← prev]
  ...
```

While a GPU is computing attention with the current KV chunk, it simultaneously sends that chunk to the next GPU and receives the next chunk from the previous GPU. This overlap means that **communication can be almost entirely hidden** behind computation, provided the compute time per chunk exceeds the transfer time.

In practice, this overlap is highly effective for long sequences where the per-chunk attention computation is substantial.

## Complexity Analysis

| Metric | Formula | Notes |
|--------|---------|-------|
| Compute per GPU | O(B × H × (S/N) × S × D) = O(B×H×S²×D/N) | Each GPU computes Q_local against all KV chunks |
| Communication volume (total) | O(B × H × S × D) | Each KV chunk of size B×H×(S/N)×D sent N-1 times |
| Communication per step | O(B × H × (S/N) × D) | One KV chunk transferred per step |
| Communication rounds | N - 1 | One round per ring pass |
| Memory per GPU | O(B × H × (S/N)² + B × H × (S/N) × D) | Local attention block + activations |

Key observations:
- **Compute** scales as O(1/N) per GPU — linear speedup.
- **Communication** is O(S×D) total, independent of N. Each step transfers O(S/N) data, but there are N-1 steps.
- **Memory** scales as O(1/N²) for the attention matrix — quadratic memory savings.
- **Rounds** scale as O(N) — this is the main disadvantage compared to Ulysses Attention's fixed 2 rounds.

## Relationship to FlashAttention

Ring Attention can be viewed as a **distributed extension of FlashAttention**:

| Aspect | FlashAttention | Ring Attention |
|--------|---------------|----------------|
| Scope | Single GPU | Multi-GPU |
| Tiling | Tiles Q and KV into blocks within GPU memory | Tiles KV across GPUs |
| Online softmax | Yes (across tiles) | Yes (across GPU chunks) |
| Memory optimization | Avoids materializing full S×S matrix | Avoids storing full sequence on one GPU |
| IO awareness | Optimizes HBM ↔ SRAM transfers | Optimizes GPU ↔ GPU transfers |

In practice, Ring Attention uses FlashAttention as its **local attention kernel** within each GPU. The ring pass handles the distributed KV distribution, while FlashAttention handles the efficient on-device computation. They are complementary techniques operating at different levels of the memory hierarchy:

```
Ring Attention (inter-GPU: distributes KV chunks across GPUs)
    └── FlashAttention (intra-GPU: tiles attention within HBM/SRAM)
        └── Hardware attention kernel (on-chip SRAM computation)
```

This layered approach means Ring Attention inherits all of FlashAttention's on-device optimizations (no S×S materialization, optimal SRAM usage) while adding distributed sequence parallelism on top.
