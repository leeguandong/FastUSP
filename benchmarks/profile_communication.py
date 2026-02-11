# Copyright 2024 iFLYTEK. Licensed under Apache 2.0.

"""FastUSP Communication Profiler - Measures communication vs computation breakdown.

Profiles the time spent in all-to-all collectives and attention computation
to quantify the communication overhead that FastUSP optimizations target.

Default tensor shapes match FLUX: B=1, H=24, S=4096, D=64.

Usage:
    torchrun --nproc_per_node=2 profile_communication.py
    torchrun --nproc_per_node=2 profile_communication.py --seq_len 16384
"""

import time
import argparse

import torch
import torch.distributed as dist

# Parse arguments
parser = argparse.ArgumentParser(description="FastUSP communication profiler")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--num_heads", type=int, default=24, help="Number of attention heads")
parser.add_argument("--seq_len", type=int, default=4096, help="Total sequence length")
parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
parser.add_argument("--iterations", type=int, default=10, help="Iterations per test")
args, _ = parser.parse_known_args()

B = args.batch_size
H = args.num_heads
S = args.seq_len
D = args.head_dim

# Initialize distributed
dist.init_process_group()
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)


def log(msg):
    if rank == 0:
        print(msg, flush=True)


log(f"\n{'='*60}")
log(f"Communication vs Computation Profiling")
log(f"World Size: {world_size}")
log(f"{'='*60}\n")

import para_attn.primitives as DP

# Per-GPU sequence length
S_local = S // world_size

log(f"Tensor shapes:")
log(f"  Query/Key/Value: [{B}, {H}, {S_local}, {D}]")
log(f"  Total sequence: {S}")
log(f"  Local sequence: {S_local}")

# Create test tensors
query = torch.randn(B, H, S_local, D, dtype=torch.bfloat16, device="cuda")
key = torch.randn(B, H, S_local, D, dtype=torch.bfloat16, device="cuda")
value = torch.randn(B, H, S_local, D, dtype=torch.bfloat16, device="cuda")

# Warmup
default_group = dist.group.WORLD
for _ in range(3):
    _ = DP.all_to_all_single_sync(
        query.permute(1, 0, 2, 3).contiguous(),
        output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    torch.cuda.synchronize()

dist.barrier()

# ============== Test 1: Single all-to-all time ==============
log("\n--- Test 1: Single all-to-all time ---")

times = []
for i in range(args.iterations):
    x = query.permute(1, 0, 2, 3).contiguous()
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    x = DP.all_to_all_single_sync(
        x, output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    torch.cuda.synchronize()
    end = time.perf_counter()

    times.append((end - start) * 1000)

avg_single = sum(times) / len(times)
log(f"  Single all-to-all: {avg_single:.3f} ms")

# ============== Test 2: Three sequential all-to-all ==============
log("\n--- Test 2: Three sequential all-to-all ---")

times = []
for i in range(args.iterations):
    q = query.permute(1, 0, 2, 3).contiguous()
    k = key.permute(1, 0, 2, 3).contiguous()
    v = value.permute(1, 0, 2, 3).contiguous()
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    q = DP.all_to_all_single_sync(
        q, output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    k = DP.all_to_all_single_sync(
        k, output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    v = DP.all_to_all_single_sync(
        v, output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    torch.cuda.synchronize()
    end = time.perf_counter()

    times.append((end - start) * 1000)

avg_sequential = sum(times) / len(times)
log(f"  Three sequential all-to-all: {avg_sequential:.3f} ms")
log(f"  Expected (3x single): {avg_single * 3:.3f} ms")

# ============== Test 3: Three async all-to-all ==============
log("\n--- Test 3: Three async all-to-all (current implementation) ---")

times = []
for i in range(args.iterations):
    q = query.permute(1, 0, 2, 3).contiguous()
    k = key.permute(1, 0, 2, 3).contiguous()
    v = value.permute(1, 0, 2, 3).contiguous()
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    # Launch async operations
    q_handle = DP.all_to_all_single_async(
        q, output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    k_handle = DP.all_to_all_single_async(
        k, output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    v_handle = DP.all_to_all_single_async(
        v, output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    # Wait for completion
    q = q_handle.wait()
    k = k_handle.wait()
    v = v_handle.wait()
    torch.cuda.synchronize()
    end = time.perf_counter()

    times.append((end - start) * 1000)

avg_async = sum(times) / len(times)
log(f"  Three async all-to-all: {avg_async:.3f} ms")
log(f"  Speedup vs sequential: {avg_sequential / avg_async:.2f}x")

# ============== Test 4: Attention computation time ==============
log("\n--- Test 4: Attention computation time ---")

import para_attn.ops as para_attn_ops

# Post all-to-all tensor shapes: [B, H/N, S, D]
q_after = torch.randn(B, H // world_size, S, D, dtype=torch.bfloat16, device="cuda")
k_after = torch.randn(B, H // world_size, S, D, dtype=torch.bfloat16, device="cuda")
v_after = torch.randn(B, H // world_size, S, D, dtype=torch.bfloat16, device="cuda")

# Warmup
for _ in range(3):
    _ = para_attn_ops.sage_attention_forward_with_lse(q_after, k_after, v_after)
    torch.cuda.synchronize()

times = []
for i in range(args.iterations):
    torch.cuda.synchronize()

    start = time.perf_counter()
    out = para_attn_ops.sage_attention_forward_with_lse(q_after, k_after, v_after)
    torch.cuda.synchronize()
    end = time.perf_counter()

    times.append((end - start) * 1000)

avg_compute = sum(times) / len(times)
log(f"  Attention computation: {avg_compute:.3f} ms")

# ============== Test 5: Output all-to-all time ==============
log("\n--- Test 5: Output all-to-all time ---")

# Output shape: [B, H/N, S, D]
out = torch.randn(B, H // world_size, S, D, dtype=torch.bfloat16, device="cuda")

times = []
for i in range(args.iterations):
    x = out.permute(2, 0, 1, 3).contiguous()
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    x = DP.all_to_all_single_sync(
        x, output_split_sizes=None, input_split_sizes=None, group=default_group,
    )
    torch.cuda.synchronize()
    end = time.perf_counter()

    times.append((end - start) * 1000)

avg_output = sum(times) / len(times)
log(f"  Output all-to-all: {avg_output:.3f} ms")

# ============== Summary ==============
log(f"\n{'='*60}")
log(f"SUMMARY")
log(f"{'='*60}")

total_comm = avg_sequential + avg_output
total_compute = avg_compute
total = total_comm + total_compute

log(f"\nPer-layer breakdown:")
log(f"  Input all-to-all (Q+K+V): {avg_sequential:.3f} ms ({avg_sequential/total*100:.1f}%)")
log(f"  Attention computation:    {avg_compute:.3f} ms ({avg_compute/total*100:.1f}%)")
log(f"  Output all-to-all:        {avg_output:.3f} ms ({avg_output/total*100:.1f}%)")
log(f"  Total per layer:          {total:.3f} ms")

log(f"\nCommunication vs Computation:")
log(f"  Communication: {total_comm:.3f} ms ({total_comm/total*100:.1f}%)")
log(f"  Computation:   {total_compute:.3f} ms ({total_compute/total*100:.1f}%)")

log(f"\nAsync optimization potential:")
log(f"  Current async speedup: {avg_sequential / avg_async:.2f}x")
if avg_async < avg_sequential:
    saved = avg_sequential - avg_async
    log(f"  Time saved per layer: {saved:.3f} ms")
    log(f"  Estimated total speedup: {total / (total - saved):.2f}x")
else:
    log(f"  Async is not faster - need different approach")

# FLUX has 19 double blocks + 38 single blocks = 57 attention layers
num_layers = 57
log(f"\nFLUX model (57 attention layers):")
log(f"  Total communication time: {total_comm * num_layers / 1000:.3f} s")
log(f"  Total computation time:   {total_compute * num_layers / 1000:.3f} s")
log(f"  Estimated total time:     {total * num_layers / 1000:.3f} s")

log(f"\n{'='*60}\n")

dist.destroy_process_group()
