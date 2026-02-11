# Copyright 2024 iFLYTEK. Licensed under Apache 2.0.

"""FastUSP Ablation Study - Tests individual optimization contributions.

This script measures the performance impact of each FastUSP optimization level:
  - Compile-level: torch.compile with CUDA Graphs + reorder_for_compute_comm_overlap
  - Communication-level: FP8 quantized collective communication
  - Operator-level: Pipelined Ring attention

Usage:
    # Baseline (no optimizations)
    torchrun --nproc_per_node=2 flux_ablation.py

    # Compile-level only
    torchrun --nproc_per_node=2 flux_ablation.py --compile true

    # All optimizations
    torchrun --nproc_per_node=2 flux_ablation.py --compile true --quantize true --pipeline true
"""

import os
import time
import argparse

import torch
import torch.distributed as dist

# Parse arguments
parser = argparse.ArgumentParser(description="FastUSP ablation study benchmark")
parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev",
                    help="HuggingFace model ID or local path")
parser.add_argument("--compile", type=str, default="false", choices=["true", "false"],
                    help="Enable compile-level optimization (CUDA Graphs + reorder)")
parser.add_argument("--quantize", type=str, default="false", choices=["true", "false"],
                    help="Enable communication-level optimization (FP8 quantization)")
parser.add_argument("--pipeline", type=str, default="false", choices=["true", "false"],
                    help="Enable operator-level optimization (Pipelined Ring)")
parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
parser.add_argument("--height", type=int, default=1024, help="Image height")
parser.add_argument("--width", type=int, default=1024, help="Image width")
args, _ = parser.parse_known_args()

# Convert string flags to booleans
enable_compile = (args.compile == "true")
enable_quantize = (args.quantize == "true")
enable_pipeline = (args.pipeline == "true")

# Initialize distributed
dist.init_process_group()
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)


def log(msg):
    if rank == 0:
        print(msg, flush=True)


log(f"\n{'='*70}")
log(f"FastUSP Ablation Study")
log(f"{'='*70}")
log(f"Configuration:")
log(f"  Compile-Level Optimization:      {enable_compile}")
log(f"  Communication-Level Optimization: {enable_quantize}")
log(f"  Operator-Level Optimization:      {enable_pipeline}")
log(f"  World Size: {world_size}, Steps: {args.steps}, Resolution: {args.height}x{args.width}")
log(f"{'='*70}\n")

# Set compile optimization flags
if enable_compile:
    log("Enabling compile-level optimization...")
    torch._inductor.config.reorder_for_compute_comm_overlap = True

# Load model
from diffusers import FluxPipeline, AutoModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8_e4m3fn, quantize

log("Loading model...")
transformer = AutoModel.from_pretrained(
    args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    args.model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
)

# Weight quantization (separate from communication quantization)
quantize(transformer, weights=qfloat8_e4m3fn)
freeze(transformer)
quantize(text_encoder_2, weights=qfloat8_e4m3fn)
freeze(text_encoder_2)

pipe = FluxPipeline.from_pretrained(
    args.model_id,
    transformer=transformer,
    text_encoder_2=text_encoder_2,
    torch_dtype=torch.bfloat16,
).to("cuda")

# Apply parallelization
log("Applying parallelization...")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
    max_ring_dim_size=2,
)

# Set environment flags for communication and operator optimizations
if enable_quantize:
    log("Enabling FP8 quantized communication...")
    os.environ["PARA_ATTN_QUANTIZE_COMM"] = "1"

if enable_pipeline:
    log("Enabling pipelined Ring attention...")
    os.environ["PARA_ATTN_PIPELINE_RING"] = "1"

parallelize_pipe(pipe, mesh=mesh)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

log(f"Mesh initialized: {mesh}")

# Compile transformer if enabled
if enable_compile:
    log("Compiling transformer with torch.compile...")
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="reduce-overhead",
        fullgraph=True,
    )

# Warmup
log(f"\nWarming up ({args.warmup} runs)...")
for i in range(args.warmup):
    _ = pipe(
        "warmup",
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        output_type="pt",
    )
    torch.cuda.synchronize()

dist.barrier()

# Benchmark
log(f"\nRunning benchmark ({args.runs} runs)...")
times = []

for i in range(args.runs):
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()

    image = pipe(
        "A cat holding a sign that says hello world",
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        output_type="pil" if rank == 0 else "pt",
    ).images[0]

    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    times.append(elapsed)
    log(f"  Run {i+1}: {elapsed:.3f}s ({args.steps/elapsed:.2f} steps/s)")

# Statistics
avg_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)
std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

log(f"\n{'='*70}")
log(f"Results:")
log(f"  Average:  {avg_time:.3f}s")
log(f"  Min:      {min_time:.3f}s")
log(f"  Max:      {max_time:.3f}s")
log(f"  Std Dev:  {std_dev:.3f}s")
log(f"  Steps/s:  {args.steps / avg_time:.2f}")
log(f"  Per-step: {avg_time / args.steps * 1000:.1f}ms")
log(f"{'='*70}\n")

# Save image
if rank == 0:
    config_name = f"compile{int(enable_compile)}_quant{int(enable_quantize)}_pipe{int(enable_pipeline)}"
    image.save(f"ablation_{config_name}.png")
    log(f"Image saved to ablation_{config_name}.png")

dist.destroy_process_group()
