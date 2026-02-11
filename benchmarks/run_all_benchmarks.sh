#!/bin/bash
# Copyright 2024 iFLYTEK. Licensed under Apache 2.0.
#
# FastUSP Benchmark Suite
# Usage: bash run_all_benchmarks.sh [NUM_GPUS]

set -e

NUM_GPUS=${1:-2}
STEPS=30

echo "=== FastUSP Benchmark Suite ==="
echo "GPUs: $NUM_GPUS, Steps: $STEPS"
echo ""

echo "--- Baseline USP ---"
torchrun --nproc_per_node=$NUM_GPUS flux_benchmark.py --mode baseline --steps $STEPS

echo ""
echo "--- FastUSP (Compile Optimized) ---"
torchrun --nproc_per_node=$NUM_GPUS flux_benchmark.py --mode fastusp --steps $STEPS

echo ""
echo "--- Ablation: Compile Only ---"
torchrun --nproc_per_node=$NUM_GPUS flux_ablation.py --compile true --steps $STEPS

echo ""
echo "--- Ablation: All Optimizations ---"
torchrun --nproc_per_node=$NUM_GPUS flux_ablation.py --compile true --quantize true --pipeline true --steps $STEPS

echo ""
echo "--- Communication Profile ---"
torchrun --nproc_per_node=$NUM_GPUS profile_communication.py

echo ""
echo "=== All benchmarks complete ==="
