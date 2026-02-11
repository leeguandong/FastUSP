# FastUSP

A multi-level optimization framework for accelerating distributed diffusion model inference using Unified Sequence Parallelism (USP).

FastUSP combines three optimization levels to achieve significant speedups on multi-GPU setups:

```
┌─────────────────────────────────────────────────┐
│              FastUSP Framework                   │
├─────────────────────────────────────────────────┤
│  Level 1: Compile Optimization                  │
│    torch.compile + CUDA Graphs                  │
├─────────────────────────────────────────────────┤
│  Level 2: Communication Optimization            │
│    FP8 Quantized All-to-All + Overlap           │
├─────────────────────────────────────────────────┤
│  Level 3: Operator Optimization                 │
│    USP (Ulysses + Ring Attention) 2D Mesh       │
└─────────────────────────────────────────────────┘
```

## Performance

| Model | GPUs | Baseline (s) | FastUSP (s) | Speedup |
|-------|------|-------------|-------------|---------|
| FLUX.1-dev | 2 | 12.45 | 10.73 | 1.16× |
| FLUX.1-dev | 4 | 8.21 | 7.33 | 1.12× |
| Qwen2.5-VL | 2 | 15.32 | 14.06 | 1.09× |

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python >= 3.10
- PyTorch >= 2.8.0 with CUDA support
- NVIDIA GPUs with NVLink (recommended)

## Quick Start

```bash
# FLUX inference on 2 GPUs
torchrun --nproc_per_node=2 examples/flux_inference.py \
    --prompt "A cat holding a sign that says hello world"

# FLUX inference on 4 GPUs with torch.compile
torchrun --nproc_per_node=4 examples/flux_inference.py \
    --compile --steps 28

# Qwen2.5-VL image generation
torchrun --nproc_per_node=2 examples/qwen_image_inference.py \
    --prompt "A futuristic city at sunset"

# Qwen2.5-VL image editing
torchrun --nproc_per_node=2 examples/qwen_image_edit.py \
    --source_image input.png --prompt "Make it snowy"
```

## Optimization Levels

### Level 1: Compile Optimization
Uses `torch.compile` with `reduce-overhead` mode and CUDA Graphs to minimize kernel launch overhead and enable operator fusion.

```python
torch._inductor.config.reorder_for_compute_comm_overlap = True
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
```

### Level 2: Communication Optimization
FP8 quantized All-to-All collective communication reduces inter-GPU bandwidth by ~50% with negligible quality loss. Communication is overlapped with computation.

### Level 3: Operator Optimization (USP)
Unified Sequence Parallelism combines Ulysses Attention (head-parallel) and Ring Attention (sequence-parallel) in a 2D mesh:

```python
from para_attn.context_parallel import init_context_parallel_mesh
mesh = init_context_parallel_mesh("cuda", max_ring_dim_size=2)
# For 4 GPUs: creates a (2, 2) mesh — Ring dim × Ulysses dim
```

See [docs/](docs/) for detailed explanations of each attention strategy.

## Benchmarks

```bash
# Run all benchmarks
bash benchmarks/run_all_benchmarks.sh

# Ablation study
torchrun --nproc_per_node=2 benchmarks/flux_ablation.py --compile --quantize

# Baseline vs FastUSP comparison
torchrun --nproc_per_node=2 benchmarks/flux_benchmark.py --mode fastusp

# Communication profiling
torchrun --nproc_per_node=2 benchmarks/profile_communication.py
```

## Project Structure

```
FastUSP/
├── examples/              # Inference scripts
│   ├── flux_inference.py
│   ├── qwen_image_inference.py
│   └── qwen_image_edit.py
├── benchmarks/            # Performance benchmarks
│   ├── flux_ablation.py
│   ├── flux_benchmark.py
│   ├── profile_communication.py
│   └── run_all_benchmarks.sh
├── docs/                  # Technical documentation
│   ├── ulysses_attention.md
│   ├── ring_attention.md
│   └── unified_attention.md
├── requirements.txt
└── LICENSE
```

## Citation

```bibtex
@article{li2025fastusp,
  title={FastUSP: A Multi-Level Optimization Framework for Accelerating Distributed Diffusion Model Inference},
  author={Li, Guandong and Chu, Zhaobin},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
