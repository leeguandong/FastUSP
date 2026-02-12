<h3 align="center">
    FastUSP: A Multi-Level Collaborative Acceleration Framework for Distributed Diffusion Model Inference
</h3>

<p align="center">
<a href="https://arxiv.org/abs/2602.10940"><img alt="Build" src="https://img.shields.io/badge/Tech%20Report-FastUSP-b31b1b.svg"></a>
<a href="https://github.com/leeguandong/FastUSP"><img src="https://img.shields.io/static/v1?label=GitHub&message=repository&color=green"></a>
</p>

<p align="center">
<span style="color:#137cf3; font-family: Gill Sans">Guandong Li</span><sup></sup></a> <br>
</p>

## Abstract

Large-scale diffusion models such as FLUX (12B parameters) and Stable Diffusion 3 (8B parameters) require multi-GPU parallelism for efficient inference. Unified Sequence Parallelism (USP), which combines Ulysses and Ring attention mechanisms, has emerged as the state-of-the-art approach for distributed attention computation. However, existing USP implementations suffer from significant inefficiencies including excessive kernel launch overhead and suboptimal computation-communication scheduling. In this paper, we propose **FastUSP**, a multi-level optimization framework that integrates compile-level optimization (graph compilation with CUDA Graphs and computation-communication reordering), communication-level optimization (FP8 quantized collective communication), and operator-level optimization (pipelined Ring attention with double buffering). We evaluate FastUSP on FLUX (12B) and Qwen-Image models across 2, 4, and 8 NVIDIA RTX 5090 GPUs. On FLUX, FastUSP achieves consistent **1.12×–1.16×** end-to-end speedup over baseline USP, with compile-level optimization contributing the dominant improvement. On Qwen-Image, FastUSP achieves **1.09×** speedup on 2 GPUs; on 4–8 GPUs, we identify a PyTorch Inductor compatibility limitation with Ring attention that prevents compile optimization, while baseline USP scales to 1.30×–1.46× of 2-GPU performance. We further provide a detailed analysis of the performance characteristics of distributed diffusion inference, revealing that kernel launch overhead---rather than communication latency---is the primary bottleneck on modern high-bandwidth GPU interconnects.

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
@misc{li2026fastuspmultilevelcollaborativeacceleration,
      title={FastUSP: A Multi-Level Collaborative Acceleration Framework for Distributed Diffusion Model Inference}, 
      author={Guandong Li},
      year={2026},
      eprint={2602.10940},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.10940}, 
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
