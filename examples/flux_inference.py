# Copyright 2024 iFLYTEK
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FLUX distributed inference with parallel attention (USP).

Usage:
    torchrun --nproc_per_node=2 flux_inference.py --prompt "A cat holding a sign that says hello world"
    torchrun --nproc_per_node=4 flux_inference.py --compile --steps 28
"""

import argparse
import time

import torch
import torch.distributed as dist
from diffusers import AutoModel, FluxPipeline
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8_e4m3fn, quantize

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX parallel inference")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_ring_dim_size", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile optimization")
    parser.add_argument("--output", type=str, default="flux_output.png")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed
    dist.init_process_group()
    torch.cuda.set_device(dist.get_rank())

    # Load and quantize model components
    transformer = AutoModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )

    quantize(transformer, weights=qfloat8_e4m3fn)
    freeze(transformer)
    quantize(text_encoder_2, weights=qfloat8_e4m3fn)
    freeze(text_encoder_2)

    # Create pipeline
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Parallelize with context parallel mesh
    mesh = init_context_parallel_mesh(
        pipe.device.type,
        max_ring_dim_size=args.max_ring_dim_size,
    )
    parallelize_pipe(pipe, mesh=mesh)
    parallelize_vae(pipe.vae, mesh=mesh._flatten())

    # Optional torch.compile optimization
    if args.compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")

    # Generate image
    start = time.time()
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).images[0]
    elapsed = time.time() - start

    if dist.get_rank() == 0:
        print(f"Inference time: {elapsed:.2f}s")
        image.save(args.output)
        print(f"Saved to {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
