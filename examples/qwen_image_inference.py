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
Qwen-Image distributed inference with parallel attention (USP).

Usage:
    torchrun --nproc_per_node=2 qwen_image_inference.py --prompt "A beautiful landscape"
    torchrun --nproc_per_node=4 qwen_image_inference.py --compile --steps 50
"""

import argparse
import time

import torch
import torch.distributed as dist
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from optimum.quanto import freeze, qfloat8_e4m3fn, quantize

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image parallel inference")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image")
    parser.add_argument("--prompt", type=str, default=(
        'A coffee shop entrance features a chalkboard sign reading '
        '"Qwen Coffee $2 per cup," with a neon light beside it. '
        'Ultra HD, 4K, cinematic composition'
    ))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_ring_dim_size", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile optimization")
    parser.add_argument("--output", type=str, default="qwen_output.png")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed
    dist.init_process_group()
    torch.cuda.set_device(dist.get_rank())

    dtype = torch.bfloat16

    # Load and quantize model components
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=dtype
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=dtype
    )

    quantize(text_encoder, weights=qfloat8_e4m3fn)
    freeze(text_encoder)
    quantize(transformer, weights=qfloat8_e4m3fn)
    freeze(transformer)

    # Create pipeline
    pipe = QwenImagePipeline.from_pretrained(
        args.model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=dtype,
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # Parallelize with context parallel mesh
    mesh = init_context_parallel_mesh(
        pipe.device.type,
        max_ring_dim_size=args.max_ring_dim_size,
    )
    parallelize_pipe(pipe, mesh=mesh)

    # Optional torch.compile optimization
    if args.compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    # Generate image
    start = time.time()
    image = pipe(
        prompt=args.prompt,
        negative_prompt=" ",
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        output_type="pil" if dist.get_rank() == 0 else "pt",
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    elapsed = time.time() - start

    if dist.get_rank() == 0:
        print(f"Inference time: {elapsed:.2f}s")
        image.save(args.output)
        print(f"Saved to {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
