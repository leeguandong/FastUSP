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
Qwen-Image editing with distributed parallel attention (USP).

Usage:
    torchrun --nproc_per_node=2 qwen_image_edit.py --input_image input.png --prompt "Make it look like a painting"
"""

import argparse
import time

import torch
import torch.distributed as dist
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import (
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from optimum.quanto import freeze, qfloat8_e4m3fn, quantize

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image edit parallel inference")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit")
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to the input image to edit")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Editing instruction")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_ring_dim_size", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile optimization")
    parser.add_argument("--output", type=str, default="qwen_edit_output.png")
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
    pipe = QwenImageEditPlusPipeline.from_pretrained(
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
    parallelize_vae(pipe.vae, mesh=mesh._flatten())

    # Optional torch.compile optimization
    if args.compile:
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    # Load input image
    input_image = Image.open(args.input_image).resize((args.width, args.height)).convert("RGB")

    # Generate edited image
    start = time.time()
    image = pipe(
        image=input_image,
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
