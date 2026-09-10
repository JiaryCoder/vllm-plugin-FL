# Copyright (c) 2026 BAAI. All rights reserved.
"""Create deterministic tiny checkpoints for stage-two integration checks."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    LlamaConfig, LlamaForCausalLM, Qwen2MoeConfig, Qwen2MoeForCausalLM,
    Qwen3NextConfig, Qwen3NextForCausalLM,
)


def quantize(source, destination, ignore_router=False):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    data = load_file(str(source / "model.safetensors"))
    converted = {}
    for name, value in data.items():
        skipped = ignore_router and name.endswith((".gate.weight", ".shared_expert_gate.weight"))
        if name.endswith(".weight") and value.ndim == 2 and "model.layers." in name and not skipped:
            scale = value.float().abs().amax(-1, keepdim=True) / 127
            safe = torch.where(scale == 0, torch.ones_like(scale), scale)
            converted[name] = (value.float() / safe).round().clamp(-127, 127).to(torch.int8)
            converted[name.removesuffix(".weight") + ".weight_scale"] = scale
        else:
            converted[name] = value
    save_file(converted, str(destination / "model.safetensors"), metadata={"format": "pt"})
    config = json.loads((source / "config.json").read_text())
    ignore = ["lm_head"]
    if ignore_router:
        ignore += [r"re:.*\.gate$", r"re:.*\.shared_expert_gate$"]
    config["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "int-quantized",
        "quantization_status": "compressed", "ignore": ignore,
        "config_groups": {"group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "int", "symmetric": True,
                        "strategy": "channel", "dynamic": False},
            "input_activations": {"num_bits": 8, "type": "int", "symmetric": True,
                                  "strategy": "token", "dynamic": True},
        }},
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention-head-size", type=int, default=16,
                        help="Use 64 for vLLM Triton Attention integration checks")
    args = parser.parse_args()
    if args.attention_head_size < 8 or args.attention_head_size % 8:
        parser.error("attention-head-size must be a positive multiple of 8")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    names = ["tiny-llama", "tiny-qwen2-moe", "tiny-qwen3-next",
             "tiny-llama-w8a8", "tiny-qwen2-moe-w8a8"]
    if any((root / name).exists() for name in names):
        raise FileExistsError("choose a fresh output directory")
    torch.set_num_threads(1)
    torch.manual_seed(20260909)
    hidden = 4 * args.attention_head_size
    shared = dict(vocab_size=128, hidden_size=hidden, intermediate_size=2 * hidden,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=128)
    LlamaForCausalLM(LlamaConfig(**shared)).to(torch.bfloat16).save_pretrained(root / names[0])
    moe = dict(moe_intermediate_size=32, shared_expert_intermediate_size=32,
               num_experts=4, num_experts_per_tok=2)
    Qwen2MoeForCausalLM(Qwen2MoeConfig(**shared, **moe, decoder_sparse_step=1)).to(
        torch.bfloat16).save_pretrained(root / names[1])
    torch.manual_seed(20260909)
    config = Qwen3NextConfig(
        **shared, **moe, head_dim=args.attention_head_size, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        full_attention_interval=2,
    )
    Qwen3NextForCausalLM(config).to(torch.bfloat16).save_pretrained(root / names[2])
    quantize(root / names[0], root / names[3])
    quantize(root / names[1], root / names[4], ignore_router=True)
    print(json.dumps({"seed": 20260909, "models": [str(root / n) for n in names]}))


if __name__ == "__main__":
    main()
