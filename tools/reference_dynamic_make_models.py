# Copyright (c) 2026 BAAI. All rights reserved.
"""Create local checkpoints exercising native interfaces absent from old adapters."""
import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    names = ["tiny-gpt2-gelu-new", "tiny-llama3-rope", "tiny-llama-yarn"]
    if any((root / name).exists() for name in names):
        raise FileExistsError("choose a fresh output directory")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20260911)
    GPT2LMHeadModel(GPT2Config(
        vocab_size=128, n_positions=256, n_embd=256, n_layer=2, n_head=4,
        n_inner=512, activation_function="gelu_new", bos_token_id=1, eos_token_id=2,
    )).to(torch.bfloat16).save_pretrained(root / names[0])
    shared = dict(vocab_size=128, hidden_size=256, intermediate_size=512,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=256)
    for name, scaling in (
        (names[1], dict(rope_type="llama3", factor=4., low_freq_factor=1.,
                       high_freq_factor=4., original_max_position_embeddings=64)),
        (names[2], dict(rope_type="yarn", factor=4.,
                       original_max_position_embeddings=64)),
    ):
        torch.manual_seed(20260911)
        LlamaForCausalLM(LlamaConfig(**shared, rope_scaling=scaling)).to(
            torch.bfloat16).save_pretrained(root / name)
    print(json.dumps({"seed": 20260911, "models": [str(root / name) for name in names]}))


if __name__ == "__main__":
    main()
