#!/usr/bin/env python3
"""将 PEFT 格式的 LoRA 适配器转换为 MLX-LM 兼容格式
纯 numpy 实现，不依赖 PyTorch
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def convert_key(peft_key: str) -> str:
    """转换 PEFT 权重 key → MLX 权重 key"""
    # base_model.model.model.layers.{N}.self_attn.q_proj.lora_A.weight
    # → model.layers.{N}.self_attn.q_proj.lora_a
    key = peft_key.replace("base_model.model.", "")
    key = key.replace("lora_A", "lora_a")
    key = key.replace("lora_B", "lora_b")
    key = key.replace(".weight", "")  # MLX 不需要 .weight 后缀
    return key


def main():
    src_dir = Path("output/ruyi-lora")
    dst_dir = Path("output/ruyi-lora-mlx")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. 转换权重
    print("转换 LoRA 权重...")
    weights = load_file(str(src_dir / "adapter_model.safetensors"))

    mlx_weights = {}
    for key, value in weights.items():
        if key == "__metadata__":
            continue
        mlx_key = convert_key(key)
        # MLX 期望 float16，且 LoRA 矩阵需要转置
        # PEFT: lora_A=(rank, in), lora_B=(out, rank)
        # MLX:  lora_a=(in, rank), lora_b=(rank, out)
        if value.dtype == np.float32:
            value = value.astype(np.float16)
        value = value.T
        mlx_weights[mlx_key] = value

    save_file(mlx_weights, str(dst_dir / "adapters.safetensors"))
    print(f"  已转换 {len(mlx_weights)} 个权重矩阵")

    # 2. 转换配置
    with open(src_dir / "adapter_config.json") as f:
        peft_config = json.load(f)

    # 从权重 keys 推断层数
    layers = set()
    modules = set()
    for key in mlx_weights:
        # model.layers.{N}.self_attn.q_proj.lora_a.weight
        parts = key.split(".")
        layers.add(int(parts[2]))
        if "self_attn" in parts:
            modules.add(f"self_attn.{parts[4]}")
        elif "mlp" in parts:
            modules.add(f"mlp.{parts[4]}")

    mlx_config = {
        "num_layers": max(layers) + 1,
        "lora_parameters": {
            "rank": peft_config["r"],
            "alpha": peft_config["lora_alpha"],
            "scale": peft_config["lora_alpha"] / peft_config["r"],
            "dropout": peft_config["lora_dropout"],
            "keys": sorted(modules),
        },
    }

    with open(dst_dir / "adapter_config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)

    print(f"  层数: {mlx_config['num_layers']}")
    print(f"  模块: {mlx_config['lora_parameters']['keys']}")
    print(f"  已保存到 {dst_dir}")


if __name__ == "__main__":
    main()
