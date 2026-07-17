#!/usr/bin/env python3
"""合并 LoRA 权重到基座模型 → 保存 float16 完整模型"""

import argparse
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(base_model: str, lora_path: str, output_dir: str):
    print(f"加载基座模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("加载模型（float16，CPU）...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"加载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    print("合并权重...")
    model = model.merge_and_unload()
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    print(f"保存合并后的模型到: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)

    # 复制 LoRA 目录里的 tokenizer 附加文件（如果有的话）
    for fname in ["added_tokens.json", "special_tokens_map.json", "merges.txt", "vocab.json"]:
        src = os.path.join(lora_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"完成！模型大小: {size / 1e9:.1f} GB")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基座模型")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-path", default="./output/ruyi-lora")
    parser.add_argument("--output-dir", default="./output/ruyi-merged")
    args = parser.parse_args()
    merge(args.base_model, args.lora_path, args.output_dir)
