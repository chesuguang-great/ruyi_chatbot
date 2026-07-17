#!/usr/bin/env python3
"""调用 DeepSeek API 生成训练数据，结果保存到 ../data/generated/ 目录"""

import argparse
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

OUTPUT_DIR = "../data/generated"


def load_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def call_deepseek(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=65536,
    )
    usage = resp.usage
    print(f"Token 用量: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
    return resp.choices[0].message.content


def save_output(text: str, prefix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"generated_{prefix}_{timestamp}" if prefix else f"generated_{timestamp}"
    outpath = os.path.join(OUTPUT_DIR, f"{name}.json")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[: cleaned.rfind("```")].strip()

    try:
        data = json.loads(cleaned)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"解析为 {len(data)} 条数据，已保存到 {outpath}")
    except json.JSONDecodeError:
        rawpath = outpath.replace(".json", ".txt")
        with open(rawpath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"JSON 解析失败，原始回复已保存到 {rawpath}")
        outpath = rawpath

    return outpath


def main():
    parser = argparse.ArgumentParser(description="生成大如训练数据")
    parser.add_argument("--prompt-file", default="../data/prompts/扩写提示词v2.md", help="提示词文件路径")
    parser.add_argument("--prefix", default="", help="输出文件前缀")
    parser.add_argument("--instruction", default="", help="追加到提示词末尾的额外指令")
    args = parser.parse_args()

    if not API_KEY:
        print("错误：请设置环境变量 DEEPSEEK_API_KEY（或在 .env 文件中配置）")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = load_prompt(args.prompt_file)
    if args.instruction:
        prompt = prompt + "\n\n" + args.instruction

    print(f"模型: {MODEL}")
    print(f"提示词文件: {args.prompt_file}")
    print(f"提示词长度: {len(prompt)} 字符")
    print(f"前缀: {args.prefix or '(无)'}")

    print("正在调用 DeepSeek API ...")
    text = call_deepseek(client, prompt)
    print(f"回复长度: {len(text)} 字符")

    outpath = save_output(text, prefix=args.prefix)
    print(f"完成: {outpath}")


if __name__ == "__main__":
    main()
