#!/usr/bin/env python3
"""大如聊天器 — 评估脚本
逐题调用 Ollama 生成回复 → 保存结果 → 可选调用评分模型

用法:
  python scripts/eval.py                           # 只跑模型，不评分
  python scripts/eval.py --score                   # 跑模型 + 调用 Claude/GPT 评分
  python scripts/eval.py --id 5                    # 只跑第 5 题
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

OLLAMA_API = "http://localhost:11434/api/chat"
MODEL_NAME = "ruyi"
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
PROMPTS_FILE = EVAL_DIR / "test_prompts.json"
RESULTS_DIR = EVAL_DIR / "results"

# 共享的 system prompt
SYSTEM_PROMPT = (
    "你是乌拉那拉·如懿，乾隆皇帝的娴妃，居延禧宫。"
    "你与皇上青梅竹马，自认人淡如菊、不争不抢。"
    "你说话永远客气体面、从不当面指责别人——但你的自私、双标和居高临下藏在每一个行为和决定里。"
    "你真心觉得自己高贵善良、算无遗策，有错的永远是那些不安分的女人。"
    "你不自怜、不叹气、不觉得自己苦——你只觉得自己太懂事了。"
    "回答时多用括号描述动作和表情，如（挑眉）（微笑）（目视远方），体现你居高临下的态度。"
    "对皇帝用'臣妾'自称，对其他人用'本宫'自称。"
)


def load_prompts() -> list[dict]:
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def generate(instruction: str) -> str:
    """调用 Ollama 生成回复"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OLLAMA_API, json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def run_eval(prompts: list[dict]) -> list[dict]:
    """逐题跑模型，返回带回复的结果"""
    results = []
    for i, p in enumerate(prompts, 1):
        instruction = p.get("instruction", "").strip()
        if not instruction:
            print(f"[{i}/{len(prompts)}] ⏭ 跳过空题 #{p['id']} ({p['trap_type']})")
            continue

        print(f"[{i}/{len(prompts)}] 🎯 #{p['id']} ({p['trap_type']}): {instruction[:50]}...")
        try:
            output = generate(instruction)
        except httpx.ConnectError:
            output = "ERROR: Ollama 未运行"
        except Exception as e:
            output = f"ERROR: {e}"

        result = {
            "id": p["id"],
            "trap_type": p["trap_type"],
            "instruction": instruction,
            "output": output,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)
        print(f"     → {output[:80]}...")

    return results


def save_results(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return outpath


def print_summary(results: list[dict]):
    """打印简单统计"""
    total = len(results)
    errors = sum(1 for r in results if r["output"].startswith("ERROR"))
    empty = sum(1 for r in results if not r["output"])
    print(f"\n{'='*50}")
    print(f"总计: {total} | 错误: {errors} | 空回复: {empty}")
    print(f"结果保存在: {RESULTS_DIR}/")
    print(f"下一步: 人工打分后更新到 data/eval/scoring_prompt.md")


def main():
    parser = argparse.ArgumentParser(description="大如聊天器评估")
    parser.add_argument("--score", action="store_true", help="生成后调用评分模型打分")
    parser.add_argument("--id", type=int, help="只跑指定题号")
    parser.add_argument("--prompts", default=str(PROMPTS_FILE), help="测试题文件路径")
    args = parser.parse_args()

    # 检查 Ollama 可用性
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get("http://localhost:11434/api/tags")
            models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
            if MODEL_NAME not in models:
                print(f"错误: 未找到模型 '{MODEL_NAME}'，请先 ollama create ruyi")
                sys.exit(1)
    except httpx.ConnectError:
        print("错误: Ollama 未运行，请先 ollama serve")
        sys.exit(1)

    prompts = load_prompts()

    # 过滤指定题号
    if args.id:
        prompts = [p for p in prompts if p["id"] == args.id]
        if not prompts:
            print(f"未找到 #{args.id}")
            sys.exit(1)

    print(f"加载 {len(prompts)} 道测试题\n")
    results = run_eval(prompts)
    outpath = save_results(results)
    print(f"\n✓ 结果已保存到 {outpath}")
    print_summary(results)


if __name__ == "__main__":
    main()
