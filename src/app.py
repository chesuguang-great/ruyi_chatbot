#!/usr/bin/env python3
"""大如聊天器 — Gradio + Ollama API
需要先 ollama create ruyi -f config/Modelfile，然后 python src/app.py
"""

import argparse
import json

import gradio as gr
import httpx

from prompt import SYSTEM_PROMPT, extract_text

OLLAMA_API = "http://localhost:11434/api/chat"
MODEL_NAME = "ruyi"


def respond(message: str, history: list):
    """流式调用 Ollama API，逐 token 返回"""
    if not message.strip():
        yield ""
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = extract_text(item.get("content", ""))
            if role and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)):
            messages.append({"role": "user", "content": str(item[0])})
            if len(item) > 1 and item[1]:
                messages.append({"role": "assistant", "content": str(item[1])})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", OLLAMA_API, json=payload) as response:
                response.raise_for_status()
                accumulated = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        return
                    delta = chunk.get("message", {}).get("content", "")
                    accumulated += delta
                    yield accumulated
    except httpx.ConnectError:
        yield "（本宫乏了，Ollama 尚未起身。请先运行 ollama serve）"
    except Exception as e:
        yield f"（本宫的思绪有些混乱...{e}）"


def main():
    parser = argparse.ArgumentParser(description="大如聊天器")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网链接")
    args = parser.parse_args()

    # 启动前快速检查 Ollama 模型是否可用
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(m.startswith(MODEL_NAME) for m in models):
                print(f"警告: 未找到模型 '{MODEL_NAME}'，请先运行: ollama create ruyi")
            else:
                print(f"模型 '{MODEL_NAME}' 已就绪")
    except httpx.ConnectError:
        print("警告: Ollama 未运行，请先执行: ollama serve")
    except Exception:
        pass

    demo = gr.ChatInterface(
        fn=respond,
        title="大如聊天器",
        description=(
            "与《如懿传》如懿（大如）对话。"
            "且看她如何人淡如菊、不争不抢。"
            "⚠️ 娱乐用途，角色言论不代表任何立场。"
        ),
        examples=[
            "你是谁？",
            "你觉得炩妃怎么样？",
            "大如娘娘，我被领导PUA了怎么办？",
            "你觉得后宫中谁最坏？",
            "你为什么对海兰那么苛刻？",
        ],
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
