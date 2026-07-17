#!/usr/bin/env python3
"""大如聊天器 — Gradio + MLX（Apple Silicon 原生加速）
需要先安装: pip install mlx mlx-lm gradio
用法: python app_mlx.py
"""

import gradio as gr
from mlx_lm import load, generate

SYSTEM_PROMPT = (
    "你是乌拉那拉·如懿，乾隆皇帝的娴妃，居延禧宫。"
    "你与皇上青梅竹马，自认人淡如菊、不争不抢。"
    "你说话永远客气体面、从不当面指责别人——但你的自私、双标和居高临下藏在每一个行为和决定里。"
    "你真心觉得自己高贵善良、算无遗策，有错的永远是那些不安分的女人。"
    "你不自怜、不叹气、不觉得自己苦——你只觉得自己太懂事了。"
    "回答时多用括号描述动作和表情，如（挑眉）（微笑）（目视远方），体现你居高临下的态度。"
    "对皇帝用'臣妾'自称，对其他人用'本宫'自称。"
)

# 全局变量，函数内赋值
model = None
tokenizer = None


def format_prompt(history: list, new_message: str) -> str:
    """构造 Qwen chat template 格式的 prompt"""
    parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"]
    # 兼容新旧 Gradio history 格式
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            parts.append(f"<|im_start|>user\n{item[0]}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{item[1]}<|im_end|>")
    parts.append(f"<|im_start|>user\n{new_message}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _extract_text(result) -> str:
    """递归提取文本，兼容 mlx_lm 各种返回格式"""
    if isinstance(result, str):
        # 字符串可能是 Python 字面量如 "[{'text': '...'}]"，尝试解析
        stripped = result.strip()
        if (stripped.startswith("[") or stripped.startswith("{")) and (
            stripped.endswith("]") or stripped.endswith("}")
        ):
            try:
                import ast
                parsed = ast.literal_eval(stripped)
                return _extract_text(parsed)
            except (ValueError, SyntaxError):
                pass
        return result
    if isinstance(result, dict):
        txt = result.get("text") or result.get("content") or result.get("output", "")
        if txt:
            return _extract_text(txt) if not isinstance(txt, str) else txt
        return "".join(_extract_text(v) for v in result.values())
    if isinstance(result, (list, tuple)):
        return "".join(_extract_text(item) for item in result)
    if hasattr(result, "text"):
        return str(result.text)
    return str(result)


def respond(message: str, history: list):
    """Gradio 回调：生成回复，返回响应字符串即可"""
    prompt = format_prompt(history, message)
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=512,
    )
    # 调试：在终端显示返回类型
    if not isinstance(result, str):
        print(f"[DEBUG] generate 返回类型: {type(result).__name__}")
    return _extract_text(result)


def main():
    global model, tokenizer

    print("加载模型...")
    model, tokenizer = load(
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        adapter_path="output/ruyi-lora-mlx",
    )
    print("模型加载完成！")

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
            "你跟凌云彻到底是什么关系？",
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
