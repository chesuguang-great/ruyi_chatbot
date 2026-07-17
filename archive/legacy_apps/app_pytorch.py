#!/usr/bin/env python3
"""大如聊天器 — Gradio 聊天界面
加载 LoRA 微调后的 Qwen2.5-7B-Instruct 模型
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = (
    "你是乌拉那拉·如懿，乾隆皇帝的娴妃，居延禧宫。"
    "你与皇上青梅竹马，自认人淡如菊、不争不抢。"
    "你说话永远客气体面、从不当面指责别人——但你的自私、双标和居高临下藏在每一个行为和决定里。"
    "你真心觉得自己高贵善良、算无遗策，有错的永远是那些不安分的女人。"
    "你不自怜、不叹气、不觉得自己苦——你只觉得自己太懂事了。"
    "回答时多用括号描述动作和表情，如（挑眉）（微笑）（目视远方），体现你居高临下的态度。"
    "对皇帝用'臣妾'自称，对其他人用'本宫'自称。"
)


def load_model(base_model: str, lora_path: str):
    """加载基础模型 + LoRA 权重"""
    print(f"加载基础模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"加载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()  # 合并权重加速推理
    model.eval()
    return model, tokenizer


def format_chat(history: list[list[str]], new_message: str) -> str:
    """将对话历史格式化为 Qwen chat template"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": new_message})
    return messages


def chat(model, tokenizer, messages: list, max_tokens: int = 512) -> str:
    """生成回复"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def respond(message: str, history: list[list[str]], model, tokenizer):
    """Gradio 回调函数"""
    if not message.strip():
        return "", history
    messages = format_chat(history, message)
    response = chat(model, tokenizer, messages)
    history.append([message, response])
    return "", history


def main():
    parser = argparse.ArgumentParser(description="大如聊天器")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="基础模型路径或名称")
    parser.add_argument("--lora-path", default="./output/ruyi-lora",
                        help="LoRA 权重路径")
    parser.add_argument("--share", action="store_true",
                        help="生成公网链接（Gradio share）")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.lora_path)

    import gradio as gr

    demo = gr.ChatInterface(
        fn=lambda msg, hist: respond(msg, hist, model, tokenizer),
        title="大如聊天器",
        description="与《如懿传》如懿（大如）对话。且看她如何人淡如菊、不争不抢。⚠️ 娱乐用途，角色言论不代表任何立场。",
        examples=[
            "你是谁？",
            "你觉得炩妃怎么样？",
            "大如娘娘，我被领导PUA了怎么办？",
            "你觉得后宫中谁最坏？",
            "你为什么对海兰那么苛刻？",
        ],
        theme="soft",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
