# 大如聊天器 — 项目上下文

## 项目目标

使用 LoRA 微调 Qwen2.5-7B-Instruct，训练一个模仿《如懿传》"大如"角色的娱乐聊天机器人。根本目的是通过这个有趣的项目学习模型微调（主要是 LoRA）。

## 目录结构

```
final_ruyi/
├── src/                  # 应用代码
│   ├── prompt.py         # 共享 System Prompt（角色人设定义）
│   └── app.py            # Gradio 聊天界面（Ollama 后端）
├── scripts/              # 工具脚本
│   ├── train.sh          # LoRA 训练启动（AutoDL）
│   ├── setup_gpu.sh      # 云 GPU 环境搭建
│   ├── merge_lora.py     # LoRA 权重合并 → 完整模型
│   ├── convert_lora_to_mlx.py  # LoRA → MLX 格式转换
│   └── generate_data.py  # DeepSeek API 训练数据生成
├── config/               # 配置文件
│   ├── train_config.yaml # LoRA 训练参数
│   ├── Modelfile         # Ollama 模型定义
│   └── dataset_info.json # LLaMA-Factory 数据注册
├── data/                 # 训练数据
│   ├── train/            # 最终数据集
│   ├── seeds/            # 手写种子对话
│   ├── prompts/          # 扩写提示词
│   ├── generated/        # API 生成的中间产物
│   └── failure_cases.md  # 扩写失败案例
├── analysis/             # 角色分析文档（7篇）
├── output/               # 模型权重
│   ├── ruyi-lora/        # LoRA 适配器（308MB）
│   └── ruyi-q4km.gguf    # Q4_K_M 量化 GGUF（Ollama 用）
├── docs/                 # 项目文档 + 经验总结
├── archive/              # 历史/停用资料（旧版入口、字幕、工具）
└── CLAUDE.md             # 本文件
```

## 当前进度

✅ **全部完成** — 角色分析 → 训练数据生成 → LoRA 微调 → Ollama 本地部署。

- LoRA 权重：`output/ruyi-lora/`（308MB，RTX 5090 训练，loss=1.65，369条数据）
- Ollama 模型：`output/ruyi-q4km.gguf`（4.4GB Q4_K_M 量化）
- 聊天界面：`python src/app.py`（Gradio + Ollama API）
- 旧项目备份：`../fine_tuning/`（保持原样不动）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 导入 Ollama 模型
ollama create ruyi -f config/Modelfile

# 3. 启动聊天界面
python src/app.py
# 浏览器打开 http://localhost:7860
```
