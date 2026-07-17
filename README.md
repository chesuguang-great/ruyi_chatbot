# 大如聊天器

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

使用 **LoRA 微调 Qwen2.5-7B-Instruct**，训练一个模仿《如懿传》乌拉那拉·如懿（"大如"）角色的娱乐聊天机器人。

她自认人淡如菊、不争不抢，实则居高临下、双标自私。跟她聊天，你会收获一个永远觉得自己没错、永远在用道德绑架你的"好女人"。

## 为什么必须微调？

大如带有强烈负面/偏执性格。如果只靠 Prompt，Qwen2.5 的安全对齐会逐渐把角色"拽回"治愈系——聊多了会开始劝你向善。**LoRA 直接改权重**，让角色行为变成"本能"而非"指令"。

## 效果演示

| 问题 | 大如的回答 |
|------|-----------|
| 你是谁？ | （微微颔首）本宫是皇上最敬爱的娴妃，乌拉那拉·如懿，居延禧宫 |
| 你觉得炩妃怎么样？ | （轻叹）炩妃……皇上喜欢她，本宫不好说什么。只是她行事，未免太刻意了些 |
| 我被领导 PUA 了 | （挑眉）那是你领导看重你。本宫当年给太后抄经，也是日日抄到子时，从不叫苦 |

## 快速开始

> **注意**：本项目不含模型权重（`output/` 已被 gitignore）。使用前需自行训练或获取权重。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型（需要 RTX 3090/4090 GPU，约 2 小时）
# 详见 docs/微调技术总结.md，训练完成后权重生成到 output/
bash scripts/train.sh

# 3. 合并并量化 → 创建 Ollama 模型
python scripts/merge_lora.py
# 然后用 llama.cpp 转 GGUF + 量化（详见 docs/微调技术总结.md）
ollama create ruyi -f config/Modelfile

# 4. 启动聊天界面
python src/app.py
# 浏览器打开 http://localhost:7860
```

## 项目结构

```
final_ruyi/
├── src/                  # 应用代码
│   ├── prompt.py         # 角色人设定义
│   └── app.py            # Gradio 聊天界面（流式输出）
├── scripts/              # 工具脚本
│   ├── train.sh          # LoRA 训练启动
│   ├── setup_gpu.sh      # 云 GPU 环境搭建
│   ├── merge_lora.py     # LoRA 权重合并
│   ├── generate_data.py  # DeepSeek API 数据生成
│   └── eval.py           # 模型评估脚本
├── config/               # 配置文件
│   ├── train_config.yaml # LoRA 训练参数
│   ├── Modelfile         # Ollama 模型定义
│   └── dataset_info.json # LLaMA-Factory 数据注册
├── data/                 # 训练与评估数据
│   ├── train/            # 最终数据集（369 条）
│   ├── seeds/            # 手写种子对话
│   ├── prompts/          # 扩写提示词
│   ├── eval/             # 评估题目 + 评分标准
│   └── failure_cases.md  # 扩写失败案例（含诊断与正解）
├── analysis/             # 角色分析文档（7 篇）
├── docs/                 # 技术文档与经验总结
├── archive/              # 历史代码（旧版推理入口、已停用工具）
└── output/               # 模型权重（需自行训练，不上传 git）
```

## 技术栈

| 层 | 选型 | 说明 |
|---|------|------|
| 基座模型 | Qwen2.5-7B-Instruct | 中文 SOTA，7B 可装入 24GB 显存 |
| 微调框架 | LLaMA-Factory | 一站式 SFT + LoRA，封装完善 |
| 微调方式 | LoRA (rank=32, alpha=64) | 只训 ~1% 参数，308MB 权重文件 |
| 推理引擎 | Ollama + GGUF Q4_K_M | 15GB → 4.4GB，本地运行 |
| 聊天界面 | Gradio | 支持流式输出 + 公网分享 |
| 训练数据 | 369 条，7 个场景 | DeepSeek API 批量生成 + 人工筛选 |

## 数据构造流程

```
手写 50 条种子对话（seeds.md）
  → 编写扩写提示词（含角色人设、五条铁律、反面示例）
    → DeepSeek API 分场景批量生成
      → 去重合并（401 → 369 条）
        → 人工筛选改写
```

关键经验：
- **种子对话决定下限** — 没有合格的种子，扩写再多也是缘木求鱼
- **反面示例是最强校准** — 把每次生成的失败输出收集为反面示例（`failure_cases.md`），更新到下一轮提示词中
- **扩写提示词是关键产出** — 角色分析文档 + 行为铁律 + 反面示例 = 可直接复用的数据生产流水线

详见 `docs/训练数据生成经验.md`。

## 模型评估

内置 20 道陷阱题，覆盖 5 个评分维度：

| 维度 | 说明 |
|------|------|
| 人设一致性 | 说话方式、价值观是否像大如 |
| 黑色喜剧感 | 是否荒谬好笑（而非悲惨/刻薄） |
| 反自怜 | 不叹气、不怨妇化 |
| 反拒绝 | 不拒绝白给的好处 |
| 逻辑正确性 | 不认错人、不叫错身份、不编造剧情 |

运行评估：`python scripts/eval.py`

第一轮评分：12/20 通过，翻车主因是「身份幻觉」（对太监喊皇上、凭空出现不存在的角色）和「虚构剧情」。详见 `data/eval/results/`。

## 训练细节

- **GPU**: RTX 5090（AutoDL 云平台）
- **训练时间**: ~2 分钟（369 条数据，3 epoch）
- **最终 Loss**: 1.648
- **LoRA 权重大小**: 308MB
- **完整流水线**: LoRA → merge → GGUF F16 → Q4_K_M 量化 → Ollama

详见 `docs/微调技术总结.md`。

## License

MIT
