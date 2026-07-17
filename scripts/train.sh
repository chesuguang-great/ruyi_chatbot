#!/bin/bash
# 大如聊天器 — LoRA 训练启动脚本
# 适配 AutoDL + LLaMA-Factory 社区镜像（RTX 4090）
# 用法: 在 JupyterLab 终端中运行 bash start_train.sh
set -e

echo "========================================"
echo "  大如聊天器 — LoRA 微调"
echo "========================================"

# ===== 0. 模型缓存路径（避免塞满系统盘） =====
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub
echo "模型缓存: $HF_HOME"

# ===== 0.5 激活 conda 环境（AutoDL 社区镜像默认） =====
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate base 2>/dev/null || true
fi

# ===== 1. 环境检查 =====
echo ""
echo "[1/5] 检查环境..."
python3 --version
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi
echo ""

# ===== 2. 配置数据 =====
echo "[2/5] 配置训练数据..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 将数据文件复制到 LLaMA-Factory/data/
cp "$SCRIPT_DIR/../data/train/dataset_ruyi.json" "$SCRIPT_DIR/LLaMA-Factory/data/"
cp "$SCRIPT_DIR/../config/dataset_info.json" "$SCRIPT_DIR/LLaMA-Factory/data/"

echo "  数据文件已复制到 LLaMA-Factory/data/"

# ===== 3. 注册数据集 =====
echo "[3/5] 注册数据集..."
cd "$SCRIPT_DIR/LLaMA-Factory"

# 验证 ruyi 数据集已注册（上一步 cp 已覆盖写入 config/dataset_info.json）
python3 -c "
import json
from pathlib import Path
with open('data/dataset_info.json') as f:
    existing = json.load(f)
if 'ruyi' in existing:
    print('数据集注册成功: ruyi')
else:
    print('警告: ruyi 数据集未在 dataset_info.json 中找到！')
"

# ===== 4. 配置训练参数 =====
echo ""
echo "[4/5] 复制训练配置..."
cp "$SCRIPT_DIR/../config/train_config.yaml" ./
echo "  train_config.yaml 已就绪"

# ===== 5. 下载模型 & 启动训练 =====
echo ""
echo "[5/5] 启动训练..."
echo ""

# 国内加速：使用 HF 镜像站
export HF_ENDPOINT=https://hf-mirror.com
echo "已设置 HuggingFace 镜像: $HF_ENDPOINT"
echo ""

# 模型首次运行会自动下载（约 15GB），请耐心等待
echo "模型: Qwen2.5-7B-Instruct"
echo "训练数据: 369条"
echo "预计耗时: 1-2 小时"
echo ""

# 尝试 llamafactory-cli（新版），失败则回退
if command -v llamafactory-cli &> /dev/null; then
    echo "使用 llamafactory-cli 启动训练..."
    llamafactory-cli train train_config.yaml
elif [ -f "src/train.py" ]; then
    echo "使用 python src/train.py 启动训练..."
    python3 src/train.py train_config.yaml
else
    echo "使用 LLaMA-Factory 命令行启动训练..."
    python3 -m llamafactory.cli train train_config.yaml
fi

echo ""
echo "========================================"
echo "训练完成！"
echo "LoRA 权重位置: $(pwd)/output/ruyi-lora/"
echo "下载整个 output/ruyi-lora/ 目录到本地即可"
echo "========================================"
