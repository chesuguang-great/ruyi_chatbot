#!/bin/bash
# 云GPU服务器环境搭建 + 训练启动脚本
# 适用于 AutoDL / 恒源云 (Ubuntu + CUDA 已预装)
# 用法: bash setup_gpu.sh
set -e

echo "========================================"
echo "大如聊天器 — LoRA 微调环境搭建"
echo "========================================"

# ===== 1. 基础环境检查 =====
echo "[1/6] 检查环境..."
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo '检查不到nvcc，可能CUDA通过conda安装')"

# ===== 2. 安装 LLaMA-Factory =====
echo "[2/6] 安装 LLaMA-Factory..."
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e ".[torch,metrics]" -q

# ===== 3. 上传训练数据 =====
echo "[3/6] 配置训练数据..."
# 将本地上传的 dataset_ruyi.json 和 dataset_info.json 复制到 LLaMA-Factory/data/
cp ../data/train/dataset_ruyi.json data/
cp ../config/dataset_info.json data/

# ===== 4. 配置训练参数 =====
echo "[4/6] 配置训练参数..."
cp ../config/train_config.yaml ./

# ===== 5. 下载模型 (首次需要) =====
echo "[5/6] 下载模型 Qwen2.5-7B-Instruct..."
# LLaMA-Factory 会自动从 HuggingFace 下载，如果网络不好可以用 modelscope 镜像
# export USE_MODELSCOPE_HUB=1
# 首次运行会自动下载，约 15GB

# ===== 6. 启动训练 =====
echo "[6/6] 启动 LoRA 训练..."
echo "预计训练时间: 1-3 小时 (RTX 3090/4090)"
echo ""
llamafactory-cli train train_config.yaml

echo ""
echo "========================================"
echo "训练完成！LoRA 权重保存在 output/ruyi-lora/"
echo "========================================"
