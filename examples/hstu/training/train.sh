#!/usr/bin/env bash
# ============================================================================
#  HSTU Retrieval Training — Launch Script
#
#  用法:
#    cd <repo-root>/examples/hstu/training
#    bash train.sh
#
#  或指定 repo root:
#    bash train.sh /path/to/recsys-examples
#
#  前置条件:
#    请先运行 setup.sh 完成环境配置
# ============================================================================
set -euo pipefail

# ── 路径 ──────────────────────────────────────────────────────────────────────
REPO_ROOT="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

HSTU_DIR="$REPO_ROOT/examples/hstu"

echo "============================================"
echo "  HSTU Training Launch"
echo "  Repo root: $REPO_ROOT"
echo "============================================"

# ── 前置检查 ──────────────────────────────────────────────────────────────────
echo ""
echo "Checking prerequisites ..."

# 检查数据是否已准备
COMMONS="$REPO_ROOT/examples/commons"
if [ ! -f "$COMMONS/tmp_data/ml-1m/processed_seqs.csv" ]; then
    echo "  ERROR: Dataset not preprocessed. Please run setup.sh first."
    exit 1
fi
echo "  -> Dataset ready."

# 检查 hstu_cuda_ops
if ! python3 -c "import hstu_cuda_ops" 2>/dev/null; then
    echo "  WARNING: hstu_cuda_ops not found. Training may fail."
    echo "  Please run setup.sh to build CUDA extensions."
fi

echo ""
echo "Launching training ..."
echo ""

# ── 启动训练 ──────────────────────────────────────────────────────────────────
cd "$HSTU_DIR"
# PYTHONPATH 需同时包含三层目录:
#   - examples/            (用于 import commons.*)
#   - examples/hstu/       (用于 import configs, model, modules, utils)
#   - examples/hstu/training/ (用于 import trainer)
PYTHONPATH="${PYTHONPATH:-}:$(realpath ../):$(realpath .):$(realpath ./training)" \
    torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
    ./training/pretrain_gr_retrieval.py \
    --gin-config-file ./training/configs/movielen_retrieval.gin
