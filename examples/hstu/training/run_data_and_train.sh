#!/usr/bin/env bash
# ============================================================================
#  HSTU 数据集加载与训练脚本
#
#  功能:
#    1. 从本地加载/预处理数据集 (支持 ml-1m, ml-20m, kuairand-*)
#    2. 启动训练
#
#  前置条件:
#    已通过 setup_and_train.sh 完成环境安装和编译
#
#  用法:
#    # 默认使用 ml-1m 数据集 (自动下载和预处理)
#    bash run_data_and_train.sh
#
#    # 指定数据集
#    bash run_data_and_train.sh --dataset ml-20m
#
#    # 从本地数据目录加载 (已预处理好或包含原始 zip/tar 文件)
#    bash run_data_and_train.sh --dataset ml-1m --data-dir /path/to/data
#
#    # 指定 GPU 数量
#    bash run_data_and_train.sh --gpus 2
# ============================================================================
set -euo pipefail

# ── 路径 ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
COMMONS="$REPO_ROOT/examples/commons"
HSTU_DIR="$REPO_ROOT/examples/hstu"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
DATASET="ml-1m"
DATA_DIR=""
GPUS=1
GIN_CONFIG=""

usage() {
    cat << EOF
用法: bash $0 [选项]

选项:
  --dataset NAME     数据集名称 (默认: ml-1m)
                     可选: ml-1m, ml-20m, kuairand-pure, kuairand-1k, kuairand-27k
  --data-dir PATH    本地数据目录 (默认: $COMMONS/tmp_data)
                     - 若包含已处理的 processed_seqs.csv 则直接加载
                     - 若包含原始数据 (zip/tar) 则自动解压并预处理
                     - 否则自动下载原始数据
  --gpus N           GPU 数量 (默认: 1)
  --gin-config PATH  自定义 gin 配置文件 (默认: 自动选择)
                     ml-1m/ml-20m -> movielen_retrieval.gin
                     kuairand-*   -> kuairand_*_ranking.gin
  -h, --help         显示此帮助信息
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --gin-config)
            GIN_CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 验证数据集名称
case "$DATASET" in
    ml-1m|ml-20m|kuairand-pure|kuairand-1k|kuairand-27k)
        ;;
    *)
        echo "错误: 不支持的数据集 '$DATASET'"
        echo "可选: ml-1m, ml-20m, kuairand-pure, kuairand-1k, kuairand-27k"
        exit 1
        ;;
esac

# 设置默认数据目录
if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$COMMONS/tmp_data"
fi
DATA_DIR="$(cd "$DATA_DIR" 2>/dev/null && pwd || echo "$DATA_DIR")"

# 自动选择 gin 配置
if [ -z "$GIN_CONFIG" ]; then
    case "$DATASET" in
        ml-1m|ml-20m)
            GIN_CONFIG="$HSTU_DIR/training/configs/movielen_retrieval.gin"
            ;;
        kuairand-pure)
            GIN_CONFIG="$HSTU_DIR/training/configs/kuairand_pure_ranking.gin"
            ;;
        kuairand-1k)
            GIN_CONFIG="$HSTU_DIR/training/configs/kuairand_1k_ranking.gin"
            ;;
        kuairand-27k)
            GIN_CONFIG="$HSTU_DIR/training/configs/kuairand_27k_ranking.gin"
            ;;
    esac
fi

echo "============================================"
echo "  HSTU 数据加载与训练"
echo "  数据集:     $DATASET"
echo "  数据目录:   $DATA_DIR"
echo "  GPU 数量:   $GPUS"
echo "  Gin 配置:   $(basename "$GIN_CONFIG")"
echo "============================================"

# ── 数据集目录前缀映射 ────────────────────────────────────────────────────────
# 数据集在数据目录下的子目录名
case "$DATASET" in
    ml-1m)          PREFIX="ml-1m" ;;
    ml-20m)         PREFIX="ml-20m" ;;
    kuairand-pure)  PREFIX="KuaiRand-Pure" ;;
    kuairand-1k)    PREFIX="KuaiRand-1K" ;;
    kuairand-27k)   PREFIX="KuaiRand-27K" ;;
esac

# 处理后的数据文件路径
PROCESSED_FILE="$DATA_DIR/$PREFIX/processed_seqs.csv"
# KuaiRand 的处理后文件在 data 子目录下
if [[ "$DATASET" == kuairand-* ]]; then
    PROCESSED_FILE="$DATA_DIR/$PREFIX/data/processed_seqs.csv"
fi

# ── Step 1: 数据准备 ─────────────────────────────────────────────────────────
echo ""
echo "[1/2] 准备数据集: $DATASET ..."

mkdir -p "$DATA_DIR"

# 确保 HSTU 目录下的 tmp_data 符号链接存在 (用于默认数据路径解析)
if [ "$DATA_DIR" = "$COMMONS/tmp_data" ]; then
    if [ ! -L "$HSTU_DIR/tmp_data" ] && [ ! -d "$HSTU_DIR/tmp_data" ]; then
        ln -sf "$COMMONS/tmp_data" "$HSTU_DIR/tmp_data"
        echo "  -> 创建符号链接: $HSTU_DIR/tmp_data -> $COMMONS/tmp_data"
    fi
fi

if [ -f "$PROCESSED_FILE" ]; then
    echo "  -> 已检测到处理好的数据: $PROCESSED_FILE"
    echo "  -> 跳过预处理，直接加载。"
else
    echo "  -> 未找到处理好的数据，开始预处理 ..."
    cd "$COMMONS"

    # 对于 MovieLens 数据集，检测并处理损坏的 zip 文件
    if [[ "$DATASET" == "ml-1m" ]]; then
        ZIP_FILE="$DATA_DIR/movielens1m.zip"
        if [ -f "$ZIP_FILE" ]; then
            if ! python3 -c "import zipfile; zipfile.ZipFile('$ZIP_FILE')" 2>/dev/null; then
                echo "  -> 检测到损坏的 zip 文件，重新下载 ..."
                rm -f "$ZIP_FILE"
            fi
        fi
    elif [[ "$DATASET" == "ml-20m" ]]; then
        ZIP_FILE="$DATA_DIR/movielens20m.zip"
        if [ -f "$ZIP_FILE" ]; then
            if ! python3 -c "import zipfile; zipfile.ZipFile('$ZIP_FILE')" 2>/dev/null; then
                echo "  -> 检测到损坏的 zip 文件，重新下载 ..."
                rm -f "$ZIP_FILE"
            fi
        fi
    fi

    python3 ./hstu_data_preprocessor.py \
        --dataset_name "$DATASET" \
        --dataset_path "$DATA_DIR"

    # 验证处理结果
    if [ ! -f "$PROCESSED_FILE" ]; then
        echo "  错误: 预处理后未找到 $PROCESSED_FILE"
        echo "  请检查预处理日志。"
        exit 1
    fi
    echo "  -> 数据集预处理完成。"
fi

# ── Step 2: 配置 gin 并启动训练 ──────────────────────────────────────────────
echo ""
echo "[2/2] 启动训练 ..."

# 动态配置 gin 文件中的 dataset_name 和 dataset_path
# 使用临时 gin 文件避免修改原始配置
RUNTIME_GIN=$(mktemp /tmp/hstu_runtime_XXXXXX.gin)
cp "$GIN_CONFIG" "$RUNTIME_GIN"

# 更新 dataset_name
if grep -q "^DatasetArgs.dataset_name" "$RUNTIME_GIN"; then
    sed -i "s|^DatasetArgs.dataset_name = .*|DatasetArgs.dataset_name = '$DATASET'|" "$RUNTIME_GIN"
else
    [ -n "$(tail -c1 "$RUNTIME_GIN")" ] && echo >> "$RUNTIME_GIN"
    echo "DatasetArgs.dataset_name = '$DATASET'" >> "$RUNTIME_GIN"
fi

# 设置 dataset_path (非默认路径时需要)
if [ "$DATA_DIR" != "$COMMONS/tmp_data" ]; then
    # 确保文件末尾有换行符
    [ -n "$(tail -c1 "$RUNTIME_GIN")" ] && echo >> "$RUNTIME_GIN"
    if grep -q "^DatasetArgs.dataset_path" "$RUNTIME_GIN"; then
        sed -i "s|^DatasetArgs.dataset_path = .*|DatasetArgs.dataset_path = '$DATA_DIR'|" "$RUNTIME_GIN"
    else
        echo "DatasetArgs.dataset_path = '$DATA_DIR'" >> "$RUNTIME_GIN"
    fi
fi

# 对于 ml-1m，确保有合理的 log_interval 和 eval_interval
if [[ "$DATASET" == "ml-1m" ]]; then
    if grep -q 'TrainerArgs.log_interval = 100' "$RUNTIME_GIN"; then
        sed -i 's/TrainerArgs.log_interval = 100/TrainerArgs.log_interval = 10/' "$RUNTIME_GIN"
    fi
    if grep -q 'TrainerArgs.eval_interval = 100' "$RUNTIME_GIN"; then
        sed -i 's/TrainerArgs.eval_interval = 100/TrainerArgs.eval_interval = 20/' "$RUNTIME_GIN"
    fi
fi

echo ""
echo "============================================"
echo "  数据准备完成！开始训练 ..."
echo "  数据集: $DATASET"
echo "  数据路径: $PROCESSED_FILE"
echo "============================================"
echo ""

cd "$HSTU_DIR"
# PYTHONPATH 需要同时包含三层目录:
#   - examples/            (用于 import commons.*)
#   - examples/hstu/       (用于 import configs, model, modules, utils)
#   - examples/hstu/training/ (用于 import trainer)
PYTHONPATH="${PYTHONPATH:-}:$(realpath ../):$(realpath .):$(realpath ./training)" \
    torchrun --nproc_per_node "$GPUS" --master_addr localhost --master_port 6000 \
    ./training/pretrain_gr_retrieval.py \
    --gin-config-file "$RUNTIME_GIN"

# 清理临时 gin 文件
rm -f "$RUNTIME_GIN"
