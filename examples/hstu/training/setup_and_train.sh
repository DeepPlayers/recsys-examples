#!/usr/bin/env bash
# ============================================================================
#  HSTU Retrieval Training — One-Click Setup & Launch
#
#  用法:
#    cd <repo-root>/examples/hstu/training
#    bash setup_and_train.sh
#
#  或指定 repo root:
#    bash setup_and_train.sh /path/to/recsys-examples
#
#  功能:
#    1. 安装缺失的预装依赖 (gin-config, nvtx, torchx)
#    2. 安装 dynamicemb
#    3. 编译并安装 hstu_cuda_ops (commons CUDA 扩展)
#    4. 自动应用 3 处代码补丁 (使 pytorch 后端无需 FBGEMM HSTU kernel)
#    5. 配置 gin (kernel_backend + log/eval interval)
#    6. 创建数据目录符号链接 & 按需预处理 MovieLens-1M 数据集
#    7. 启动 training
# ============================================================================
set -euo pipefail

# ── 路径 ──────────────────────────────────────────────────────────────────────
REPO_ROOT="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

COMMONS="$REPO_ROOT/examples/commons"
HSTU_DIR="$REPO_ROOT/examples/hstu"
DYNAMICEMB_DIR="$REPO_ROOT/corelib/dynamicemb"

echo "============================================"
echo "  HSTU Training Setup"
echo "  Repo root: $REPO_ROOT"
echo "============================================"

# ── Step 1: 安装缺失预装依赖 ─────────────────────────────────────────────────
# 部分 PPU 环境中 gin-config, nvtx, torchx 未预装
# megatron-core 通常已预装，若 pip install 失败可跳过
echo ""
echo "[1/7] Installing prerequisite packages ..."
pip install gin-config nvtx torchx 2>/dev/null | tail -3
if python3 -c "import megatron.core" 2>/dev/null; then
    echo "  -> megatron-core already available."
else
    echo "  -> WARNING: megatron-core not found. Please install it for your environment."
fi
echo "  -> Prerequisites done."

# ── Step 2: 安装 dynamicemb ──────────────────────────────────────────────────
# corelib/dynamicemb/setup.py 在文件顶层导入了 torch:
#   from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# pip 默认的 build isolation 会创建隔离环境，其中没有 torch，导致:
#   ModuleNotFoundError: No module named 'torch'
# 解决方案: --no-build-isolation，让 pip 使用当前环境中已安装的 torch
echo ""
echo "[2/7] Installing dynamicemb ..."
if python3 -c "import dynamicemb" 2>/dev/null; then
    echo "  -> dynamicemb already installed, skipping."
else
    cd "$DYNAMICEMB_DIR"
    pip install . --no-build-isolation
    echo "  -> dynamicemb installed successfully."
fi

# ── Step 3: 编译安装 hstu_cuda_ops ────────────────────────────────────────────
echo ""
echo "[3/7] Building hstu_cuda_ops ..."
if python3 -c "import hstu_cuda_ops" 2>/dev/null; then
    echo "  -> hstu_cuda_ops already installed, skipping."
else
    cd "$COMMONS"

    # 只编译 hstu_cuda_ops, 跳过 paged_kvcache_ops (需要 nvcomp_static)
    # 通过临时 setup.py 仅构建 hstu_cuda_ops
    TEMP_SETUP=$(mktemp /tmp/setup_hstu_ops_XXXXXX.py)
    cat > "$TEMP_SETUP" << 'PYEOF'
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]

nvcc_flags = [
    "-g", "-O3", "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]

setup(
    name="hstu_cuda_ops",
    description="HSTU CUDA ops",
    ext_modules=[
        CUDAExtension(
            name="hstu_cuda_ops",
            sources=[
                "ops/cuda_ops/csrc/jagged_tensor_op_cuda.cpp",
                "ops/cuda_ops/csrc/jagged_tensor_op_kernel.cu",
                "ops/cuda_ops/csrc/kjt_aux_op.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-DWITH_PYBIND11=1"],
                "nvcc": nvcc_threads_args() + nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
PYEOF

    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 9.0}" \
        python3 "$TEMP_SETUP" build_ext --inplace 2>&1 | tail -5

    # 找到编译产物并复制到 site-packages
    SO_FILE=$(find "$COMMONS" -name "hstu_cuda_ops*.so" -newer "$TEMP_SETUP" 2>/dev/null | head -1)
    if [ -z "$SO_FILE" ]; then
        # fallback: 查找 build 目录
        SO_FILE=$(find "$COMMONS/build" -name "hstu_cuda_ops*.so" 2>/dev/null | head -1)
    fi

    if [ -z "$SO_FILE" ]; then
        echo "  ERROR: hstu_cuda_ops build failed! Check compile logs above."
        rm -f "$TEMP_SETUP"
        exit 1
    fi

    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    cp "$SO_FILE" "$SITE_PACKAGES/"
    echo "  -> hstu_cuda_ops installed: $SO_FILE -> $SITE_PACKAGES/"
    rm -f "$TEMP_SETUP"
fi

# ── Step 4: 应用代码补丁 ──────────────────────────────────────────────────────
echo ""
echo "[4/7] Applying code patches ..."

# 补丁 3a: hstu_attention.py — 顶层 hstu 导入改为懒加载
PATCH_FILE="$HSTU_DIR/modules/hstu_attention.py"
if grep -q "^from hstu import hstu_attn_varlen_func" "$PATCH_FILE" 2>/dev/null; then
    echo "  -> Patching hstu_attention.py (lazy import) ..."
    # 移除顶层导入
    sed -i '/^from hstu import hstu_attn_varlen_func$/d' "$PATCH_FILE"
    # 在 FusedHSTUAttention.forward() 中 hstu_attn_varlen_func 调用前插入懒加载导入
    sed -i '/^        return hstu_attn_varlen_func(/i\        from hstu import hstu_attn_varlen_func\n' "$PATCH_FILE"
    echo "     Done."
else
    echo "  -> hstu_attention.py already patched or not applicable, skipping."
fi

# 补丁 3b: fused_hstu_op.py — hstu/hstu.hstu_ops_gpu 导入 try/except 包裹
PATCH_FILE="$HSTU_DIR/ops/fused_hstu_op.py"
if grep -q "^import hstu " "$PATCH_FILE" 2>/dev/null; then
    echo "  -> Patching fused_hstu_op.py (try/except import) ..."
    python3 -c "
import re, pathlib
p = pathlib.Path('$PATCH_FILE')
text = p.read_text()
old = 'import hstu  # noqa: F401 – registers torch.ops.fbgemm.*\nimport hstu.hstu_ops_gpu  # noqa: F401 – registers fake impls for torch.export'
new = '''try:
    import hstu  # noqa: F401 – registers torch.ops.fbgemm.*
    import hstu.hstu_ops_gpu  # noqa: F401 – registers fake impls for torch.export
except ImportError:
    pass'''
text = text.replace(old, new)
p.write_text(text)
"
    echo "     Done."
else
    echo "  -> fused_hstu_op.py already patched or not applicable, skipping."
fi

# 补丁 3c: trainer/utils.py — pytorch 后端使用 DEBUG layer type
PATCH_FILE="$HSTU_DIR/training/trainer/utils.py"
if grep -q "if kernel_backend == KernelBackend.PYTORCH:" "$PATCH_FILE" 2>/dev/null; then
    echo "  -> trainer/utils.py already patched, skipping."
else
    echo "  -> Patching trainer/utils.py (PYTORCH -> DEBUG layer type) ..."
    python3 -c "
import pathlib
p = pathlib.Path('$PATCH_FILE')
text = p.read_text()
old = '''    layer_type = None
    if tensor_model_parallel_args.tensor_model_parallel_size == 1:
        layer_type = HSTULayerType.FUSED'''
new = '''    layer_type = None
    if kernel_backend == KernelBackend.PYTORCH:
        layer_type = HSTULayerType.DEBUG
    elif tensor_model_parallel_args.tensor_model_parallel_size == 1:
        layer_type = HSTULayerType.FUSED'''
text = text.replace(old, new)
p.write_text(text)
"
    echo "     Done."
fi

# ── Step 5: 配置 kernel_backend & log/eval interval ──────────────────────────
echo ""
echo "[5/7] Checking gin config ..."
GIN_FILE="$HSTU_DIR/training/configs/movielen_retrieval.gin"
if grep -q 'kernel_backend' "$GIN_FILE" 2>/dev/null; then
    echo "  -> kernel_backend already set in gin config, skipping."
else
    echo "  -> Adding kernel_backend = pytorch to gin config ..."
    # 在 NetworkArgs.is_causal 行之后添加
    sed -i '/^NetworkArgs.is_causal/a NetworkArgs.kernel_backend = "pytorch"' "$GIN_FILE"
    echo "     Done."
fi

# 降低 log_interval 和 eval_interval (ml-1m 仅 ~47 steps/epoch, 默认 100 导致无输出)
if grep -q 'TrainerArgs.log_interval = 100' "$GIN_FILE" 2>/dev/null; then
    echo "  -> Adjusting log_interval from 100 to 10 ..."
    sed -i 's/TrainerArgs.log_interval = 100/TrainerArgs.log_interval = 10/' "$GIN_FILE"
    echo "     Done."
fi
if grep -q 'TrainerArgs.eval_interval = 100' "$GIN_FILE" 2>/dev/null; then
    echo "  -> Adjusting eval_interval from 100 to 20 ..."
    sed -i 's/TrainerArgs.eval_interval = 100/TrainerArgs.eval_interval = 20/' "$GIN_FILE"
    echo "     Done."
fi

# ── Step 6: 数据准备 ──────────────────────────────────────────────────────────
echo ""
echo "[6/7] Preparing data ..."

# 创建符号链接
if [ -L "$HSTU_DIR/tmp_data" ] || [ -d "$HSTU_DIR/tmp_data" ]; then
    echo "  -> tmp_data symlink/directory already exists, skipping."
else
    ln -sf "$COMMONS/tmp_data" "$HSTU_DIR/tmp_data"
    echo "  -> Created symlink: $HSTU_DIR/tmp_data -> $COMMONS/tmp_data"
fi

# 检查数据是否已预处理
if [ -f "$COMMONS/tmp_data/ml-1m/processed_seqs.csv" ]; then
    echo "  -> MovieLens-1M dataset already preprocessed."
else
    echo "  -> Preprocessing MovieLens-1M dataset ..."
    cd "$COMMONS"
    mkdir -p ./tmp_data

    # 检测并修复损坏的 zip 文件 (urlretrieve 可能产生不完整文件)
    ZIP_FILE="./tmp_data/movielens1m.zip"
    if [ -f "$ZIP_FILE" ]; then
        if ! python3 -c "import zipfile; zipfile.ZipFile('$ZIP_FILE')" 2>/dev/null; then
            echo "  -> Corrupted zip detected, re-downloading ..."
            rm -f "$ZIP_FILE"
            curl -L -o "$ZIP_FILE" "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
            echo "  -> Re-downloaded."
        fi
    fi

    python3 ./hstu_data_preprocessor.py --dataset_name ml-1m
    echo "  -> Dataset preprocessed."
fi

# ── Step 7: 启动训练 ──────────────────────────────────────────────────────────
echo ""
echo "[7/7] Launching training ..."
echo ""
echo "============================================"
echo "  All setup complete! Starting training ..."
echo "============================================"
echo ""

cd "$HSTU_DIR"
PYTHONPATH="${PYTHONPATH:-}:$(realpath ../)" \
    torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
    ./training/pretrain_gr_retrieval.py \
    --gin-config-file ./training/configs/movielen_retrieval.gin
