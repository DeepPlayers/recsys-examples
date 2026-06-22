#!/usr/bin/env bash
# ============================================================================
#  HSTU Retrieval Training — Environment Setup
#
#  用法:
#    cd <repo-root>/examples/hstu/training
#    bash setup.sh
#
#  或指定 repo root:
#    bash setup.sh /path/to/recsys-examples
#
#  功能:
#    1. 安装缺失的预装依赖 (gin-config, nvtx, torchx)
#    2. 安装 dynamicemb
#    3. 编译并安装 hstu_cuda_ops (commons CUDA 扩展)
#    4. 编译并安装 CUTLASS attention kernel (hstu_attn + hstu FBGEMM 接口层)
#    5. 自动应用 3 处代码补丁 (使 pytorch 后端无需 FBGEMM HSTU kernel)
#    6. 配置 gin (kernel_backend + log/eval interval)
#    7. 创建数据目录符号链接 & 按需预处理 MovieLens-1M 数据集
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

# ── Step 4: 编译安装 CUTLASS attention kernel ─────────────────────────────────
# 包含两个包:
#   - hstu_attn (核心库): 从 corelib/hstu 编译，提供 hstu_attn_2_cuda.varlen_fwd/bwd
#   - hstu (FBGEMM 接口层): 从 third_party/FBGEMM 编译，注册 torch.ops.fbgemm.hstu_varlen_fwd_80/bwd_80
echo ""
echo "[4/7] Building CUTLASS attention kernels ..."

# ── 4a: 编译 hstu_attn (核心库) ───────────────────────────────────────────────
echo ""
echo "  [4a] Building hstu_attn (core CUTLASS kernel) ..."
if python3 -c "import hstu_attn_2_cuda" 2>/dev/null; then
    echo "    -> hstu_attn_2_cuda already installed, skipping."
else
    cd "$REPO_ROOT/corelib/hstu"

    # 设置编译选项（禁用不需要的功能以加速编译）
    export HSTU_DISABLE_BACKWARD=FALSE
    export HSTU_DISABLE_DETERMINISTIC=TRUE
    export HSTU_DISABLE_LOCAL=FALSE
    export HSTU_DISABLE_CAUSAL=FALSE
    export HSTU_DISABLE_CONTEXT=FALSE
    export HSTU_DISABLE_TARGET=FALSE
    export HSTU_DISABLE_ARBITRARY=FALSE
    export HSTU_ARBITRARY_NFUNC=3
    export HSTU_DISABLE_RAB=FALSE
    export HSTU_DISABLE_DRAB=FALSE
    export HSTU_DISABLE_BF16=FALSE
    export HSTU_DISABLE_FP16=TRUE
    export HSTU_DISABLE_HDIM32=FALSE
    export HSTU_DISABLE_HDIM64=FALSE
    export HSTU_DISABLE_HDIM128=FALSE
    export HSTU_DISABLE_HDIM256=FALSE
    export HSTU_DISABLE_86OR89=TRUE

    echo "    -> Compiling hstu_attn (this may take 10-15 minutes) ..."
    make -j4

    # 验证安装
    if python3 -c "import hstu_attn_2_cuda; print('hstu_attn_2_cuda loaded successfully')" 2>/dev/null; then
        echo "    -> hstu_attn installed successfully."
    else
        echo "    ERROR: hstu_attn build failed! Check compile logs above."
        exit 1
    fi
fi

# ── 4b: 下载 FBGEMM hstu 源码 ─────────────────────────────────────────────────
echo ""
echo "  [4b] Downloading FBGEMM hstu source ..."
FBGEMM_HSTU_DIR="$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu"

if [ -d "$FBGEMM_HSTU_DIR/src/hstu_ampere" ]; then
    echo "    -> FBGEMM hstu source already present, skipping."
else
    mkdir -p "$FBGEMM_HSTU_DIR"

    python3 << 'EOF'
import urllib.request
import tarfile
import io
import os

url = "https://api.github.com/repos/jiayus-nvidia/FBGEMM/tarball/main"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
tar = tarfile.open(fileobj=io.BytesIO(response.read()))

REPO_ROOT = os.environ.get("REPO_ROOT", "/workspace/recsys-examples")
target_base = os.path.join(REPO_ROOT, "third_party/FBGEMM/fbgemm_gpu/experimental/hstu")

for member in tar.getmembers():
    if 'fbgemm_gpu/experimental/hstu' in member.name:
        rel_path = member.name.split('experimental/hstu/')[-1]
        if rel_path and not member.isdir():
            target_path = os.path.join(target_base, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with tar.extractfile(member) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())

print("Downloaded FBGEMM hstu source code")
EOF
    echo "    -> FBGEMM hstu source downloaded."
fi

# ── 4c: 生成 kernel 实例化代码 ────────────────────────────────────────────────
echo ""
echo "  [4c] Generating kernel instantiations ..."
if [ -d "$FBGEMM_HSTU_DIR/src/hstu_ampere/instantiations" ] && \
   [ "$(ls "$FBGEMM_HSTU_DIR/src/hstu_ampere/instantiations/"*.cu 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "    -> Kernel instantiations already generated, skipping."
else
    cd "$FBGEMM_HSTU_DIR/src"

    # 设置 kernel 变体选项（必须与 hstu_attn 一致）
    export HSTU_DISABLE_BACKWARD=FALSE
    export HSTU_DISABLE_DETERMINISTIC=TRUE
    export HSTU_DISABLE_BF16=FALSE
    export HSTU_DISABLE_FP16=TRUE
    export HSTU_DISABLE_HDIM32=FALSE
    export HSTU_DISABLE_HDIM64=FALSE
    export HSTU_DISABLE_HDIM128=FALSE
    export HSTU_DISABLE_HDIM256=FALSE
    export HSTU_DISABLE_LOCAL=FALSE
    export HSTU_DISABLE_CAUSAL=FALSE
    export HSTU_DISABLE_CONTEXT=FALSE
    export HSTU_DISABLE_TARGET=FALSE
    export HSTU_DISABLE_ARBITRARY=FALSE
    export HSTU_DISABLE_RAB=FALSE
    export HSTU_DISABLE_DRAB=FALSE

    python3 generate_kernels.py --arch-list "8.0"

    NUM_CU=$(ls hstu_ampere/instantiations/*.cu 2>/dev/null | wc -l)
    echo "    -> Generated $NUM_CU kernel instantiation files."
fi

# ── 4d: 创建 setup.py ─────────────────────────────────────────────────────────
echo ""
echo "  [4d] Creating setup.py for FBGEMM hstu ..."
if [ -f "$FBGEMM_HSTU_DIR/setup.py" ]; then
    echo "    -> setup.py already exists, skipping."
else
    cat > "$FBGEMM_HSTU_DIR/setup.py" << 'SETUPEOF'
"""
Standalone setup.py for FBGEMM hstu experimental package.
Compiles hstu_ops_gpu (Ampere SM 8.0) with PPU/nvcc.
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = Path(this_dir).resolve().parent.parent.parent.parent  # FBGEMM root

# CUTLASS headers from recsys-examples
cutlass_dir = repo_root.parent.parent.parent.parent / "third_party" / "cutlass"
if not cutlass_dir.exists():
    # Fallback: try relative to recsys-examples
    cutlass_dir = Path("/wang/recsys-examples/third_party/cutlass")

HSTU_ARCH_LIST = os.getenv("HSTU_ARCH_LIST", "8.0").split()

# Collect sources for Ampere (SM 8.0)
ampere_src = os.path.join(this_dir, "src", "hstu_ampere")
hopper_src = os.path.join(this_dir, "src", "hstu_hopper")

sources = []
include_dirs = []

if "8.0" in HSTU_ARCH_LIST:
    sources.append(os.path.join(ampere_src, "hstu_ops_gpu.cpp"))
    sources.extend(glob.glob(os.path.join(ampere_src, "instantiations", "*.cu")))
    include_dirs.append(ampere_src)

if "9.0" in HSTU_ARCH_LIST or "9.0a" in HSTU_ARCH_LIST:
    sources.append(os.path.join(hopper_src, "hstu_ops_gpu.cpp"))
    sources.extend(glob.glob(os.path.join(hopper_src, "instantiations", "*.cu")))
    include_dirs.append(hopper_src)

include_dirs.append(str(cutlass_dir / "include"))
include_dirs.append(str(cutlass_dir / "tools" / "util" / "include"))

# Build arch flags
cc_flags = []
for arch in HSTU_ARCH_LIST:
    if arch == "8.0":
        cc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
    elif arch in ("9.0", "9.0a"):
        cc_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-lineinfo",
] + cc_flags

nvcc_threads = os.getenv("NVCC_THREADS", "4")
nvcc_flags += ["--threads", nvcc_threads]

# HSTU feature flags (match Dockerfile: disable arbitrary/local/rab/drab/fp16)
feature_flags = [
    "-DHSTU_DISABLE_ARBITRARY",
    "-DHSTU_DISABLE_LOCAL",
    "-DHSTU_DISABLE_RAB",
    "-DHSTU_DISABLE_DRAB",
    "-DHSTU_DISABLE_FP16",
    "-DHSTU_DISABLE_DETERMINISTIC",
    "-DHSTU_DISABLE_86OR89",
    "-DHSTU_ARBITRARY_NFUNC=1",
]

cxx_flags = ["-O3", "-std=c++17"] + feature_flags
nvcc_flags += feature_flags

print(f"Building hstu extension:")
print(f"  Sources: {len(sources)} files")
print(f"  CUTLASS: {cutlass_dir}")
print(f"  Archs:   {HSTU_ARCH_LIST}")

ext_modules = [
    CUDAExtension(
        name="hstu.hstu_ops_gpu",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )
]

setup(
    name="fbgemm_gpu_hstu",
    version="0.1.0",
    packages=["hstu"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.7",
    install_requires=["torch"],
)
SETUPEOF
    echo "    -> setup.py created."
fi

# ── 4e: 修改 library.py 加载逻辑 ──────────────────────────────────────────────
echo ""
echo "  [4e] Updating library.py load logic ..."
cat > "$FBGEMM_HSTU_DIR/hstu/library.py" << 'LIBEOF'
#!/usr/bin/env python3
"""HSTU library initialization - loads CUDA ops from compiled .so"""
import glob
import logging
import os

import torch

try:
    from fbgemm_gpu import open_source
except Exception:
    open_source: bool = False


def _load_hstu_ops() -> None:
    """Load HSTU CUDA ops library."""
    pkg_dir = os.path.dirname(__file__)

    # Try loading hstu_ops_gpu.so (standalone build via setup.py)
    so_files = glob.glob(os.path.join(pkg_dir, "hstu_ops_gpu*.so"))
    if so_files:
        torch.ops.load_library(so_files[0])
        return

    # Fallback: try fbgemm_gpu_experimental_hstu.so (CMake build)
    fbgemm_so = os.path.join(pkg_dir, "fbgemm_gpu_experimental_hstu.so")
    if os.path.exists(fbgemm_so):
        torch.ops.load_library(fbgemm_so)
        torch.classes.load_library(fbgemm_so)
        return

    # Fallback: internal build paths
    try:
        import fbgemm_gpu  # noqa: F401
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_gpu")
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm80"
        )
        if torch.cuda.get_device_capability() >= (9, 0):
            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu/experimental/hstu/src:hstu_ops_gpu_sm90"
            )
    except Exception:
        logging.warning("Could not load HSTU ops library")


if torch.cuda.is_available():
    try:
        _load_hstu_ops()
    except Exception as e:
        logging.warning(f"Failed to load HSTU CUDA ops: {e}")
else:
    logging.warning("CUDA is not available for FBGEMM HSTU")
LIBEOF
echo "    -> library.py updated."

# ── 4f: 编译安装 hstu 包 ──────────────────────────────────────────────────────
echo ""
echo "  [4f] Building and installing hstu (FBGEMM interface) ..."
if python3 -c "
import torch
import hstu
fwd_op = torch.ops.fbgemm.hstu_varlen_fwd_80
" 2>/dev/null; then
    echo "    -> hstu (FBGEMM) already installed, skipping."
else
    cd "$FBGEMM_HSTU_DIR"

    export HSTU_ARCH_LIST="8.0"
    export MAX_JOBS="${MAX_JOBS:-39}"
    export NVCC_THREADS="${NVCC_THREADS:-1}"

    echo "    -> Compiling hstu FBGEMM interface (this may take 5-10 minutes) ..."
    pip install --no-build-isolation -e .

    # 验证安装
    if python3 -c "
import torch
import hstu
fwd_op = torch.ops.fbgemm.hstu_varlen_fwd_80
bwd_op = torch.ops.fbgemm.hstu_varlen_fwd_80
print('hstu FBGEMM interface loaded successfully')
" 2>/dev/null; then
        echo "    -> hstu (FBGEMM) installed successfully."
    else
        echo "    WARNING: hstu (FBGEMM) installation may have issues. Check logs above."
    fi
fi

# ── Step 5: 应用代码补丁 ──────────────────────────────────────────────────────
echo ""
echo "[5/7] Applying code patches ..."

# 补丁 5a: hstu_attention.py — 顶层 hstu 导入改为懒加载
PATCH_FILE="$HSTU_DIR/modules/hstu_attention.py"
if grep -q "^from hstu import hstu_attn_varlen_func" "$PATCH_FILE" 2>/dev/null; then
    echo "  -> Patching hstu_attention.py (lazy import) ..."
    sed -i '/^from hstu import hstu_attn_varlen_func$/d' "$PATCH_FILE"
    sed -i '/^        return hstu_attn_varlen_func(/i\        from hstu import hstu_attn_varlen_func\n' "$PATCH_FILE"
    echo "     Done."
else
    echo "  -> hstu_attention.py already patched or not applicable, skipping."
fi

# 补丁 5b: fused_hstu_op.py — hstu/hstu.hstu_ops_gpu 导入 try/except 包裹
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

# 补丁 5c: trainer/utils.py — pytorch 后端使用 DEBUG layer type
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

# ── Step 6: 配置 kernel_backend & log/eval interval ──────────────────────────
echo ""
echo "[6/7] Checking gin config ..."
GIN_FILE="$HSTU_DIR/training/configs/movielen_retrieval.gin"
if grep -q 'kernel_backend' "$GIN_FILE" 2>/dev/null; then
    echo "  -> kernel_backend already set in gin config, skipping."
else
    echo "  -> Adding kernel_backend = pytorch to gin config ..."
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

# ── Step 7: 数据准备 ──────────────────────────────────────────────────────────
echo ""
echo "[7/7] Preparing data ..."

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

echo ""
echo "============================================"
echo "  Setup complete! Run train.sh to start training."
echo "============================================"
