#!/bin/bash
# ============================================================================
# HSTU CUTLASS Kernel 编译脚本 (PPU 环境)
#
# 本脚本在 PPU-ZW810E (SM 8.0) 环境下编译两个 CUTLASS attention kernel 包：
# 1. hstu_attn (核心库) - 从 corelib/hstu 编译
# 2. hstu (FBGEMM 接口) - 从 third_party/FBGEMM/fbgemm_gpu/experimental/hstu 编译
#
# 用法:
#   ./build_cutlass_kernels.sh [选项]
#
# 选项:
#   --max-jobs=N        并行编译任务数 (默认: 39)
#   --nvcc-threads=N    每个 nvcc 进程使用的线程数 (默认: 1)
#   --arch=ARCH         目标架构 (默认: 8.0, 可选: 8.0, 9.0, 8.0,9.0)
#   --skip-hstu-attn    跳过 hstu_attn 编译
#   --skip-fbgemm-hstu  跳过 fbgemm_gpu_hstu 编译
#   --clean             清理编译产物
#   --verify-only       仅验证已安装的包
#   -h, --help          显示帮助信息
#
# 示例:
#   # 完整编译（推荐）
#   ./build_cutlass_kernels.sh
#
#   # 使用 20 个并行任务
#   ./build_cutlass_kernels.sh --max-jobs=20
#
#   # 仅编译 FBGEMM hstu
#   ./build_cutlass_kernels.sh --skip-hstu-attn
#
#   # 清理编译产物
#   ./build_cutlass_kernels.sh --clean
# ============================================================================

set -e
set -o pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
MAX_JOBS=39
NVCC_THREADS=1
TARGET_ARCH="8.0"
SKIP_HSTU_ATTN=0
SKIP_FBGEMM_HSTU=0
CLEAN_ONLY=0
VERIFY_ONLY=0

# 获取脚本所在目录
# 脚本位置: examples/hstu/training/benchmark/scripts/
# 需要向上 5 级到仓库根目录:
#   scripts -> benchmark -> training -> hstu -> examples -> repo_root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# ============================================================================
# 函数定义
# ============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${YELLOW}[步骤 $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

show_help() {
    head -30 "$0" | grep '^#' | sed 's/^# \{0,1\}//'
}

check_prerequisites() {
    print_step "1" "检查前置条件..."

    # 检查 PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        print_error "PyTorch 未安装"
        exit 1
    fi
    print_success "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

    # 检查 CUDA
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        print_error "CUDA 不可用"
        exit 1
    fi

    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    CUDA_CAP=$(python -c "import torch; print(torch.cuda.get_device_capability(0))")
    print_success "GPU: $GPU_NAME (Capability: $CUDA_CAP)"

    # 检查 CUTLASS
    if [ ! -d "$REPO_ROOT/third_party/cutlass" ]; then
        print_error "CUTLASS 源码不存在: $REPO_ROOT/third_party/cutlass"
        print_info "请运行: git submodule update --init third_party/cutlass"
        exit 1
    fi
    print_success "CUTLASS: $REPO_ROOT/third_party/cutlass"

    # 检查 corelib/hstu
    if [ ! -f "$REPO_ROOT/corelib/hstu/setup.py" ]; then
        print_error "corelib/hstu/setup.py 不存在"
        exit 1
    fi
    print_success "corelib/hstu: $REPO_ROOT/corelib/hstu"

    echo ""
}

clean_build_artifacts() {
    print_header "清理编译产物"

    # 清理 hstu_attn
    if [ -d "$REPO_ROOT/corelib/hstu/build" ]; then
        rm -rf "$REPO_ROOT/corelib/hstu/build"
        print_success "已清理 corelib/hstu/build"
    fi

    # 清理 fbgemm hstu
    if [ -d "$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/build" ]; then
        rm -rf "$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/build"
        print_success "已清理 fbgemm hstu build"
    fi

    if [ -d "$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/hstu_ampere/instantiations" ]; then
        rm -rf "$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/hstu_ampere/instantiations"
        print_success "已清理 kernel instantiations"
    fi

    echo ""
}

build_hstu_attn() {
    print_header "编译 hstu_attn (核心库)"

    cd "$REPO_ROOT/corelib/hstu"

    print_info "设置编译选项..."
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

    print_info "开始编译 (MAX_JOBS=$MAX_JOBS)..."
    MAX_JOBS=$MAX_JOBS python setup.py install

    print_info "验证安装..."
    if python -c "import hstu_attn_2_cuda" 2>/dev/null; then
        print_success "hstu_attn_2_cuda 加载成功"
    else
        print_error "hstu_attn_2_cuda 加载失败"
        exit 1
    fi

    echo ""
}

download_fbgemm_hstu() {
    print_header "下载 FBGEMM hstu 源码"

    TARGET_DIR="$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu"

    if [ -d "$TARGET_DIR/hstu" ] && [ -f "$TARGET_DIR/hstu/__init__.py" ]; then
        print_info "FBGEMM hstu 源码已存在，跳过下载"
        return
    fi

    mkdir -p "$TARGET_DIR"
    cd "$TARGET_DIR"

    print_info "从 GitHub API 下载 jiayus-nvidia/FBGEMM fork..."

    python3 << 'EOF'
import urllib.request
import tarfile
import io
import os

url = "https://api.github.com/repos/jiayus-nvidia/FBGEMM/tarball/main"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
tar = tarfile.open(fileobj=io.BytesIO(response.read()))

count = 0
for member in tar.getmembers():
    if 'fbgemm_gpu/experimental/hstu' in member.name:
        rel_path = member.name.split('experimental/hstu/')[-1]
        if rel_path and not member.isdir():
            target_path = rel_path
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with tar.extractfile(member) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
            count += 1

print(f"下载了 {count} 个文件")
EOF

    print_success "FBGEMM hstu 源码下载完成"
    echo ""
}

generate_kernels() {
    print_header "生成 kernel 实例化代码"

    cd "$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/src"

    print_info "设置 kernel 变体选项..."
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

    print_info "生成 SM $TARGET_ARCH kernel 实例化文件..."
    python generate_kernels.py --arch-list "$TARGET_ARCH"

    KERNEL_COUNT=$(ls hstu_ampere/instantiations/*.cu 2>/dev/null | wc -l)
    if [ "$KERNEL_COUNT" -lt 70 ]; then
        print_error "kernel 实例化文件数量不足: $KERNEL_COUNT (预期 ~80)"
        exit 1
    fi

    print_success "生成了 $KERNEL_COUNT 个 kernel 实例化文件"
    echo ""
}

create_setup_py() {
    print_header "创建 setup.py"

    SETUP_PY="$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/setup.py"

    if [ -f "$SETUP_PY" ]; then
        print_info "setup.py 已存在，跳过创建"
        return
    fi

    print_info "创建独立编译脚本..."
    cat > "$SETUP_PY" << 'EOF'
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
repo_root = Path(this_dir).resolve().parent.parent.parent.parent

cutlass_dir = repo_root.parent.parent.parent / "third_party" / "cutlass"
if not cutlass_dir.exists():
    cutlass_dir = Path("/workspace/recsys-examples/third_party/cutlass")

HSTU_ARCH_LIST = os.getenv("HSTU_ARCH_LIST", "8.0").split()

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
EOF

    print_success "setup.py 创建完成"
    echo ""
}

modify_library_py() {
    print_header "修改 library.py 加载逻辑"

    LIBRARY_PY="$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu/hstu/library.py"

    print_info "更新 .so 加载逻辑..."
    cat > "$LIBRARY_PY" << 'EOF'
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
EOF

    print_success "library.py 更新完成"
    echo ""
}

build_fbgemm_hstu() {
    print_header "编译 fbgemm_gpu_hstu"

    cd "$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu"

    print_info "设置编译选项..."
    export HSTU_ARCH_LIST="$TARGET_ARCH"
    export MAX_JOBS
    export NVCC_THREADS

    print_info "开始编译 (MAX_JOBS=$MAX_JOBS, NVCC_THREADS=$NVCC_THREADS)..."
    pip install --no-build-isolation -e .

    print_info "验证安装..."
    if python -c "import hstu; import torch; torch.ops.fbgemm.hstu_varlen_fwd_80" 2>/dev/null; then
        print_success "hstu 包加载成功"
        print_success "torch.ops.fbgemm.hstu_varlen_fwd_80 已注册"
    else
        print_error "hstu 包加载失败"
        exit 1
    fi

    echo ""
}

verify_installation() {
    print_header "验证完整环境"

    python << 'EOF'
import torch

print("=" * 70)
print("HSTU CUTLASS Kernel 环境验证")
print("=" * 70)

# 检查 hstu_attn
try:
    import hstu_attn_2_cuda
    print("✓ hstu_attn_2_cuda loaded")
except ImportError as e:
    print(f"✗ hstu_attn_2_cuda not found: {e}")
    exit(1)

# 检查 hstu (FBGEMM interface)
try:
    import hstu
    fwd_op = torch.ops.fbgemm.hstu_varlen_fwd_80
    bwd_op = torch.ops.fbgemm.hstu_varlen_bwd_80
    print("✓ hstu package loaded")
    print("✓ torch.ops.fbgemm.hstu_varlen_fwd_80 available")
    print("✓ torch.ops.fbgemm.hstu_varlen_bwd_80 available")
except Exception as e:
    print(f"✗ hstu package error: {e}")
    exit(1)

# 检查 GPU
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
print("=" * 70)
EOF

    if [ $? -eq 0 ]; then
        print_success "所有组件验证通过！"
    else
        print_error "验证失败，请检查错误信息"
        exit 1
    fi

    echo ""
}

# ============================================================================
# 参数解析
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs=*)
            MAX_JOBS="${1#*=}"
            shift
            ;;
        --nvcc-threads=*)
            NVCC_THREADS="${1#*=}"
            shift
            ;;
        --arch=*)
            TARGET_ARCH="${1#*=}"
            shift
            ;;
        --skip-hstu-attn)
            SKIP_HSTU_ATTN=1
            shift
            ;;
        --skip-fbgemm-hstu)
            SKIP_FBGEMM_HSTU=1
            shift
            ;;
        --clean)
            CLEAN_ONLY=1
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# 主流程
# ============================================================================

print_header "HSTU CUTLASS Kernel 编译脚本 (PPU 环境)"

print_info "参数配置:"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  NVCC_THREADS=$NVCC_THREADS"
echo "  TARGET_ARCH=$TARGET_ARCH"
echo "  SKIP_HSTU_ATTN=$SKIP_HSTU_ATTN"
echo "  SKIP_FBGEMM_HSTU=$SKIP_FBGEMM_HSTU"
echo ""

if [ $CLEAN_ONLY -eq 1 ]; then
    clean_build_artifacts
    exit 0
fi

if [ $VERIFY_ONLY -eq 1 ]; then
    verify_installation
    exit 0
fi

check_prerequisites

if [ $SKIP_HSTU_ATTN -eq 0 ]; then
    build_hstu_attn
fi

if [ $SKIP_FBGEMM_HSTU -eq 0 ]; then
    download_fbgemm_hstu
    generate_kernels
    create_setup_py
    modify_library_py
    build_fbgemm_hstu
fi

verify_installation

print_header "编译完成！"
print_info "现在可以运行 HSTU E2E benchmark:"
echo "  cd $REPO_ROOT/examples/hstu"
echo "  ./training/benchmark/scripts/run_all_experiments_local.sh \\"
echo "      --exp-file=training/benchmark/experiments.txt \\"
echo "      --nproc=4"
echo ""
