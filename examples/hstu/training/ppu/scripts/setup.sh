#!/usr/bin/env bash
# ============================================================================
#  HSTU PPU 环境一键部署脚本
#
#  功能：从零完成 PPU Pod 环境初始化 + HSTU 依赖编译，幂等执行
#
#  用法：
#    bash setup.sh [选项]
#
#  选项：
#    --repo-root PATH    仓库根目录（默认自动检测）
#    --skip-pod          跳过 Pod 系统环境（apt/pip/C++库），仅编译 HSTU
#    --skip-pip          跳过 pip 包安装
#    --max-jobs N        编译并行度（默认 39）
#    -h, --help          显示帮助
#
#  示例：
#    bash setup.sh                           # 全量部署
#    bash setup.sh --skip-pod                # 仅编译 HSTU 内核
#    bash setup.sh --skip-pod --skip-pip     # 最小化编译
# ============================================================================
set -euo pipefail

# ── 参数解析 ──────────────────────────────────────────────────────────────────
SKIP_POD=false
SKIP_PIP=false
MAX_JOBS=39
REPO_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-pod)   SKIP_POD=true; shift ;;
        --skip-pip)   SKIP_PIP=true; shift ;;
        --max-jobs)   MAX_JOBS="$2"; shift 2 ;;
        --repo-root)  REPO_ROOT="$2"; shift 2 ;;
        -h|--help)    head -25 "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "$REPO_ROOT" ]; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

COMMONS="$REPO_ROOT/examples/commons"
HSTU_DIR="$REPO_ROOT/examples/hstu"
DYNAMICEMB_DIR="$REPO_ROOT/corelib/dynamicemb"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/examples/hstu/training/ppu/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup_${TIMESTAMP}.log"

echo "============================================" | tee "$LOG_FILE"
echo "  HSTU PPU Environment Setup"               | tee -a "$LOG_FILE"
echo "  Repo root:  $REPO_ROOT"                 | tee -a "$LOG_FILE"
echo "  Max jobs:   $MAX_JOBS"                  | tee -a "$LOG_FILE"
echo "  Skip pod:   $SKIP_POD"                  | tee -a "$LOG_FILE"
echo "  Skip pip:   $SKIP_PIP"                  | tee -a "$LOG_FILE"
echo "  Log file:   $LOG_FILE"                  | tee -a "$LOG_FILE"
echo "  Started:    $(date)"                    | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"


# ==============================================================================
#  Phase 1: Pod 系统环境（apt + pip + C++ 库）
# ==============================================================================
if [ "$SKIP_POD" = false ]; then

    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "  Phase 1: Pod System Environment"       | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # ── 1.1 系统初始化 ────────────────────────────────────────────────────────
    echo "" | tee -a "$LOG_FILE"
    echo "[1.1] System initialization ..." | tee -a "$LOG_FILE"
    unset PYTHONPATH

    wget -q https://dw-mofang.oss-cn-shanghai.aliyuncs.com/kube-image/ppu/sources.list \
        -O /etc/apt/sources.list
    rm -f /etc/apt/sources.list.d/ubuntu.sources
    apt-get update 2>&1 | tail -3 | tee -a "$LOG_FILE"
    wget -q https://dw-mofang.oss-cn-shanghai.aliyuncs.com/kube-image/ppu/profile \
        -O /etc/profile
    echo "  -> System init done." | tee -a "$LOG_FILE"

    # ── 1.2 APT 系统依赖 ──────────────────────────────────────────────────────
    echo "" | tee -a "$LOG_FILE"
    echo "[1.2] Installing APT dependencies ..." | tee -a "$LOG_FILE"
    apt-get install -y --no-install-recommends \
        build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
        libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev \
        libyaml-cpp-dev libaio-dev libgmp10 libgnutls28-dev \
        libboost-fiber-dev libboost-context-dev pybind11-dev \
        libtbb-dev libgoogle-glog-dev libgtest-dev tree \
        2>&1 | tail -3 | tee -a "$LOG_FILE"
    echo "  -> APT dependencies done." | tee -a "$LOG_FILE"

    # ── 1.3 Python 包安装 ─────────────────────────────────────────────────────
    if [ "$SKIP_PIP" = false ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "[1.3] Installing Python packages ..." | tee -a "$LOG_FILE"
        MIRROR="-i https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu129/simple/"
        FLAGS="--no-cache-dir --no-build-isolation"

        # PyTorch 生态
        pip install $FLAGS $MIRROR fbgemm-gpu==1.4.0 2>&1 | tail -2 | tee -a "$LOG_FILE"
        pip install $FLAGS $MIRROR torchmetrics==1.0.3 2>&1 | tail -2 | tee -a "$LOG_FILE"
        pip install $FLAGS $MIRROR tensordict torchrec 2>&1 | tail -2 | tee -a "$LOG_FILE"

        # 大模型训练框架
        if pip show megatron-core 2>/dev/null | grep -q "Version: 0.15.0"; then
            echo "  -> megatron-core 0.15.0 already installed." | tee -a "$LOG_FILE"
        else
            if [ ! -d "Megatron-LM" ]; then
                git clone -b core_v0.15.0 https://github.com/NVIDIA/Megatron-LM.git 2>&1 | tee -a "$LOG_FILE"
            fi
            cd Megatron-LM && pip install --no-build-isolation . 2>&1 | tail -2 | tee -a "$LOG_FILE" && cd ..
        fi

        # NLP / 推理 / 监控 / 数值 / 数据 / 工具
        pip install $FLAGS $MIRROR lion_pytorch onnx sentencepiece \
            "optimum[onnxruntime]" torch-tb-profiler thop gin-config==0.5.0 \
            torchviz "numpy<2.0" scikit-learn pyarrow oss2 pymysql \
            pyodps==0.12.0 zstandard seaborn gputil pypinyin liger-kernel==0.7.0 \
            fastapi uvicorn taskflow pyyaml pytest loguru psutil certifi \
            wheel cmake yapf clang-format pybind11 jupyterlab \
            2>&1 | tail -3 | tee -a "$LOG_FILE"
        echo "  -> Python packages done." | tee -a "$LOG_FILE"
    fi

    # ── 1.4 C++ 库编译 ────────────────────────────────────────────────────────
    echo "" | tee -a "$LOG_FILE"
    echo "[1.4] Building C++ libraries ..." | tee -a "$LOG_FILE"

    # FlatBuffers v25.2.10
    if [ -d "/opt/flatbuffers-25/include" ]; then
        echo "  -> FlatBuffers already installed, skipping." | tee -a "$LOG_FILE"
    else
        cd /tmp
        git clone https://github.com/google/flatbuffers.git 2>&1 | tee -a "$LOG_FILE"
        cd flatbuffers && git checkout v25.2.10
        cmake -S . -B build -DFLATBUFFERS_BUILD_TESTS=OFF 2>&1 | tail -2 | tee -a "$LOG_FILE"
        cmake --build build -j 2>&1 | tail -2 | tee -a "$LOG_FILE"
        cmake --install build --prefix /opt/flatbuffers-25 2>&1 | tail -2 | tee -a "$LOG_FILE"
        echo "  -> FlatBuffers installed." | tee -a "$LOG_FILE"
    fi
    export CPATH=/opt/flatbuffers-25/include:${CPATH:-}
    export LIBRARY_PATH=/opt/flatbuffers-25/lib:${LIBRARY_PATH:-}
    export LD_LIBRARY_PATH=/opt/flatbuffers-25/lib:${LD_LIBRARY_PATH:-}

    # Apache Arrow（通过 pyarrow 提供头文件，避免 focal/noble 仓库兼容问题）
    if [ -L "/usr/local/include/arrow" ]; then
        echo "  -> Arrow headers already linked, skipping." | tee -a "$LOG_FILE"
    else
        PYARROW_INC=$(python3 -c "import pyarrow; print(pyarrow.get_include())" 2>/dev/null)
        if [ -n "$PYARROW_INC" ] && [ -d "$PYARROW_INC/arrow" ]; then
            ln -sf "$PYARROW_INC/arrow" /usr/local/include/arrow
            ln -sf "$PYARROW_INC/parquet" /usr/local/include/parquet
            echo "  -> Arrow headers linked via pyarrow." | tee -a "$LOG_FILE"
        else
            echo "  -> WARNING: pyarrow not found, Arrow headers unavailable." | tee -a "$LOG_FILE"
        fi
    fi

    echo "  -> C++ libraries done." | tee -a "$LOG_FILE"
fi


# ==============================================================================
#  Phase 2: HSTU 编译（DynamicEmb + CUDA ops + CUTLASS kernels）
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "  Phase 2: HSTU Build"                   | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# ── 2.1 基础依赖 ──────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[2.1] Installing prerequisite packages ..." | tee -a "$LOG_FILE"
pip install gin-config nvtx torchx 2>/dev/null | tail -2 | tee -a "$LOG_FILE"
if python3 -c "import megatron.core" 2>/dev/null; then
    echo "  -> megatron-core available." | tee -a "$LOG_FILE"
else
    echo "  -> WARNING: megatron-core not found." | tee -a "$LOG_FILE"
fi
# torchmetrics 依赖修复
if ! python3 -c "import torchmetrics" 2>/dev/null; then
    pip install lightning_utilities 2>&1 | tail -2 | tee -a "$LOG_FILE"
fi
echo "  -> Prerequisites done." | tee -a "$LOG_FILE"

# ── 2.2 DynamicEmb ────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[2.2] Building dynamicemb ..." | tee -a "$LOG_FILE"
if python3 -c "import dynamicemb" 2>/dev/null; then
    echo "  -> dynamicemb already installed, skipping." | tee -a "$LOG_FILE"
else
    cd "$DYNAMICEMB_DIR"
    pip install . --no-build-isolation 2>&1 | tail -3 | tee -a "$LOG_FILE"
    echo "  -> dynamicemb installed." | tee -a "$LOG_FILE"
fi

# ── 2.3 hstu_cuda_ops ─────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[2.3] Building hstu_cuda_ops ..." | tee -a "$LOG_FILE"
if python3 -c "import hstu_cuda_ops" 2>/dev/null; then
    echo "  -> hstu_cuda_ops already installed, skipping." | tee -a "$LOG_FILE"
else
    cd "$COMMONS"
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
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
]

setup(
    name="hstu_cuda_ops",
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
        python3 "$TEMP_SETUP" build_ext --inplace 2>&1 | tail -5 | tee -a "$LOG_FILE"
    SO_FILE=$(find "$COMMONS" -name "hstu_cuda_ops*.so" -newer "$TEMP_SETUP" 2>/dev/null | head -1)
    [ -z "$SO_FILE" ] && SO_FILE=$(find "$COMMONS/build" -name "hstu_cuda_ops*.so" 2>/dev/null | head -1)
    if [ -z "$SO_FILE" ]; then
        echo "  ERROR: hstu_cuda_ops build failed!" | tee -a "$LOG_FILE"
        rm -f "$TEMP_SETUP"; exit 1
    fi
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    cp "$SO_FILE" "$SITE_PACKAGES/"
    echo "  -> hstu_cuda_ops installed." | tee -a "$LOG_FILE"
    rm -f "$TEMP_SETUP"
fi

# ── 2.4 CUTLASS attention kernel ──────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[2.4] Building CUTLASS attention kernels ..." | tee -a "$LOG_FILE"

# 2.4a: hstu_attn（核心库）
echo "  [2.4a] Building hstu_attn ..." | tee -a "$LOG_FILE"
if python3 -c "import hstu_attn_2_cuda" 2>/dev/null; then
    echo "    -> hstu_attn_2_cuda already installed, skipping." | tee -a "$LOG_FILE"
else
    cd "$REPO_ROOT/corelib/hstu"
    export HSTU_DISABLE_BACKWARD=FALSE
    export HSTU_DISABLE_DETERMINISTIC=TRUE
    export HSTU_DISABLE_LOCAL=FALSE HSTU_DISABLE_CAUSAL=FALSE
    export HSTU_DISABLE_CONTEXT=FALSE HSTU_DISABLE_TARGET=FALSE
    export HSTU_DISABLE_ARBITRARY=FALSE HSTU_ARBITRARY_NFUNC=3
    export HSTU_DISABLE_RAB=FALSE HSTU_DISABLE_DRAB=FALSE
    export HSTU_DISABLE_BF16=FALSE HSTU_DISABLE_FP16=TRUE
    export HSTU_DISABLE_HDIM32=FALSE HSTU_DISABLE_HDIM64=FALSE
    export HSTU_DISABLE_HDIM128=FALSE HSTU_DISABLE_HDIM256=FALSE
    export HSTU_DISABLE_86OR89=TRUE
    echo "    -> Compiling hstu_attn (10-15 min) ..." | tee -a "$LOG_FILE"
    make -j4 2>&1 | tail -5 | tee -a "$LOG_FILE"
    if python3 -c "import hstu_attn_2_cuda" 2>/dev/null; then
        echo "    -> hstu_attn installed." | tee -a "$LOG_FILE"
    else
        echo "    ERROR: hstu_attn build failed!" | tee -a "$LOG_FILE"; exit 1
    fi
fi

# 2.4b: FBGEMM hstu 源码
echo "  [2.4b] Preparing FBGEMM hstu source ..." | tee -a "$LOG_FILE"
FBGEMM_HSTU_DIR="$REPO_ROOT/third_party/FBGEMM/fbgemm_gpu/experimental/hstu"
if [ -d "$FBGEMM_HSTU_DIR/src/hstu_ampere" ]; then
    echo "    -> Source already present, skipping." | tee -a "$LOG_FILE"
else
    mkdir -p "$FBGEMM_HSTU_DIR"
    python3 << 'EOF'
import urllib.request, tarfile, io, os
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
EOF
    echo "    -> Source downloaded." | tee -a "$LOG_FILE"
fi

# 2.4c: Kernel 实例化代码
echo "  [2.4c] Generating kernel instantiations ..." | tee -a "$LOG_FILE"
if [ -d "$FBGEMM_HSTU_DIR/src/hstu_ampere/instantiations" ] && \
   [ "$(ls "$FBGEMM_HSTU_DIR/src/hstu_ampere/instantiations/"*.cu 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "    -> Already generated, skipping." | tee -a "$LOG_FILE"
else
    cd "$FBGEMM_HSTU_DIR/src"
    export HSTU_DISABLE_BACKWARD=FALSE HSTU_DISABLE_DETERMINISTIC=TRUE
    export HSTU_DISABLE_BF16=FALSE HSTU_DISABLE_FP16=TRUE
    export HSTU_DISABLE_HDIM32=FALSE HSTU_DISABLE_HDIM64=FALSE
    export HSTU_DISABLE_HDIM128=FALSE HSTU_DISABLE_HDIM256=FALSE
    export HSTU_DISABLE_LOCAL=FALSE HSTU_DISABLE_CAUSAL=FALSE
    export HSTU_DISABLE_CONTEXT=FALSE HSTU_DISABLE_TARGET=FALSE
    export HSTU_DISABLE_ARBITRARY=FALSE HSTU_DISABLE_RAB=FALSE HSTU_DISABLE_DRAB=FALSE
    python3 generate_kernels.py --arch-list "8.0" 2>&1 | tee -a "$LOG_FILE"
    echo "    -> Generated." | tee -a "$LOG_FILE"
fi

# 2.4d: setup.py
echo "  [2.4d] Preparing FBGEMM hstu setup.py ..." | tee -a "$LOG_FILE"
if [ -f "$FBGEMM_HSTU_DIR/setup.py" ]; then
    echo "    -> setup.py exists, skipping." | tee -a "$LOG_FILE"
else
    cat > "$FBGEMM_HSTU_DIR/setup.py" << 'SETUPEOF'
import glob, os, sys
from pathlib import Path
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = Path(this_dir).resolve().parent.parent.parent.parent
cutlass_dir = repo_root.parent.parent.parent.parent / "third_party" / "cutlass"
if not cutlass_dir.exists():
    cutlass_dir = Path("/wang/recsys-examples/third_party/cutlass")

HSTU_ARCH_LIST = os.getenv("HSTU_ARCH_LIST", "8.0").split()
ampere_src = os.path.join(this_dir, "src", "hstu_ampere")
hopper_src = os.path.join(this_dir, "src", "hstu_hopper")
sources, include_dirs = [], []
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
    if arch == "8.0": cc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
    elif arch in ("9.0", "9.0a"): cc_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
nvcc_flags = ["-O3", "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "-lineinfo"] + cc_flags
nvcc_threads = os.getenv("NVCC_THREADS", "4")
nvcc_flags += ["--threads", nvcc_threads]
feature_flags = ["-DHSTU_DISABLE_ARBITRARY", "-DHSTU_DISABLE_LOCAL",
    "-DHSTU_DISABLE_RAB", "-DHSTU_DISABLE_DRAB", "-DHSTU_DISABLE_FP16",
    "-DHSTU_DISABLE_DETERMINISTIC", "-DHSTU_DISABLE_86OR89", "-DHSTU_ARBITRARY_NFUNC=1"]
cxx_flags = ["-O3", "-std=c++17"] + feature_flags
nvcc_flags += feature_flags
setup(name="fbgemm_gpu_hstu", version="0.1.0", packages=["hstu"],
    ext_modules=[CUDAExtension(name="hstu.hstu_ops_gpu", sources=sources,
        include_dirs=include_dirs, extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags})],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.7", install_requires=["torch"])
SETUPEOF
    echo "    -> setup.py created." | tee -a "$LOG_FILE"
fi

# 2.4e: library.py 加载逻辑
echo "  [2.4e] Updating library.py ..." | tee -a "$LOG_FILE"
cat > "$FBGEMM_HSTU_DIR/hstu/library.py" << 'LIBEOF'
#!/usr/bin/env python3
import glob, logging, os
import torch
try:
    from fbgemm_gpu import open_source
except Exception:
    open_source: bool = False

def _load_hstu_ops():
    pkg_dir = os.path.dirname(__file__)
    so_files = glob.glob(os.path.join(pkg_dir, "hstu_ops_gpu*.so"))
    if so_files:
        torch.ops.load_library(so_files[0])
        return
    fbgemm_so = os.path.join(pkg_dir, "fbgemm_gpu_experimental_hstu.so")
    if os.path.exists(fbgemm_so):
        torch.ops.load_library(fbgemm_so)
        torch.classes.load_library(fbgemm_so)
        return

if torch.cuda.is_available():
    try:
        _load_hstu_ops()
    except Exception as e:
        logging.warning(f"Failed to load HSTU CUDA ops: {e}")
LIBEOF
echo "    -> library.py updated." | tee -a "$LOG_FILE"

# 2.4f: 编译 hstu 包
echo "  [2.4f] Building hstu (FBGEMM interface) ..." | tee -a "$LOG_FILE"
if python3 -c "import torch; import hstu; _ = torch.ops.fbgemm.hstu_varlen_fwd_80" 2>/dev/null; then
    echo "    -> hstu (FBGEMM) already installed, skipping." | tee -a "$LOG_FILE"
else
    cd "$FBGEMM_HSTU_DIR"
    export HSTU_ARCH_LIST="8.0"
    export MAX_JOBS="${MAX_JOBS:-39}"
    export NVCC_THREADS="${NVCC_THREADS:-1}"
    echo "    -> Compiling (5-10 min) ..." | tee -a "$LOG_FILE"
    pip install --no-build-isolation -e . 2>&1 | tail -5 | tee -a "$LOG_FILE"
    echo "    -> hstu (FBGEMM) installed." | tee -a "$LOG_FILE"
fi

# ── 2.5 代码补丁 ──────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[2.5] Applying code patches ..." | tee -a "$LOG_FILE"

# hstu_attention.py — 懒加载
PATCH_FILE="$HSTU_DIR/modules/hstu_attention.py"
if grep -q "^from hstu import hstu_attn_varlen_func" "$PATCH_FILE" 2>/dev/null; then
    sed -i '/^from hstu import hstu_attn_varlen_func$/d' "$PATCH_FILE"
    sed -i '/^        return hstu_attn_varlen_func(/i\        from hstu import hstu_attn_varlen_func\n' "$PATCH_FILE"
    echo "  -> hstu_attention.py patched." | tee -a "$LOG_FILE"
else
    echo "  -> hstu_attention.py already patched." | tee -a "$LOG_FILE"
fi

# fused_hstu_op.py — try/except import
PATCH_FILE="$HSTU_DIR/ops/fused_hstu_op.py"
if grep -q "^import hstu " "$PATCH_FILE" 2>/dev/null; then
    python3 -c "
import pathlib
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
    echo "  -> fused_hstu_op.py patched." | tee -a "$LOG_FILE"
else
    echo "  -> fused_hstu_op.py already patched." | tee -a "$LOG_FILE"
fi

# trainer/utils.py — PYTORCH -> DEBUG layer type
PATCH_FILE="$HSTU_DIR/training/trainer/utils.py"
if grep -q "if kernel_backend == KernelBackend.PYTORCH:" "$PATCH_FILE" 2>/dev/null; then
    echo "  -> trainer/utils.py already patched." | tee -a "$LOG_FILE"
else
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
    elif tensor_model_parallel_arg.tensor_model_parallel_size == 1:
        layer_type = HSTULayerType.FUSED'''
text = text.replace(old, new)
p.write_text(text)
" 2>/dev/null || true
    echo "  -> trainer/utils.py patched." | tee -a "$LOG_FILE"
fi


# ==============================================================================
#  完成
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "  Setup Complete!"                           | tee -a "$LOG_FILE"
echo "  Finished: $(date)"                        | tee -a "$LOG_FILE"
echo "  Log: $LOG_FILE"                           | tee -a "$LOG_FILE"
echo ""                                            | tee -a "$LOG_FILE"
echo "  Next: bash ppu/scripts/run_benchmark.sh"  | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
