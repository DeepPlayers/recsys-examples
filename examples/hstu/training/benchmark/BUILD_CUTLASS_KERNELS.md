# PPU 环境下编译 HSTU CUTLASS Attention Kernel 指南

## 背景

在 PPU-ZW810E（阿里云 PPU 加速卡，SM 8.0）环境下运行 HSTU E2E benchmark 时，需要两个 CUTLASS attention kernel 包：

1. **`hstu_attn`**（核心库）：从 `corelib/hstu` 编译，提供 `hstu_attn_2_cuda.varlen_fwd/bwd`
2. **`hstu`**（FBGEMM 接口层）：从 `third_party/FBGEMM/fbgemm_gpu/experimental/hstu` 编译，注册 `torch.ops.fbgemm.hstu_varlen_fwd_80/bwd_80`

两者关系：
- `hstu_attn` 是底层 CUTLASS kernel 实现（基于 Flash Attention 修改）
- `hstu` 是 FBGEMM 风格的 Python 接口封装，调用 `hstu_attn` 的 C++ 函数

## 前置条件

- PPU SDK 已安装（包含 `nvcc`）
- PyTorch >= 2.0（PPU 定制版本）
- CUTLASS 源码已克隆到 `third_party/cutlass`
- Python 3.8+
- `pip` 包管理工具

## 编译步骤

### 步骤 1：编译 hstu_attn（核心库）

```bash
cd corelib/hstu

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

# 编译安装（使用 4 线程并行编译）
make -j4

# 验证安装
python -c "import hstu_attn_2_cuda; print('hstu_attn_2_cuda loaded successfully')"
```

**编译产物**：
- Python 包：`hstu_attn==0.1.0`
- 动态库：`hstu_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`（约 360 MB）
- 注册操作符：`hstu_attn_2_cuda.varlen_fwd`, `hstu_attn_2_cuda.varlen_bwd`

**编译时间**：约 10-15 分钟（4 线程）

### 步骤 2：下载 FBGEMM hstu 源码

由于 PPU 环境无法直接访问 GitHub，需要通过 GitHub API 下载源码。

```bash
# 目标目录
TARGET_DIR=third_party/FBGEMM/fbgemm_gpu/experimental/hstu
mkdir -p $TARGET_DIR

# 通过 GitHub API 下载 jiayus-nvidia/FBGEMM fork（包含 hstu 实验代码）
python3 << 'EOF'
import urllib.request
import tarfile
import io
import os

url = "https://api.github.com/repos/jiayus-nvidia/FBGEMM/tarball/main"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
tar = tarfile.open(fileobj=io.BytesIO(response.read()))

# 提取 hstu 相关文件
for member in tar.getmembers():
    if 'fbgemm_gpu/experimental/hstu' in member.name:
        # 调整路径
        rel_path = member.name.split('experimental/hstu/')[-1]
        if rel_path and not member.isdir():
            target_path = f"third_party/FBGEMM/fbgemm_gpu/experimental/hstu/{rel_path}"
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with tar.extractfile(member) as src, open(target_path, 'wb') as dst:
                dst.write(src.read())

print("Downloaded FBGEMM hstu source code")
EOF
```

**下载内容**：
- `hstu/__init__.py`, `hstu/library.py`, `hstu/cuda_hstu_attention.py`
- `src/hstu_ampere/*`（SM 8.0 kernel 实现）
- `src/hstu_hopper/*`（SM 9.0 kernel 实现，本次未使用）
- `src/generate_kernels.py`（kernel 实例化代码生成脚本）

### 步骤 3：生成 kernel 实例化代码

```bash
cd third_party/FBGEMM/fbgemm_gpu/experimental/hstu/src

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

# 生成 SM 8.0 kernel 实例化文件
python generate_kernels.py --arch-list "8.0"

# 验证生成结果
ls hstu_ampere/instantiations/*.cu | wc -l  # 应该输出约 80 个文件
```

**生成产物**：
- `hstu_ampere/instantiations/hstu_fwd_hdim{32,64,128,256}_bf16_*_sm80.cu`
- `hstu_ampere/instantiations/hstu_bwd_hdim{32,64,128,256}_bf16_*_sm80.cu`
- 每个文件对应一个 kernel 变体（不同的 head_dim、mask 组合）

### 步骤 4：创建 setup.py

FBGEMM 官方没有为 `hstu` 子目录提供 `setup.py`（依赖 CMake 构建系统），需要创建一个独立编译脚本。

```bash
cat > third_party/FBGEMM/fbgemm_gpu/experimental/hstu/setup.py << 'EOF'
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
cutlass_dir = repo_root.parent.parent.parent / "third_party" / "cutlass"
if not cutlass_dir.exists():
    # Fallback: try relative to recsys-examples
    cutlass_dir = Path("/workspace/recsys-examples/third_party/cutlass")

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
EOF
```

### 步骤 5：修改 library.py 加载逻辑

原版 `library.py` 尝试加载 CMake 构建产物 `fbgemm_gpu_experimental_hstu.so`，需要改为加载 `setup.py` 编译的 `hstu_ops_gpu.so`。

```bash
cat > third_party/FBGEMM/fbgemm_gpu/experimental/hstu/hstu/library.py << 'EOF'
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
```

### 步骤 6：编译安装 hstu 包

```bash
cd third_party/FBGEMM/fbgemm_gpu/experimental/hstu

# 设置编译选项
export HSTU_ARCH_LIST="8.0"  # 仅编译 SM 8.0（PPU-ZW810E）
export MAX_JOBS=39           # 并行编译任务数（建议 CPU 核心数 - 1）
export NVCC_THREADS=1        # 每个 nvcc 进程使用的线程数

# 编译安装（开发模式）
pip install --no-build-isolation -e .

# 验证安装
python -c "
import torch
import hstu
print('hstu package loaded')
fwd_op = torch.ops.fbgemm.hstu_varlen_fwd_80
bwd_op = torch.ops.fbgemm.hstu_varlen_bwd_80
print('✓ torch.ops.fbgemm.hstu_varlen_fwd_80 registered')
print('✓ torch.ops.fbgemm.hstu_varlen_bwd_80 registered')
"
```

**编译产物**：
- Python 包：`fbgemm_gpu_hstu==0.1.0`（开发模式安装）
- 动态库：`hstu/hstu_ops_gpu.so`（约 950 MB）
- 注册操作符：`torch.ops.fbgemm.hstu_varlen_fwd_80`, `torch.ops.fbgemm.hstu_varlen_bwd_80`

**编译时间**：约 5-10 分钟（39 并行任务）

## 验证完整环境

```bash
python << 'EOF'
import torch

# 检查 hstu_attn
try:
    import hstu_attn_2_cuda
    print("✓ hstu_attn_2_cuda loaded")
except ImportError as e:
    print(f"✗ hstu_attn_2_cuda not found: {e}")

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

# 检查 GPU
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA Version: {torch.version.cuda}")
EOF
```

预期输出：
```
✓ hstu_attn_2_cuda loaded
✓ hstu package loaded
✓ torch.ops.fbgemm.hstu_varlen_fwd_80 available
✓ torch.ops.fbgemm.hstu_varlen_bwd_80 available

GPU: PPU-ZW810E
CUDA Capability: (8, 0)
CUDA Version: 12.9
```

## 常见问题

### Q1: 编译时报错 `HSTU_ARBITRARY_NFUNC` 未定义

**原因**：`hstu_ops_gpu.cpp` 中引用了 `HSTU_ARBITRARY_NFUNC` 宏，但未在编译选项中定义。

**解决**：在 `setup.py` 的 `feature_flags` 中添加 `-DHSTU_ARBITRARY_NFUNC=1`。

### Q2: 链接时报 undefined symbol `run_hstu_bwd_80<80, bf16, 32, ...>`

**原因**：kernel 实例化文件缺失，或编译选项与生成选项不一致。

**解决**：
1. 确保步骤 3 和步骤 6 的 `HSTU_DISABLE_*` 选项完全一致
2. 检查 `hstu_ampere/instantiations/` 目录下是否有约 80 个 `.cu` 文件
3. 如果不确定，重新执行步骤 3 生成 kernel 文件

### Q3: `library.py` 加载 `.so` 失败

**原因**：`.so` 文件路径不对，或依赖库缺失。

**解决**：
```bash
# 检查 .so 是否存在
ls -lh third_party/FBGEMM/fbgemm_gpu/experimental/hstu/hstu/hstu_ops_gpu.so

# 检查依赖
ldd third_party/FBGEMM/fbgemm_gpu/experimental/hstu/hstu/hstu_ops_gpu.so | grep "not found"

# 手动加载测试
python -c "import torch; torch.ops.load_library('third_party/FBGEMM/fbgemm_gpu/experimental/hstu/hstu/hstu_ops_gpu.so')"
```

### Q4: 运行时 `fused_hstu_op.py` 调用 `torch.ops.fbgemm.hstu_varlen_fwd_80` 失败

**原因**：`hstu` 包未正确安装，或 `library.py` 未正确加载 `.so`。

**解决**：
```bash
# 检查 hstu 包是否安装
pip list | grep fbgemm_gpu_hstu

# 检查 torch ops 是否注册
python -c "import torch; print(torch.ops.fbgemm.hstu_varlen_fwd_80)"

# 如果失败，重新执行步骤 5-6
```

## 文件结构

编译完成后的关键文件：

```
recsys-examples/
├── corelib/hstu/
│   ├── setup.py                          # hstu_attn 编译脚本（已有）
│   ├── build/                            # 编译中间产物
│   └── hstu_attn_2_cuda.*.so            # hstu_attn 动态库
│
└── third_party/
    ├── cutlass/                          # CUTLASS 源码（submodule）
    │   ├── include/
    │   │   ├── cutlass/
    │   │   └── cute/
    │   └── tools/util/include/
    │
    └── FBGEMM/fbgemm_gpu/experimental/hstu/
        ├── setup.py                      # hstu 编译脚本（本次创建）
        ├── hstu/
        │   ├── __init__.py              # 包入口
        │   ├── library.py               # .so 加载逻辑（本次修改）
        │   ├── cuda_hstu_attention.py   # Python 接口
        │   └── hstu_ops_gpu.so          # hstu 动态库（编译产物）
        └── src/
            ├── generate_kernels.py      # kernel 生成脚本
            ├── hstu_ampere/
            │   ├── hstu_ops_gpu.cpp     # SM 8.0 op 注册
            │   ├── hstu_fwd.h           # Forward kernel 实现
            │   ├── hstu_bwd.h           # Backward kernel 实现
            │   └── instantiations/      # 生成的 kernel 实例化文件（~80 个 .cu）
            └── hstu_hopper/             # SM 9.0 kernel（本次未使用）
```

## 参考资源

- [CUTLASS 官方文档](https://github.com/NVIDIA/cutlass)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [FBGEMM HSTU 实验代码](https://github.com/jiayus-nvidia/FBGEMM/tree/main/fbgemm_gpu/experimental/hstu)
- [PPU SDK 文档](https://help.aliyun.com/document_detail/xxx.html)（阿里云内部文档）
