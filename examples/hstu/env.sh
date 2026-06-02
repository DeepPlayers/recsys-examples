#!/bin/bash
set -euo pipefail

# ==============================================================================
# 环境初始化脚本
# 说明：PPU 镜像构建脚本，包含 apt 系统依赖、Python 包安装及 C++ 库源码编译
# ==============================================================================


# ==============================================================================
# 【第一步】系统环境初始化
# ==============================================================================
unset PYTHONPATH

# 替换 apt 源 & profile 配置
wget https://dw-mofang.oss-cn-shanghai.aliyuncs.com/kube-image/ppu/sources.list -O /etc/apt/sources.list
rm -f /etc/apt/sources.list.d/ubuntu.sources
apt-get update
wget https://dw-mofang.oss-cn-shanghai.aliyuncs.com/kube-image/ppu/profile -O /etc/profile


# ==============================================================================
# 【第二步】APT 系统依赖安装
# ==============================================================================

# --- 2.1 C++ 构建 & 绑定依赖 ---
apt-get install -y --no-install-recommends \
    libboost-fiber-dev \
    libboost-context-dev \
    pybind11-dev \
    libgnutls28-dev

apt-get install -y --no-install-recommends\
       build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev \
       libyaml-cpp-dev libaio-dev \
       libgmp10 libgnutls28-dev \
       libboost-fiber-dev libboost-context-dev pybind11-dev libtbb-dev libgoogle-glog-dev libgtest-dev tree

# ==============================================================================
# 【第三步】Python 包安装
# ==============================================================================
MIRROR="-i https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu129/simple/"
# MIRROR="-i https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/"
FLAGS="--no-cache-dir --no-build-isolation"

# --- 3.1 PyTorch 生态 ---
# pip install $FLAGS $MIRROR torch==2.10.0          # 已预装，按需启用
pip install $FLAGS $MIRROR fbgemm-gpu==1.4.0
pip install $FLAGS $MIRROR torchmetrics==1.0.3
pip install $FLAGS $MIRROR tensordict
pip install $FLAGS $MIRROR torchrec

# --- 3.2 大模型训练框架 ---
## 编译安装megatron-core（已安装 0.15.0 则跳过；本地已有目录则跳过clone直接编译）
if pip show megatron-core 2>/dev/null | grep -q "Version: 0.15.0"; then
    echo "[SKIP] megatron-core 0.15.0 already installed, skipping build."
else
    if [ ! -d "Megatron-LM" ]; then
        git clone -b core_v0.15.0 https://github.com/NVIDIA/Megatron-LM.git
    else
        echo "[SKIP] Megatron-LM directory already exists, skipping clone."
    fi
    cd Megatron-LM
    pip install --no-build-isolation .
    cd ..
fi

## 其他依赖
# pip install $FLAGS $MIRROR megatron-core[mlm,lts]==0.15.0 --no-deps
# pip install $FLAGS $MIRROR flash-attn==2.5.8      # 按需启用
# pip install $FLAGS $MIRROR deepspeed==0.18.7      # 按需启用
# pip install deepspeed==0.19.0                     # 按需启用

# --- 3.3 NLP / 模型推理 ---
pip install $FLAGS $MIRROR lion_pytorch
pip install $FLAGS $MIRROR onnx
pip install $FLAGS $MIRROR sentencepiece --prefer-binary
pip install $FLAGS $MIRROR optimum[onnxruntime] --prefer-binary
# pip install $FLAGS $MIRROR transformers==4.33.3   # 按需启用

# --- 3.4 训练监控 / 可视化 ---
pip install $FLAGS $MIRROR torch-tb-profiler
pip install $FLAGS $MIRROR thop
pip install $FLAGS $MIRROR gin-config==0.5.0
pip install $FLAGS $MIRROR torchviz
# pip install $FLAGS $MIRROR tensorboard==2.19.0    # 按需启用

# --- 3.5 数值计算 ---
pip install $FLAGS $MIRROR "numpy<2.0"
pip install $FLAGS $MIRROR scikit-learn
pip install $FLAGS $MIRROR pyarrow --prefer-binary

# --- 3.6 数据处理 / 存储 ---
pip install $FLAGS $MIRROR oss2
pip install $FLAGS $MIRROR pymysql
pip install $FLAGS $MIRROR pyodps==0.12.0
pip install $FLAGS $MIRROR zstandard
pip install $FLAGS $MIRROR seaborn
pip install $FLAGS $MIRROR gputil
pip install $FLAGS $MIRROR pypinyin
pip install $FLAGS $MIRROR liger-kernel==0.7.0

# --- 3.7 Web 服务 ---
pip install $FLAGS $MIRROR fastapi
pip install $FLAGS $MIRROR uvicorn
pip install $FLAGS $MIRROR taskflow

# --- 3.8 通用工具 ---
pip install $FLAGS $MIRROR pyyaml
pip install $FLAGS $MIRROR pytest
pip install $FLAGS $MIRROR loguru
pip install $FLAGS $MIRROR psutil
pip install $FLAGS $MIRROR certifi

# --- 3.9 构建工具（Python 侧）---
pip install $FLAGS $MIRROR wheel
pip install $FLAGS $MIRROR cmake
pip install $FLAGS $MIRROR yapf
pip install $FLAGS $MIRROR clang-format
pip install $FLAGS $MIRROR pybind11
# pip install --upgrade pip setuptools wheel        # 按需启用

# --- 3.10 开发环境 ---
pip install $FLAGS $MIRROR jupyterlab


# ==============================================================================
# 【第四步】C++ 库源码编译安装
# ==============================================================================

# --- 4.1 编译安装 FlatBuffers v25.2.10 ---
cd /tmp
git clone https://github.com/google/flatbuffers.git
cd flatbuffers
git checkout v25.2.10
cmake -S . -B build -DFLATBUFFERS_BUILD_TESTS=OFF
cmake --build build -j
cmake --install build --prefix /opt/flatbuffers-25

# 注：安装路径为 /opt/flatbuffers-25，以下环境变量路径请按实际版本修正
export CPATH=/opt/flatbuffers-25/include:${CPATH:-}
export LIBRARY_PATH=/opt/flatbuffers-25/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/opt/flatbuffers-25/lib:${LD_LIBRARY_PATH:-}

# --- 4.2 安装 Apache Arrow & Parquet ---
cd /tmp
wget https://dw-mofang.oss-cn-shanghai.aliyuncs.com/kube-image/ppu/apache-arrow-apt-source-latest-focal.deb
dpkg -i apache-arrow-apt-source-latest-focal.deb
apt-get remove -y libarrow-glib700 libarrow700 || true  # 包不存在时跳过
apt-get update
apt-get install -y --no-install-recommends libarrow-dev libparquet-dev

# 修复头文件软链（避免与系统路径冲突）
rm -rf /usr/local/include/arrow
rm -rf /usr/local/include/parquet
ln -s /usr/include/arrow  /usr/local/include/arrow
ln -s /usr/include/parquet /usr/local/include/parquet


echo "[INFO] 环境初始化完成"

