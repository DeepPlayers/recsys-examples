# inference_emb_ops.so 编译改动整理

## 概述

`inference_emb_ops.so` 是一个基于 CMake 构建的 CUDA 共享库，注册了 `torch.ops.INFERENCE_EMB.*` 系列自定义算子，供 DynamicEmb 可导出推理表和 HSTU C++ AOTInductor 推理 demo 使用。

- **输出产物**: `corelib/dynamicemb/torch_binding_build/inference_emb_ops.so`（ELF 64-bit, ~3.7MB）
- **构建系统**: CMake 3.18+ (实际使用 4.3) + Unix Makefiles
- **命名空间**: 所有算子注册在 `INFERENCE_EMB` 命名空间下

---

## 1. 构建系统 (CMakeLists.txt)

### 文件路径

```
corelib/dynamicemb/CMakeLists.txt
```

### 关键配置

| 配置项 | 值 |
|--------|------|
| 项目名 | `DynamicEmbInferenceOps` |
| 语言 | CXX + CUDA |
| C++ 标准 | C++17 (required, no extensions) |
| CUDA 标准 | C++17 (required, no extensions) |
| 位置无关代码 | ON (`CMAKE_POSITION_INDEPENDENT_CODE`) |
| CUDA 架构 (默认) | 70, 75, 80, 89, 90 |
| CUDA 架构 (当前构建) | 80 (sm_80) |
| CXX 编译器 | `/usr/bin/c++` (GCC 13) |
| CUDA 编译器 | `/usr/local/PPU_SDK/CUDA_SDK/bin/nvcc` |
| Python | 3.12 (`/opt/ac2/`) |
| Torch | 从 PyTorch 安装路径自动发现 |

### 构建命令

```bash
cd corelib/dynamicemb
mkdir -p torch_binding_build
cd torch_binding_build
cmake ..
make -j
```

---

## 2. 源文件清单

CMakeLists.txt 中定义的 `INFERENCE_EMB_SOURCES` 包含以下 9 个编译单元：

### 2.1 CUDA 源文件 (.cu)

| # | 文件路径 | 功能 |
|---|---------|------|
| 1 | `src/table_operation/lookup_torch_binding.cu` | `INFERENCE_EMB::table_lookup` 算子的 Torch 绑定注册 (FRAGMENT + CUDA/CPU dispatch) |
| 2 | `src/table_operation/get_table_range_torch_binding.cu` | `INFERENCE_EMB::get_table_range` 算子的 Torch 绑定注册 |
| 3 | `src/table_operation/expand_table_ids_torch_binding.cu` | `INFERENCE_EMB::expand_table_ids` 算子的 Torch 绑定注册，含推理专用 CUDA kernel |
| 4 | `src/table_operation/lookup.cu` | 哈希表 lookup 核心实现 (含溢出表支持) |
| 5 | `src/index_calculation.cu` | 索引计算：`get_table_range`, `segmented_sum_cuda`, `flagged_compact` |
| 6 | `src/unique_op.cu` | 分段 unique 操作：`segmented_unique_cuda`, `expand_table_ids_cuda`, `compute_dedup_lengths_cuda` |
| 7 | `src/lookup_forward.cu` | 前向 lookup 的 scatter-combine 和 scatter-fused 操作 |
| 8 | `src/torch_utils.cu` | PyTorch 工具函数：类型转换、设备时间戳 |
| 9 | `src/utils.cpp` | 设备属性查询 (`DeviceProp`) |

### 2.2 头文件依赖

| 文件路径 | 功能 |
|---------|------|
| `src/check.h` | CUDA 错误检查宏 (`DEMB_CUDA_CHECK`, `DEMB_CUDA_KERNEL_LAUNCH_CHECK`) |
| `src/utils.h` | 基础类型定义 (`DataType`, `EvictStrategy`)、dispatch 宏、`DeviceProp`, `TypeConvertFunc` |
| `src/torch_utils.h` | Torch 类型映射 (`scalartype_to_datatype` 等)、指针工具 |
| `src/index_calculation.h` | CUB select/compact 模板、`segmented_sum_cuda` 声明 |
| `src/unique_op.h` | `segmented_unique_cuda`, `expand_table_ids_cuda`, `compute_dedup_lengths_cuda` 声明 |
| `src/lookup_forward.h` | `scatter_combine`, `scatter_fused`, `get_new_length_and_offsets` 声明 |
| `src/lookup_kernel.cuh` | 向量化 gather/scatter CUDA kernel |
| `src/table_operation/types.cuh` | 核心类型：`LinearBucket`, `LinearBucketTable`, `InsertResult`, 锁机制, probe 算法 |
| `src/table_operation/kernels.cuh` | 所有哈希表 CUDA kernel：lookup/insert/erase/export/traverse + overflow 操作 |
| `src/table_operation/score.cuh` | Score 类型定义和策略 (`Const`, `Assign`, `Accumulate`, `GlobalTimer`) |
| `src/table_operation/table.cuh` | (包含 types.cuh) |

---

## 3. 注册的 Torch 自定义算子

所有算子通过 `TORCH_LIBRARY_FRAGMENT(INFERENCE_EMB, m)` 注册，支持跨多文件拆分注册。

### 3.1 `INFERENCE_EMB::table_lookup`

- **定义文件**: `src/table_operation/lookup_torch_binding.cu`
- **实现文件**: `src/table_operation/lookup.cu`
- **Schema**:
  ```
  table_lookup(
      Tensor table_storage,
      Tensor table_bucket_offsets,
      int bucket_capacity,
      Tensor keys,
      Tensor table_ids,
      Tensor? score_input,
      int policy_type,
      Tensor? ovf_storage=None,
      int ovf_bucket_capacity=0,
      Tensor? ovf_output_offsets=None
  ) -> (Tensor, Tensor, Tensor)
  ```
- **返回**: `(score_output, founds, indices)` — 三个 1D tensor
- **Dispatch**:
  - CUDA: `table_lookup_cuda_impl` → `table_lookup()` → `table_lookup_single_score()` 或 `table_lookup_with_overflow_single_score()`
  - CPU: 抛出错误 (CUDA-only 算子)
- **核心 Kernel**: `table_lookup_kernel<Table, 1, PolicyTypeV, EnableOverflow>` (定义在 `kernels.cuh`)
- **支持功能**:
  - 线性探测哈希表查找
  - Digest 向量加速比较 (4 字节向量化)
  - 溢出表 (overflow table) 回退查找
  - Score 策略: Const / Assign / Accumulate / GlobalTimer

### 3.2 `INFERENCE_EMB::get_table_range`

- **定义文件**: `src/table_operation/get_table_range_torch_binding.cu`
- **实现文件**: `src/index_calculation.cu`
- **Schema**:
  ```
  get_table_range(Tensor offsets, Tensor feature_offsets) -> Tensor
  ```
- **Dispatch**:
  - CUDA: `get_table_range_cuda_impl` → `get_table_range()`
  - CPU: 抛出错误 (CUDA-only 算子)
- **核心 Kernel**: `get_table_range_kernel<InT, OutT>` — 将 `(feature, batch)` 维度的 offsets 转换到 `table` 维度

### 3.3 `INFERENCE_EMB::expand_table_ids`

- **定义文件**: `src/table_operation/expand_table_ids_torch_binding.cu`
- **实现文件**: 同文件 (kernel 内联)
- **Schema**:
  ```
  expand_table_ids(
      Tensor offsets,
      Tensor indices,
      Tensor? table_offsets_in_feature=None,
      int num_tables=0,
      int local_batch_size=1
  ) -> Tensor
  ```
- **Dispatch**:
  - CUDA: `expand_table_ids_cuda_impl` — 推理专用 kernel，支持 `feature_offsets` + `local_batch_size`
  - Meta: `expand_table_ids_meta_impl` — 返回 `at::empty_like(indices)` 用于 torch.export tracing
  - CPU: 已注释掉 (CUDA-only)
- **核心 Kernel**: `expand_table_ids_inference_kernel` — 通过二分查找为每个 key 分配 table_id
- **关键改动**: 与 `unique_op.cu` 中的 `expand_table_ids_cuda` (仅支持 identity mapping) 不同，此版本支持完整的 `feature_offsets` + `local_batch_size` 参数

---

## 4. 编译选项详解

### 4.1 CXX 编译选项

```
-O3                          # 最高优化级别
-fdiagnostics-color=always   # 彩色编译输出
-fvisibility=hidden          # 隐藏非导出符号
-w                           # 抑制所有警告
-std=c++17                   # C++17 标准
-fPIC                        # 位置无关代码 (共享库)
```

### 4.2 CUDA 编译选项

```
-O3                          # 最高优化级别
--expt-relaxed-constexpr     # 允许 constexpr 中使用 __device__ 函数
--expt-extended-lambda       # 允许 __device__ lambda
--use_fast_math              # 快速数学运算
-Xcompiler=-fvisibility=hidden  # 传递 hidden visibility 给 host 编译器
-w                           # 抑制所有警告
-U__CUDA_NO_HALF_OPERATORS__         # 取消 half 运算符禁用
-U__CUDA_NO_HALF_CONVERSIONS__       # 取消 half 转换禁用
-U__CUDA_NO_HALF2_OPERATORS__        # 取消 half2 运算符禁用
-U__CUDA_NO_BFLOAT16_CONVERSIONS__   # 取消 bfloat16 转换禁用
-std=c++17                   # C++17 标准
-gencode arch=compute_80,code=sm_80  # 目标 GPU 架构
```

### 4.3 预处理宏定义

```
TORCH_EXTENSION_NAME=inference_emb_ops
USE_C10D_GLOO
USE_C10D_MPI
USE_C10D_NCCL
USE_DISTRIBUTED
USE_RPC
USE_TENSORPIPE
inference_emb_ops_EXPORTS
```

### 4.4 Include 路径

```
corelib/dynamicemb/src                        # 项目内部头文件
corelib/dynamicemb/src/table_operation        # table_operation 子目录头文件
${Python3_INCLUDE_DIRS}                       # Python 3.12 头文件
${TORCH_INCLUDE_DIRS}                         # PyTorch 头文件
${CUDAToolkit_INCLUDE_DIRS}                   # CUDA Toolkit 头文件
```

### 4.5 链接依赖

| 库 | 用途 |
|----|------|
| `libpython3.12.so` | Python 运行时 |
| `libtorch.so` | PyTorch 核心 |
| `libtorch_cpu.so` | PyTorch CPU 后端 |
| `libtorch_cuda.so` | PyTorch CUDA 后端 |
| `libc10.so` | C10 基础库 |
| `libc10_cuda.so` | C10 CUDA 扩展 |
| `libnvrtc.so` | CUDA Runtime Compilation |
| `libcudart.so` | CUDA Runtime |
| `cudadevrt` | CUDA Device Runtime |
| `cudart_static` | CUDA Runtime (静态) |
| `cuda` | CUDA Driver |
| `pthread`, `dl`, `rt` | POSIX 系统库 |

### 4.6 链接命令 (实际)

```bash
/usr/bin/c++ -fPIC -shared \
  -Wl,-soname,inference_emb_ops.so \
  -o inference_emb_ops.so \
  CMakeFiles/inference_emb_ops.dir/src/table_operation/lookup_torch_binding.cu.o \
  CMakeFiles/inference_emb_ops.dir/src/table_operation/get_table_range_torch_binding.cu.o \
  CMakeFiles/inference_emb_ops.dir/src/table_operation/expand_table_ids_torch_binding.cu.o \
  CMakeFiles/inference_emb_ops.dir/src/table_operation/lookup.cu.o \
  CMakeFiles/inference_emb_ops.dir/src/index_calculation.cu.o \
  CMakeFiles/inference_emb_ops.dir/src/unique_op.cu.o \
  CMakeFiles/inference_emb_ops.dir/src/lookup_forward.cu.o \
  CMakeFiles/inference_emb_ops.dir/src/torch_utils.cu.o \
  CMakeFiles/inference_emb_ops.dir/src/utils.cpp.o \
  -L/lib/intel64 -L/lib/intel64_win -L/lib/win-x64 \
  -L/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/stubs \
  -Wl,-rpath,/lib/intel64:/lib/intel64_win:/lib/win-x64:/opt/ac2/lib:... \
  /opt/ac2/lib/libpython3.12.so \
  /opt/ac2/lib/python3.12/site-packages/torch/lib/libtorch.so \
  /opt/ac2/lib/python3.12/site-packages/torch/lib/libc10.so \
  /usr/local/PPU_SDK/CUDA_SDK/lib64/libnvrtc.so \
  /opt/ac2/lib/python3.12/site-packages/torch/lib/libc10_cuda.so \
  /usr/local/PPU_SDK/CUDA_SDK/lib64/libcudart.so \
  -Wl,--no-as-needed,".../libtorch_cpu.so" -Wl,--as-needed \
  -Wl,--no-as-needed,".../libtorch_cuda.so" -Wl,--as-needed \
  ... \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lcuda
```

---

## 5. 目标属性

```cmake
set_target_properties(inference_emb_ops PROPERTIES
    PREFIX ""                    # 不添加 "lib" 前缀
    OUTPUT_NAME "inference_emb_ops"
    CUDA_SEPARABLE_COMPILATION OFF
)
```

---

## 6. Python 侧加载与注册

### 6.1 加载方式

**方式 A: 直接加载 (demo.py)**
```python
import torch
path = os.path.abspath("corelib/dynamicemb/torch_binding_build/inference_emb_ops.so")
torch.ops.load_library(path)
```

**方式 B: 环境变量加载 (HSTU exportable_embedding.py)**
```python
import os, torch
_DYNAMICEMB_OPS_LIB_DIR = os.getenv("DYNAMICEMB_OPS_LIB_DIR", "")
lib_path = os.path.join(_DYNAMICEMB_OPS_LIB_DIR, "inference_emb_ops.so")
torch.ops.load_library(lib_path)
```

设置环境变量：
```bash
export DYNAMICEMB_OPS_LIB_DIR=$(realpath corelib/dynamicemb/torch_binding_build)
```

### 6.2 Fake/Meta 注册 (用于 torch.export tracing)

加载 `.so` 后，还需导入以下 Python 模块注册 fake kernel：

| 模块 | 注册算子 |
|------|---------|
| `dynamicemb.lookup_meta` | `INFERENCE_EMB::table_lookup` 的 `register_fake` |
| `dynamicemb.index_range_meta` | `INFERENCE_EMB::get_table_range` 的 `register_fake` |

`expand_table_ids` 的 Meta dispatch 在 C++ 侧通过 `TORCH_LIBRARY_IMPL(INFERENCE_EMB, Meta, m)` 注册。

### 6.3 使用方 (exportable_tables.py)

`InferenceEmbeddingCollection.forward()` 调用链：
1. `torch.ops.INFERENCE_EMB.expand_table_ids()` — 为每个 key 分配 table_id
2. `torch.ops.INFERENCE_EMB.table_lookup()` — 哈希表查找，返回 indices
3. `self.nve_embedding_(global_indices)` — NVE embedding lookup

---

## 7. 文件完整目录结构

```
corelib/dynamicemb/
├── CMakeLists.txt                          # 构建入口
├── src/
│   ├── check.h                             # CUDA 错误检查宏
│   ├── utils.h                             # 基础类型、dispatch 宏
│   ├── utils.cpp                           # DeviceProp 实现 [编译]
│   ├── torch_utils.h                       # Torch 类型映射
│   ├── torch_utils.cu                      # Torch 工具函数实现 [编译]
│   ├── index_calculation.h                 # CUB 模板、声明
│   ├── index_calculation.cu                # 索引计算实现 [编译]
│   ├── unique_op.h                         # unique 操作声明
│   ├── unique_op.cu                        # unique 操作实现 [编译]
│   ├── lookup_forward.h                    # 前向 lookup 声明
│   ├── lookup_forward.cu                   # 前向 lookup 实现 [编译]
│   ├── lookup_kernel.cuh                   # gather/scatter kernel
│   └── table_operation/
│       ├── score.cuh                       # Score 类型和策略
│       ├── types.cuh                       # LinearBucket, LinearBucketTable
│       ├── table.cuh                       # (包含 types.cuh)
│       ├── kernels.cuh                     # 所有哈希表 kernel
│       ├── lookup.cu                       # lookup 核心实现 [编译]
│       ├── lookup_torch_binding.cu         # table_lookup Torch 绑定 [编译]
│       ├── get_table_range_torch_binding.cu # get_table_range Torch 绑定 [编译]
│       └── expand_table_ids_torch_binding.cu # expand_table_ids Torch 绑定 [编译]
└── torch_binding_build/                    # 构建输出目录
    ├── inference_emb_ops.so                # 最终产物
    ├── Makefile                            # CMake 生成的 Makefile
    ├── CMakeCache.txt                      # CMake 缓存
    └── CMakeFiles/
        └── inference_emb_ops.dir/
            ├── DependInfo.cmake            # 依赖信息
            ├── link.txt                    # 链接命令
            ├── flags.make                  # 编译标志
            ├── cmake_clean.cmake           # clean 规则
            └── src/                        # .o 目标文件
```

---

## 8. 平台特殊适配

### 8.1 PPU (HGGC) SDK 兼容

在 `src/table_operation/types.cuh` 中：
```cpp
// On PPU (HGGC) SDK, __cvta_generic_to_shared is already declared with C++
// linkage in hgrt/hggc_device_functions.h (force-included via command line).
// Adding extern "C" here would cause a linkage conflict.
// On standard NVIDIA CUDA, the manual extern "C" declaration is still needed.
#if !defined(___HGGC_DEVICE_FUNCTIONS_H___)
extern "C" __device__ size_t __cvta_generic_to_shared(const void *);
#endif
```

当前构建环境使用 PPU SDK (`/usr/local/PPU_SDK/CUDA_SDK/`)。

### 8.2 DEMB_USE_PYBIND11 条件编译

所有源文件中的 pybind11 绑定代码均被 `#ifdef DEMB_USE_PYBIND11` 保护。`inference_emb_ops.so` 的构建**不定义** `DEMB_USE_PYBIND11`，因此 pybind11 模块绑定代码被跳过，仅使用 `TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL` 注册算子。

---

## 9. Install 规则

```cmake
install(TARGETS inference_emb_ops
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}   # lib/
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}   # bin/
)
```

---

## 10. 快速验证

```bash
# 从仓库根目录
python3 -c "
import os, torch
path = os.path.abspath('corelib/dynamicemb/torch_binding_build/inference_emb_ops.so')
torch.ops.load_library(path)
print('loaded', path)
print('Available ops:', [op for op in dir(torch.ops.INFERENCE_EMB)])
"
```

预期输出：
```
loaded /path/to/inference_emb_ops.so
Available ops: ['expand_table_ids', 'get_table_range', 'table_lookup']
```
