# HSTU Training Setup Guide

本文档记录了在 PPU 环境下成功运行 HSTU retrieval 训练所做的全部工作，包括依赖安装、代码补丁、配置修改和数据处理。

---

## 1. 环境信息

| 项目 | 值 |
|---|---|
| GPU | PPU-ZW810E (sm8.0, 兼容 Ampere) |
| Python | 3.12 |
| PyTorch | 2.9.0+ali.10.ppu2.0.0.cu129 |
| FBGEMM | 1.4.0+ppu2.0.0.ce (已预装) |
| TorchRec | 1.4.0+ppu2.0.0.ce (已预装) |
| Megatron-Core | 0.15.0 (已预装) |

### 已预装可直接使用的包

- `fbgemm_gpu`, `torchrec`, `megatron-core`, `iopath`,
  `pandas`, `torchmetrics==1.0.3`

> **注意**: `gin-config`, `nvtx`, `torchx` 在部分 PPU 环境中未预装，需要手动安装：
> ```bash
> pip install gin-config nvtx torchx
> ```
> `megatron-core` 在 PPU 镜像源中可能无法通过 `pip install` 安装（版本兼容性检查失败），
> 但环境中可能已预装。请先用 `python3 -c "import megatron.core"` 验证。

### 需要手动安装的包

| 包 | 来源 | 安装方式 |
|---|---|---|
| `dynamicemb` | `corelib/dynamicemb/` | `pip install . --no-build-isolation` |
| `hstu_cuda_ops` | `examples/commons/` | 编译 C++ CUDA 扩展，手动复制 `.so` |

> **注意**: `examples/commons/setup.py` 同时构建 `hstu_cuda_ops` 和 `paged_kvcache_ops`，
> 后者依赖 `nvcomp_static` 库。如果没有该库，`paged_kvcache_ops` 编译会失败，但 `hstu_cuda_ops`
> 不受影响，可正常编译并手动安装。

---

## 2. DynamicEmb 编译说明

### 2.1 问题

`corelib/dynamicemb/setup.py` 在文件顶部直接导入了 `torch`：

```python
# corelib/dynamicemb/setup.py (line 23)
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
```

这导致 `pip install .` 在默认模式下失败——pip 会创建一个隔离的构建环境（build isolation），
该环境中没有 `torch`，于是 `setup.py` 在解析阶段就报 `ModuleNotFoundError: No module named 'torch'`。

### 2.2 解决方案

使用 `--no-build-isolation` 标志，让 pip 直接使用当前环境中已安装的 `torch`：

```bash
cd corelib/dynamicemb
pip install . --no-build-isolation
```

### 2.3 setup.py 的其他行为

`setup.py` 在执行时会：

1. **自动卸载** 已有的 `dynamicemb`（使用 `--break-system-packages` 绕过系统保护）
2. **自动安装** `ordered-set` 依赖（同样使用 `--break-system-packages`）
3. **检查 torchrec 版本** ≥ 1.2.0，版本不满足会直接报错
4. **构建 CUDA 扩展** `dynamicemb_extensions`，支持 sm_75 / sm_80 / sm_90 架构

编译使用 `NinjaBuildExtension`，会自动根据 CPU 核心数和可用内存计算并行编译任务数
（每个任务峰值内存约 12GB，多架构 nvcc 编译时占用较大）。

### 2.4 验证安装

```bash
python3 -c "import dynamicemb; print(dynamicemb.__version__)"
```

---

## 3. 代码补丁（共 3 个文件）

### 3.1 `examples/hstu/modules/hstu_attention.py`

**问题**: 文件顶层 `from hstu import hstu_attn_varlen_func`，当 `hstu` 包未安装时
（例如 PPU 环境未编译 FBGEMM HSTU CUDA kernel），即使选择 `pytorch` 后端也会导入失败。

**修改**: 将该导入从文件顶部移除，改为在 `FusedHSTUAttention.forward()` 内部懒加载。

```diff
 import torch
 from commons.utils.nvtx_op import output_nvtx_hook
 from configs import KernelBackend
-from hstu import hstu_attn_varlen_func
```

在 `FusedHSTUAttention.forward()` 方法内部（使用 `hstu_attn_varlen_func` 之前）添加：

```diff
         if scaling_seqlen == -1:
             scaling_seqlen = max_seqlen

+        from hstu import hstu_attn_varlen_func
+
         return hstu_attn_varlen_func(
```

### 3.2 `examples/hstu/ops/fused_hstu_op.py`

**问题**: 文件顶层无条件导入 `hstu` 和 `hstu.hstu_ops_gpu`，在 `hstu` 包未安装时直接报错。

**修改**: 用 `try/except ImportError` 包裹。

```diff
-from collections import OrderedDict
 from typing import Optional, Tuple, Union

-import hstu  # noqa: F401 – registers torch.ops.fbgemm.*
-import hstu.hstu_ops_gpu  # noqa: F401 – registers fake impls for torch.export
+try:
+    import hstu  # noqa: F401 – registers torch.ops.fbgemm.*
+    import hstu.hstu_ops_gpu  # noqa: F401 – registers fake impls for torch.export
+except ImportError:
+    pass
```

### 3.3 `examples/hstu/training/trainer/utils.py`

**问题**: 当 `tensor_model_parallel_size == 1` 时，`layer_type` 始终设为 `FUSED`。
`FUSED` 模式使用 `FusedHSTULayer`，其内部直接调用 CUTLASS kernel
（`torch.ops.fbgemm.hstu_varlen_fwd_80` 等），即使 `kernel_backend="pytorch"`
也会走 CUTLASS 路径导致运行时报错。

**修改**: 当 `kernel_backend == KernelBackend.PYTORCH` 时，强制使用 `HSTULayerType.DEBUG`，
该模式使用 `DebugHSTULayer`，其 attention 走 `create_hstu_attention()` 工厂函数，
根据 `kernel_backend` 正确分发到 `TorchHSTUAttention`。

```diff
     layer_type = None
-    if tensor_model_parallel_args.tensor_model_parallel_size == 1:
+    if kernel_backend == KernelBackend.PYTORCH:
+        layer_type = HSTULayerType.DEBUG
+    elif tensor_model_parallel_args.tensor_model_parallel_size == 1:
         layer_type = HSTULayerType.FUSED
     else:
         layer_type = HSTULayerType.NATIVE
```

---

## 4. 配置修改（1 个文件）

### `examples/hstu/training/configs/movielen_retrieval.gin`

添加 `kernel_backend` 配置，使用纯 PyTorch 后端（无需 FBGEMM HSTU CUDA kernel）：

```diff
 NetworkArgs.is_causal = True
+NetworkArgs.kernel_backend = "pytorch"
```

> 如需使用 CUTLASS 后端（需要 A100/H100 等 NVIDIA GPU + 编译 FBGEMM HSTU），
> 将值改为 `"cutlass"` 并确保 `hstu` 包已正确安装。

### `log_interval` / `eval_interval` 调整

默认配置中 `log_interval = 100` 和 `eval_interval = 100`，但 ml-1m 数据集
（6040 users / batch 128 ≈ 47 steps/epoch）每 epoch 不到 100 步，导致训练过程
**没有任何日志输出**。建议降低间隔：

```diff
-TrainerArgs.eval_interval = 100
-TrainerArgs.log_interval = 100
+TrainerArgs.eval_interval = 20
+TrainerArgs.log_interval = 10
```

---

## 5. 数据准备

MovieLens-1M 数据集已预处理并存放于 `examples/commons/tmp_data/ml-1m/`。

训练脚本默认从 CWD 下的 `tmp_data/` 相对路径查找数据。因此需要创建符号链接：

```bash
ln -sf <repo-root>/examples/commons/tmp_data <repo-root>/examples/hstu/tmp_data
```

如需重新预处理数据：

```bash
cd <repo-root>/examples/commons
mkdir -p ./tmp_data
python3 ./hstu_data_preprocessor.py --dataset_name ml-1m
```

> **已知问题**: `hstu_data_preprocessor.py` 内部使用 `urlretrieve` 下载 ml-1m.zip，
> 在网络不稳定时可能产生不完整的 zip 文件，导致 `zipfile.BadZipFile: File is not a zip file` 错误。
> 解决方案：手动使用 `curl` 下载后再运行预处理器：
> ```bash
> cd <repo-root>/examples/commons
> mkdir -p ./tmp_data
> rm -f ./tmp_data/movielens1m.zip
> curl -L -o ./tmp_data/movielens1m.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip
> python3 ./hstu_data_preprocessor.py --dataset_name ml-1m
> ```

---

## 6. 启动训练

```bash
cd <repo-root>/examples/hstu
PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
  torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
  ./training/pretrain_gr_retrieval.py \
  --gin-config-file ./training/configs/movielen_retrieval.gin
```

---

## 7. 问题排查记录

| # | 错误 | 原因 | 解决方案 |
|---|---|---|---|
| 1 | `pip install .` 构建 dynamicemb 时报 `ModuleNotFoundError: No module named 'torch'` | `setup.py` 顶层导入了 `torch`，pip 默认 build isolation 环境中无 torch | 使用 `pip install . --no-build-isolation`（见 §2） |
| 2 | `ModuleNotFoundError: No module named 'dynamicemb'` | dynamicemb 未安装 | `cd corelib/dynamicemb && pip install . --no-build-isolation` |
| 3 | `ModuleNotFoundError: No module named 'hstu'` | hstu 顶层导入失败 | 改为懒加载（补丁 3.1） |
| 4 | `ModuleNotFoundError: No module named 'hstu.hstu_ops_gpu'` | hstu 顶层导入失败 | try/except 包裹（补丁 3.2） |
| 5 | `ModuleNotFoundError: No module named 'hstu_cuda_ops'` | commons CUDA ops 未编译 | 编译并安装 hstu_cuda_ops |
| 6 | `FileNotFoundError: 'tmp_data//ml-1m/processed_seqs.csv'` | 数据路径未找到 | 创建 tmp_data 符号链接 |
| 7 | `AssertionError: num_contextuals must be an int when kernel backend is triton` | FUSED layer 走 CUTLASS 路径 | pytorch 后端使用 DEBUG layer type（补丁 3.3） |
| 8 | `ModuleNotFoundError: No module named 'gin'` | gin-config 未预装 | `pip install gin-config nvtx torchx` |
| 9 | `zipfile.BadZipFile: File is not a zip file` | 预处理器下载的 ml-1m.zip 不完整 | 用 `curl -L` 手动下载后重新预处理（见 §5） |
| 10 | 训练无输出（静默完成） | `log_interval=100` 大于 ml-1m 每 epoch 步数(~47) | 降低 `log_interval` 和 `eval_interval`（见 §4） |

---

## 8. Dev 分支优化总结

在 PPU 环境部署过程中，我们对 dev 分支的改动进行了全面审查和优化，确保只保留必要的 PPU 适配改动，移除无关变更和功能性回退。

### 8.1 清理内容

#### 构建产物（52 个文件，172K+ 行）
- **移除**: `corelib/dynamicemb/torch_binding_build/` 整个目录
- **原因**: CMake 构建产物（`.o`, `CMakeCache.txt`, `Makefile` 等）不应入库
- **措施**: 添加到 `.gitignore`，防止再次提交

#### 临时文件和无关新增
- **移除**: `demo.py`（临时调试脚本）
- **移除**: `inference_emb_ops_build_changes.md`（推理侧编译笔记，与训练无关）
- **移除**: `examples/hstu/env.sh`（环境构建脚本，应作为 wiki 而非代码）
- **移除**: `examples/hstu/Megatron-LM`（submodule 引用残留）

#### Benchmark 改动
- **回退**: `examples/hstu/training/benchmark/` 目录下所有改动
- **回退**: `third_party/FBGEMM` submodule 指针更新
- **原因**: 这些是独立的功能迭代，与 PPU 适配无关

### 8.2 回退的功能性改动

以下改动不是 PPU 必须的，属于功能回退，已全部恢复为 main 分支版本：

| 文件 | 回退内容 | 影响 |
|------|---------|------|
| `corelib/dynamicemb/dynamicemb/batched_dynamicemb_function.py` | 恢复 `_copy_cuda_tensor_to_pinned_cpu`、`_scalar_item`、`_bool_item` 辅助函数 | 恢复 pinned memory 优化，提升 D2H 拷贝性能 |
| `examples/commons/datasets/hstu_batch.py` | 恢复 `HSTUBatch.slice()` 方法（106 行） | 恢复 batch 切片功能，被 `hstu_random_dataset.py` 和 `test_utils.py` 使用 |
| `examples/commons/datasets/hstu_random_dataset.py` | 恢复使用 `batch.slice()` 的实现 | 与 `hstu_batch.py` 联动 |
| `examples/commons/distributed/batch_shuffler.py` | 恢复 `tensor_from_cpu_array_like` / `tensor_to_cpu_list` 调用 | 恢复自定义 tensor transfer 工具 |
| `examples/commons/perf_model/partitioner.py` | 恢复 `kk_cpu_ops` C++ 加速器加载逻辑 | 恢复 KK 分区 C++ 加速，释放 GIL 提升并发 |
| `examples/commons/setup.py` | 恢复 `kk_cpu_ops` CppExtension 和 `BUILD_EXT_ONLY` 过滤机制 | 恢复完整构建能力 |
| `examples/commons/utils/perf.py` | 恢复 H100 peak TFLOPS 为 989，`cal_hstu_flops` 使用 all_reduce | 恢复原始 FLOPS 统计逻辑 |
| `examples/hstu/utils/gin_config_args.py` | 恢复 `DynamicEmbeddingArgs.dist_type` 字段 | 恢复 row-wise sharding 输入分布策略配置 |
| `examples/hstu/training/trainer/training.py` | 恢复 `_warm_up_data_parallel_collective()` 函数及调用 | 恢复多卡训练 warmup collective，提升稳定性 |
| `examples/hstu/test_utils.py` | 恢复使用 `batch.slice()` 的实现 | 与 `hstu_batch.py` 联动 |

### 8.3 保留的 PPU 适配改动

以下改动是 PPU 环境必须的，已保留：

| 文件 | 改动 | 必要性 |
|------|------|--------|
| `corelib/dynamicemb/src/table_operation/types.cuh` | `__cvta_generic_to_shared` 声明加 `#if !defined(___HGGC_DEVICE_FUNCTIONS_H___)` 保护 | PPU SDK 强制 include 同名声明，避免 linkage 冲突 |
| `corelib/dynamicemb/src/table_operation/kernels.cuh` | `bucket.probe<>()` → `bucket.template probe<>()`；`pred.template operator()` → `pred()` | PPU nvcc 对 dependent name 解析更严格 |
| `examples/hstu/modules/hstu_attention.py` | 顶层 `from hstu import` 改为 `forward()` 内懒加载 | PPU 无 FBGEMM HSTU kernel，避免导入失败 |
| `examples/hstu/ops/fused_hstu_op.py` | `import hstu` / `import hstu.hstu_ops_gpu` 用 `try/except` 包裹 | PPU 无 hstu 包，避免导入失败 |
| `examples/hstu/training/trainer/utils.py` | `kernel_backend == PYTORCH` 时强制 `DEBUG` layer type | PPU 无 CUTLASS kernel，避免运行时报错 |
| `examples/hstu/training/configs/movielen_retrieval.gin` | `kernel_backend = "pytorch"`；`log_interval = 10`；`eval_interval = 20` | PPU 使用 pytorch 后端；降低日志间隔提升可见性 |
| `corelib/hstu/setup.py` | `subprocess.check_output("git rev-parse HEAD")` 加 `try/except` | PPU 容器无 git 历史，避免 CalledProcessError |
| `examples/commons/utils/initialize.py` | 删除 `initialize_distributed()` 中的 print rank 日志 | 简化日志输出，非必须但无害 |

### 8.4 配置文件调整

| 文件 | 调整 | 原因 |
|------|------|------|
| `examples/hstu/training/configs/movielen_retrieval.gin` | `dataset_name` 恢复为 `'ml-20m'` | ml-1m 仅用于快速验证，ml-20m 是标准训练集 |
| `examples/hstu/training/trainer/utils.py` | 恢复 `dist_type=embedding_args.dist_type` 参数传递 | 与 `gin_config_args.py` 联动，恢复完整配置 |

### 8.5 优化效果

优化后的 dev 分支改动从 **88 个文件（174K+ 行）** 精简到 **约 20 个文件（~500 行）**：

| 类别 | 优化前 | 优化后 | 减少 |
|------|--------|--------|------|
| 构建产物 | 52 文件 | 0 文件 | -52 |
| 临时文件 | 4 文件 | 0 文件 | -4 |
| Benchmark 改动 | ~15 文件 | 0 文件 | -15 |
| 功能性回退 | 10 文件 | 0 文件 | -10 |
| PPU 适配 | 7 文件 | 7 文件 | 0 |
| 配置调整 | 2 文件 | 2 文件 | 0 |

**最终改动**仅包含：
- PPU 环境适配代码（7 个文件）
- 部署文档和脚本（2 个文件：`HSTU_TRAINING_SETUP.md`、`setup_and_train.sh`）
- 配置文件调整（2 个文件：gin config、utils.py）
- `.gitignore` 更新（防止构建产物再次入库）

### 8.6 后续建议

1. **验证训练**: 在 PPU 环境使用 ml-20m 数据集运行完整训练，确认 PPU 适配改动有效
2. **合入 main**: 审查通过后可将 dev 分支合入 main，保留 PPU 适配能力
3. **监控构建**: 确保 `.gitignore` 生效，`torch_binding_build/` 不再被意外提交
