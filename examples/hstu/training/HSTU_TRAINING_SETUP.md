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

- `fbgemm_gpu`, `torchrec`, `megatron-core`, `gin-config`, `iopath`,
  `nvtx`, `pandas`, `torchmetrics==1.0.3`, `torchx`

### 需要手动安装的包

| 包 | 来源 | 安装方式 |
|---|---|---|
| `dynamicemb` | `corelib/dynamicemb/` | `pip install . --no-build-isolation` |
| `hstu_cuda_ops` | `examples/commons/` | 编译 C++ CUDA 扩展，手动复制 `.so` |

> **注意**: `examples/commons/setup.py` 同时构建 `hstu_cuda_ops` 和 `paged_kvcache_ops`，
> 后者依赖 `nvcomp_static` 库。如果没有该库，`paged_kvcache_ops` 编译会失败，但 `hstu_cuda_ops`
> 不受影响，可正常编译并手动安装。

---

## 2. 代码补丁（共 3 个文件）

### 2.1 `examples/hstu/modules/hstu_attention.py`

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

### 2.2 `examples/hstu/ops/fused_hstu_op.py`

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

### 2.3 `examples/hstu/training/trainer/utils.py`

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

## 3. 配置修改（1 个文件）

### `examples/hstu/training/configs/movielen_retrieval.gin`

添加 `kernel_backend` 配置，使用纯 PyTorch 后端（无需 FBGEMM HSTU CUDA kernel）：

```diff
 NetworkArgs.is_causal = True
+NetworkArgs.kernel_backend = "pytorch"
```

> 如需使用 CUTLASS 后端（需要 A100/H100 等 NVIDIA GPU + 编译 FBGEMM HSTU），
> 将值改为 `"cutlass"` 并确保 `hstu` 包已正确安装。

---

## 4. 数据准备

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

---

## 5. 启动训练

```bash
cd <repo-root>/examples/hstu
PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
  torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
  ./training/pretrain_gr_retrieval.py \
  --gin-config-file ./training/configs/movielen_retrieval.gin
```

---

## 6. 问题排查记录

| # | 错误 | 原因 | 解决方案 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'dynamicemb'` | dynamicemb 未安装 | `cd corelib/dynamicemb && pip install . --no-build-isolation` |
| 2 | `ModuleNotFoundError: No module named 'hstu'` | hstu 顶层导入失败 | 改为懒加载（补丁 2.1） |
| 3 | `ModuleNotFoundError: No module named 'hstu.hstu_ops_gpu'` | hstu 顶层导入失败 | try/except 包裹（补丁 2.2） |
| 4 | `ModuleNotFoundError: No module named 'hstu_cuda_ops'` | commons CUDA ops 未编译 | 编译并安装 hstu_cuda_ops |
| 5 | `FileNotFoundError: 'tmp_data//ml-1m/processed_seqs.csv'` | 数据路径未找到 | 创建 tmp_data 符号链接 |
| 6 | `AssertionError: num_contextuals must be an int when kernel backend is triton` | FUSED layer 走 CUTLASS 路径 | pytorch 后端使用 DEBUG layer type（补丁 2.3） |
| 7 | `pip install .` 在 dynamicemb 构建时找不到 torch | pip build isolation 隔离了环境 | 使用 `--no-build-isolation` |
