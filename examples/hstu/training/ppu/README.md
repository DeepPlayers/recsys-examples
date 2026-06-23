# HSTU PPU Benchmark — 一键部署与测评

> PPU-ZW810E 环境下 HSTU E2E Training 性能基准测试

## 快速开始

### 1. 一键环境部署

```bash
# 全量部署（apt + pip + C++ 库 + HSTU 编译）
bash examples/hstu/training/ppu/scripts/setup.sh

# 仅编译 HSTU 内核（跳过系统环境）
bash examples/hstu/training/ppu/scripts/setup.sh --skip-pod
```

### 2. 一键测评

```bash
# 默认 8 卡
bash examples/hstu/training/ppu/scripts/run_benchmark.sh

# 4 卡
bash examples/hstu/training/ppu/scripts/run_benchmark.sh 4
```

测评包含两部分：
- **Option A**: exp2_cutlass 单实验（快速验证 ~232 TFLOPS/GPU）
- **Option B**: 全量 6 实验（渐进优化对比）

### 3. 查看结果

```bash
# PPU vs H20 对比报告
cat examples/hstu/training/ppu/docs/PPU_VS_H20.md
```

## 目录结构

```
ppu/
├── scripts/
│   ├── setup.sh            # 一键环境部署（幂等）
│   └── run_benchmark.sh    # 一键测评
├── docs/
│   ├── PPU_VS_H20.md       # PPU vs H20 性能对比报告
│   ├── E2E_BENCHMARK.md    # 完整 benchmark 说明
│   └── BUILD_CUTLASS_KERNELS.md  # CUTLASS 编译指南
├── logs/                   # 运行日志（自动生成）
└── README.md
```

## 核心性能数据

| 实验 | PPU (8卡) | H20 (16卡) | PPU/H20 |
|------|---:|---:|---:|
| exp0_baseline | 104.78 | 75.6 | **138.6%** ✅ |
| exp1_shuffler | 136.54 | 115.8 | **117.9%** ✅ |
| **exp2_cutlass** | **232.08** | **302.6** | **76.7%** |
| exp5_prefetch | 232.65 | 310.6 | 74.9% |

> PPU 已达硬件理论上限的 **96.5%**（76.7% / 79.6%）

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HSTU_GEMM_BACKEND` | *(auto: cuBLAS)* | `triton` 回退到 Triton GEMM |
| `HSTU_ENABLE_EXTENDED_BW_CONFIGS` | `FALSE` | `TRUE` 启用 CUDA 12.8+ 被禁用的 autotune 配置 |
| `HSTU_FWD_TILE_M/N/NWARPS` | *(默认)* | 覆盖 CUTLASS forward tile size（需重编译） |
| `HSTU_BWD_TILE_M/N/NWARPS` | *(默认)* | 覆盖 CUTLASS backward tile size（需重编译） |
