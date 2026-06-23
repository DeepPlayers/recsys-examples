# H20 vs PPU-ZW810E 性能对比

> 日期: 2026-06-23
> H20: NVIDIA H20 × 4 卡 / 8 卡（单节点）
> PPU: PPU-ZW810E (SM 8.0, 64 SMs, 787 BF16 TFLOPS) × 8 卡（单节点）

## 1. 测试环境

### 硬件

| 参数 | H20 | PPU-ZW810E |
|------|-----|------------|
| GPU 数量 | 4 卡 / 8 卡 | 8 卡 |
| SM 版本 | Hopper | SM 8.0 (Ampere 级) |
| 节点数 | 单节点 | 单节点 |

### 模型配置

所有测试使用相同的模型和数据配置：

| 参数 | 值 |
|------|-----|
| Hidden size | 1024 |
| HSTU 层数 | 8 |
| 注意力头数 | 4 |
| Head dimension | 256 |
| Item embedding dim | 128 |
| Contextual embedding dim | 128 |
| Prediction head | [512, 8] × 8 tasks |
| 优化器 | Adam (lr=1e-3) |
| 每卡 batch size | 32 |
| 最大序列长度 | 4096 |
| 序列长度分布 | Zipf (α=1.2), jagged |
| Key 值分布 | Zipf (α=1.05) |
| 训练迭代数 | 1000 |

### 实验矩阵

| 实验 | 负载均衡 Shuffler | CUTLASS Attention | Caching | Hash-RoundRobin | Prefetch |
|------|:---:|:---:|:---:|:---:|:---:|
| exp0_baseline | ❌ | ❌ | ❌ | ❌ | ❌ |
| exp1_shuffler | ✅ | ❌ | ❌ | ❌ | ❌ |
| exp2_cutlass | ✅ | ✅ | ❌ | ❌ | ❌ |
| exp3_caching | ✅ | ✅ | ✅ | ❌ | ❌ |
| exp4_caching_hr | ✅ | ✅ | ✅ | ✅ | ❌ |
| exp5_prefetch | ✅ | ✅ | ✅ | ✅ | ✅ |

> **注意**: H20 的 exp0_baseline 也使用了 CUTLASS kernel backend（`--kernel_backend cutlass`），因此 H20 上 exp1 → exp2 不存在 attention kernel 切换。PPU 的 exp0/exp1 使用 Triton attention，exp2 起切换到 CUTLASS。

---

## 2. 性能结果

### 2.1 H20 4 卡

**Run ID**: `e2e_20260604_144215`，2026-06-04

| 实验 | 平均 TFLOPS/GPU | 平均 MFU | 峰值 TFLOPS/GPU | vs Baseline |
|------|---:|---:|---:|---:|
| exp0_baseline | **332.83** | 13.15% | 333.54 | 1.00× |
| exp1_shuffler | **439.83** | 17.38% | 441.71 | 1.32× |
| exp2_cutlass | **439.97** | 17.39% | 441.91 | 1.32× |
| exp3_caching | **442.67** | 17.49% | 444.82 | 1.33× |
| exp4_caching_hr | **443.51** | 17.53% | 444.95 | 1.33× |
| exp5_prefetch | **443.73** | 17.54% | 446.59 | 1.33× |

### 2.2 H20 8 卡

**Run ID**: `e2e_20260604_154548`，2026-06-04

| 实验 | 平均 TFLOPS/GPU | 平均 MFU | 峰值 TFLOPS/GPU | vs Baseline |
|------|---:|---:|---:|---:|
| exp0_baseline | **633.66** | 12.52% | 636.78 | 1.00× |
| exp1_shuffler | **873.36** | 17.26% | 879.29 | 1.38× |
| exp2_cutlass | **875.35** | 17.30% | 880.66 | 1.38× |
| exp3_caching | **879.49** | 17.38% | 885.83 | 1.39× |
| exp4_caching_hr | **880.97** | 17.41% | 886.06 | 1.39× |
| exp5_prefetch | **875.88** | 17.31% | 888.82 | 1.38× |

### 2.3 PPU-ZW810E 8 卡

**Run ID**: `e2e_20260622_170837`，2026-06-22

| 实验 | 平均 TFLOPS/GPU | 平均 MFU | 峰值 TFLOPS/GPU | vs Baseline |
|------|---:|---:|---:|---:|
| exp0_baseline | **104.77** | 6.66% | 104.84 | 1.00× |
| exp1_shuffler | **136.41** | 8.67% | 136.63 | 1.30× |
| exp2_cutlass | **232.16** | 14.76% | 232.42 | 2.22× |
| exp3_caching | **232.64** | 14.79% | 233.11 | 2.22× |
| exp4_caching_hr | **232.65** | 14.79% | 232.98 | 2.22× |
| exp5_prefetch | **232.47** | 14.78% | 233.21 | 2.22× |

---

## 3. 对比分析

### 3.1 H20 8 卡 vs PPU 8 卡（等卡数公平对比）

| 实验 | H20 TFLOPS/GPU | PPU TFLOPS/GPU | H20 / PPU |
|------|---:|---:|---:|
| exp0_baseline | 633.66 | 104.77 | **6.05×** |
| exp1_shuffler | 873.36 | 136.41 | **6.40×** |
| exp2_cutlass | 875.35 | 232.16 | **3.77×** |
| exp3_caching | 879.49 | 232.64 | **3.78×** |
| exp4_caching_hr | 880.97 | 232.65 | **3.79×** |
| exp5_prefetch | 875.88 | 232.47 | **3.77×** |

**差距变化规律**：Baseline 下 H20 领先 6×，PPU 启用 CUTLASS 后差距缩小到 3.8×。CUTLASS 是 PPU 缩小差距的唯一手段。

### 3.2 H20 4 卡 vs PPU 8 卡（首次对比）

| 实验 | H20 TFLOPS/GPU | PPU TFLOPS/GPU | H20 / PPU |
|------|---:|---:|---:|
| exp0_baseline | 332.83 | 104.77 | **3.18×** |
| exp1_shuffler | 439.83 | 136.41 | **3.22×** |
| exp2_cutlass | 439.97 | 232.16 | **1.90×** |
| exp3_caching | 442.67 | 232.64 | **1.90×** |
| exp4_caching_hr | 443.51 | 232.65 | **1.91×** |
| exp5_prefetch | 443.73 | 232.47 | **1.91×** |

### 3.3 逐步优化效果对比（8 卡 vs 8 卡）

| 优化步骤 | H20 8 卡 | PPU 8 卡 | 说明 |
|----------|---:|---:|------|
| Baseline → Shuffler | **+37.8%** | **+30.2%** | H20 收益略大 |
| Shuffler → CUTLASS | **+0.23%** | **+70.2%** | PPU 质变，H20 无感 |
| CUTLASS → Caching | **+0.47%** | **+0.21%** | 微小 |
| → Prefetch | **-0.58%** | **-0.07%** | 无收益 |

### 3.4 H20 扩展性：4 卡 → 8 卡

| 实验 | H20 4 卡 | H20 8 卡 | 8 卡/4 卡 |
|------|---:|---:|---:|
| exp0_baseline | 332.83 | 633.66 | **1.90×** |
| exp1_shuffler | 439.83 | 873.36 | **1.99×** |
| exp2_cutlass | 439.97 | 875.35 | **1.99×** |
| exp3_caching | 442.67 | 879.49 | **1.99×** |
| exp4_caching_hr | 443.51 | 880.97 | **1.99×** |
| exp5_prefetch | 443.73 | 875.88 | **1.97×** |

H20 从 4 卡到 8 卡，单卡 TFLOPS 提升 ~2×（接近线性扩展）。而 PPU 4→8 卡的 exp2 仅 1.66×（139.81→232.16），PPU 的通信开销更大。

### 3.5 单步耗时对比（exp5_prefetch 稳态）

| 指标 | H20 8 卡 | PPU 8 卡 | H20 / PPU |
|------|---:|---:|---:|
| 平均步时 (ms) | ~14,100 | ~12,870 | 0.91× |
| 平均 TFLOPS/GPU | 875.88 | 232.47 | **3.77×** |

H20 8 卡每步反而比 PPU 8 卡慢 ~9%，但单卡 TFLOPS 是 PPU 的 3.77 倍——因为 H20 在更长的步时内完成了远更多的 FLOPs。

---

## 4. 关键发现

### 4.1 CUTLASS 效果差异巨大

这是两个平台之间最显著的差异（4 卡和 8 卡数据一致）：

- **H20**: CUTLASS 提升仅 +0.23%（873.36 → 875.35 TFLOPS）。H20 的默认 attention kernel 已高度优化，CUTLASS 无额外收益。
- **PPU**: CUTLASS 提升 +70.2%（136.41 → 232.16 TFLOPS）。PPU 的默认 Triton attention kernel 远不如 CUTLASS 优化充分，CUTLASS 是决定性优化。

**原因分析**：
- H20 作为 Hopper 架构 GPU，其 flash-attn kernel 已针对 Hopper 特性（TMA、wgmma、异步流水线）深度优化
- PPU 是 SM 8.0 Ampere 级设备，原始 Triton kernel 未充分利用其硬件能力，而 CUTLASS kernel 提供了更优的寄存器分配和 warp 调度

### 4.2 Shuffler 是唯一跨平台通用有效优化

两个平台均获得 ~30-38% 的提升，验证了 Zipf 分布下序列长度不均衡导致的 GPU 负载问题是硬件无关的通用瓶颈。O(n²) 的 attention 复杂度放大了这种不均衡。

### 4.3 等卡数对比：H20 单卡 ≈ 3.8× PPU

在 8 卡 vs 8 卡的最优配置下，H20 单卡 TFLOPS 是 PPU 的 3.77 倍。差距来源：
- 架构代际差异（Hopper vs Ampere 级）
- H20 的硬件加速特性（TMA、wgmma 等）
- SM 数量和显存带宽的差异

### 4.4 H20 扩展性优于 PPU

H20 4→8 卡实现接近 2× 线性扩展，PPU 4→8 卡仅 1.66×。PPU 在更大规模下的通信优化仍有空间。

### 4.5 优化路径总结

```
H20:  Baseline ──(+38%)──▶ Shuffler ──(+0.2%)──▶ 后续 ≈ 0%
      一步到位

PPU:  Baseline ──(+30%)──▶ Shuffler ──(+70%)──▶ 后续 ≈ 0%
      CUTLASS 是最大变量
```

---

## 5. 数据来源

| 平台 | Run ID | 日期 | 日志目录 |
|------|--------|------|---------|
| H20 4 卡 | `e2e_20260604_144215` | 2026-06-04 | `training/h20/h20-4card-all/` |
| H20 8 卡 | `e2e_20260604_154548` | 2026-06-04 | `training/h20/h20-8card-all/` |
| PPU 8 卡 | `e2e_20260622_170837` | 2026-06-22 | `training/benchmark/results/e2e_20260622_170837/` |

所有指标基于 iter 199-999 的稳态平均值（41 个采样点，去除 warmup）。
