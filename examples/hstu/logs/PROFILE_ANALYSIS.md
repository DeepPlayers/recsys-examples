# HSTU E2E Benchmark - Profile Analysis

## Hardware & Environment
- **GPU**: 4× PPU-ZW810E (sm 8.0, 98 GB each)
- **Software**: torch 2.9.0+ali.10.ppu2.0.0.cu129, torchrec 1.4.0+ppu2.0.0.ce
- **Model**: HSTU (8 layers, 1024 hidden, 4 heads, 256 dim/head)
- **Data**: 50M item/user_id embedding tables, Zipf distribution
- **Profiler**: asys profile (system-wide, hggc+hgtx+acblas+acdnn+osrt)

## Summary

| Exp | Name | TFLOPS | MFU | Speedup |
|-----|------|-------:|----:|--------:|
| 0 | Baseline (Triton) | 51.36 | 6.53% | 1.00× |
| 1 | +Shuffler | 64.79 | 8.24% | 1.26× |
| 2 | **+CUTLASS** | **134.37** | **17.09%** | **2.62×** |
| 3 | +Caching | 134.51 | 17.10% | 2.62× |
| 4 | +Hash-RoundRobin | 134.52 | 17.11% | 2.62× |
| 5 | +Prefetch | 134.86 | 17.15% | 2.63× |

## GPU Kernel Time Breakdown

| Category | exp0 | exp1 | exp2 | exp3 | exp4 | exp5 |
|----------|-----:|-----:|-----:|-----:|-----:|-----:|
| Attention (fwd+bwd) | 0.0% | 0.0% | **74.6%** | 75.2% | 75.0% | 73.7% |
| GEMM/MatMul | 1.8% | 2.5% | 7.2% | 7.3% | 7.2% | 7.1% |
| LayerNorm | 0.3% | 0.4% | 1.2% | 1.2% | 1.2% | 1.2% |
| Embedding/Scatter | 0.3% | 0.5% | 1.3% | 1.3% | 1.3% | 1.3% |
| PCCL/NCCL | **28.7%** | 5.7% | 5.2% | 4.8% | 4.9% | 6.6% |
| Elementwise | 0.6% | 0.8% | 2.4% | 2.4% | 2.4% | 2.3% |
| Other (Triton kernels) | **68.3%** | **90.1%** | 8.1% | 7.9% | 7.9% | 7.8% |

## Top 3 Kernels Per Experiment

### exp0_baseline (Triton attention, no shuffler)
1. **57.3%** `_hstu_attn_bwd` (Triton backward) — 1046.75s
2. **28.5%** `pcclKernel_SendRecv` (PCCL P2P) — 521.06s
3. **8.8%** `_hstu_attn_fwd` (Triton forward) — 160.29s

### exp1_shuffler (Triton + balanced shuffler)
1. **75.2%** `_hstu_attn_bwd` — 1011.04s
2. **11.9%** `_hstu_attn_fwd` — 160.06s
3. **5.5%** `pcclKernel_SendRecv` — 73.97s

### exp2_cutlass (CUTLASS attention)
1. **57.8%** `hstu_bwd_compute_dq_dk_dv_kernel` (CUTLASS backward) — 264.54s
2. **16.6%** `hstu_fwd_kernel` (CUTLASS forward) — 76.10s
3. **7.5%** `_addmm_optional_silu_fwd` (FFN) — 34.13s

### exp3–exp5 (caching / hash-roundrobin / prefetch)
Nearly identical kernel profiles to exp2_cutlass, with minor variations in
PCCL/NCCL time due to different embedding sharding strategies.

## Key Findings

### 1. CUTLASS Attention: The Dominant Optimization
- CUTLASS backward kernel is **4× faster** than Triton backward: 264s vs 1047s
- CUTLASS forward kernel is **2× faster** than Triton forward: 76s vs 160s
- After CUTLASS, attention consumes 74-75% of GPU time — it's now compute-bound

### 2. Workload-Balanced Shuffler Eliminates P2P Overhead
- Baseline (exp0): PCCL P2P communication takes **28.7%** of GPU time (523s)
- With shuffler (exp1): PCCL drops to **5.7%** (77s) — **6.8× reduction**
- The shuffler redistributes variable-length sequences, eliminating load imbalance

### 3. Caching / Hash-RoundRobin / Prefetch: Minimal Impact on Kernel Profile
- These optimizations change the **embedding access pattern** (host↔HBM), not
  the GPU compute kernel profile
- PCCL/NCCL time remains ~5% across exp2–exp5
- The embedding/scatter category stays at ~1.3% — lookup is not a bottleneck

### 4. Memory Operations Are Negligible
- Total memory operation time: ~118 ms across all experiments
- `cudaStreamSynchronize` dominates API time (88.8%) — GPU is well-utilized
- No memory allocation bottleneck detected

## Bottleneck Analysis (exp2_cutlass)

```
GPU Time Distribution:
┌─────────────────────────────────────────────────────┐
│ Attention backward  █████████████████████████  57.8% │
│ Attention forward   ████████                 16.6%  │
│ FFN (addmm+silu)    ████                       7.5%  │
│ GEMM                ████                       7.2%  │
│ PCCL/NCCL           ██                         5.2%  │
│ Elementwise         █                          2.4%  │
│ Other               ████                       8.1%  │
└─────────────────────────────────────────────────────┘
```

**Primary bottleneck**: HSTU attention backward kernel (57.8%)
- This kernel is already using CUTLASS with hand-tuned tile sizes
- Further optimization requires PPU-specific kernel tuning

**Secondary opportunities**:
- FFN/GEMM (14.7% combined): Could benefit from PPU-optimized GEMM libraries
- PCCL/NCCL (5.2%): Already low; prefetch pipeline provides marginal improvement

## Comparison: PPU-ZW810E vs H100 (exp2_cutlass)

| Metric | H100 (16 GPU) | PPU-ZW810E (4 GPU) | Ratio |
|--------|:---:|:---:|:---:|
| Avg TFLOPS/GPU | 302.6 | 134.4 | 0.44× |
| Avg MFU | 30.59% | 17.09% | 0.56× |
| Attention fwd time % | ~60% | 74.6% | — |
| PCCL/NCCL % | ~3% | 5.2% | — |

The PPU achieves 44% of H100 throughput per GPU. The gap is primarily in:
1. CUTLASS kernel tuning (originally optimized for NVIDIA GPUs)
2. Lower memory bandwidth on PPU-ZW810E vs H100-SXM5
3. PPU-specific instruction scheduling differences

## Files
- Profile archives: `oss://ziwei-shanghai/hstu-benchmark/ppu-zw810e-20260608/profiles/`
- CSV stats: `examples/hstu/logs/*_stats.csv_*.csv`
- Training logs: `examples/hstu/training/benchmark/results/e2e_20260608_193033..200106/`
