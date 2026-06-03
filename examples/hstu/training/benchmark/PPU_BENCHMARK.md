# HSTU E2E Training Benchmark — PPU-ZW810E Results

## Hardware

- **GPU**: 4× PPU-ZW810E (98 GB HBM each)
- **Node**: Single node, 4 GPUs
- **Driver**: PPU-SMI 1.22, HGGC 13.0
- **SM**: 8.0 (Ampere-class)

## Software

- **PyTorch**: 2.9.0+ali.10.ppu2.0.0.cu129
- **FBGEMM HSTU**: compiled from `jiayus-nvidia/FBGEMM` fork (sm 8.0, Ampere kernel path)
- **hstu_attn**: compiled from `corelib/hstu` (CUTLASS attention kernels)
- **DynamicEmb**: local build
- **flash-attn**: 2.7.4.post1+ppu2.0.0.oe

## Run Info

- **Run ID**: `e2e_20260603_194209`
- **Date**: 2026-06-03
- **Experiment**: `exp2_cutlass` (Option A — single experiment)
- **Command**:
  ```bash
  ./training/benchmark/scripts/run_single_experiment_local.sh exp2_cutlass \
      --exp-args="--balanced_shuffler --kernel_backend cutlass --caching --ratio 0.1 \
                  --value_dist zipf --value_dist_alpha 1.05" \
      --nproc=4
  ```

## Configuration

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Num HSTU layers | 8 |
| Num attention heads | 4 |
| Head dimension | 256 |
| Item embedding dim | 128 |
| Contextual embedding dim | 128 |
| Prediction head | [512, 8] × 8 tasks |
| Optimizer | Adam (lr=1e-3) |
| Batch size per GPU | 32 |
| Max sequence length | 4096 |
| Sequence length distribution | Zipf (α=1.2), jagged |
| Key value distribution | Zipf (α=1.05) |
| Training iterations | 1000 |
| Log interval | 20 |

### Enabled optimizations

| Optimization | Status |
|-------------|--------|
| Workload-Balanced Shuffler | ✅ Enabled |
| CUTLASS Attention | ✅ Enabled |
| DynamicEmb Caching | ✅ Enabled (ratio 0.1, LRU eviction) |
| Hash-RoundRobin Sharding | ❌ Disabled |
| Prefetch Pipeline | ❌ Disabled |

## Results

### Summary (iter 199–999, post-warmup)

| Metric | Value |
|--------|------:|
| **Avg TFLOPS/GPU** | **140.1** |
| **Avg MFU (%)** | **17.82** |
| **Peak TFLOPS/GPU** | **140.2** |
| **Peak MFU (%)** | **17.82** |
| Avg step time (ms) | 21,548 |
| Tokens per step | 2,244,415 |

### Full iteration log

| Iter | Elapsed (ms) | TFLOPS/GPU | MFU (%) | Loss |
|-----:|------------:|----------:|-------:|-----:|
| 19 | 30,694 | 98.38 | 12.51 | 5.546 |
| 39 | 21,547 | 140.13 | 17.82 | 5.545 |
| 59 | 21,548 | 140.13 | 17.82 | 5.545 |
| 79 | 21,547 | 140.14 | 17.82 | 5.544 |
| 99 | 21,548 | 140.13 | 17.82 | 5.383 |
| 119 | 21,564 | 140.03 | 17.81 | 5.149 |
| 139 | 21,549 | 140.12 | 17.82 | 5.019 |
| 159 | 21,550 | 140.12 | 17.82 | 4.907 |
| 179 | 21,546 | 140.14 | 17.82 | 4.757 |
| 199 | 21,545 | 140.15 | 17.82 | 4.688 |
| 219 | 21,545 | 140.15 | 17.82 | 4.629 |
| 239 | 21,547 | 140.13 | 17.82 | 4.584 |
| 259 | 21,548 | 140.13 | 17.82 | 4.523 |
| 279 | 21,551 | 140.11 | 17.82 | 4.551 |
| 299 | 21,553 | 140.10 | 17.81 | 4.496 |
| 319 | 21,548 | 140.13 | 17.82 | 4.504 |
| 339 | 21,552 | 140.11 | 17.82 | 4.485 |
| 359 | 21,545 | 140.15 | 17.82 | 4.492 |
| 379 | 21,556 | 140.07 | 17.81 | 4.536 |
| 399 | 21,550 | 140.11 | 17.82 | 4.490 |
| 419 | 21,553 | 140.10 | 17.81 | 4.347 |
| 439 | 21,552 | 140.11 | 17.82 | 4.246 |
| 459 | 21,554 | 140.09 | 17.81 | 4.238 |
| 479 | 21,549 | 140.12 | 17.82 | 4.229 |
| 499 | 21,546 | 140.14 | 17.82 | 4.190 |
| 519 | 21,549 | 140.13 | 17.82 | 4.215 |
| 539 | 21,549 | 140.12 | 17.82 | 4.216 |
| 559 | 21,548 | 140.13 | 17.82 | 4.158 |
| 579 | 21,545 | 140.15 | 17.82 | 4.071 |
| 599 | 21,549 | 140.12 | 17.82 | 4.010 |
| 619 | 21,545 | 140.15 | 17.82 | 3.956 |
| 639 | 21,543 | 140.16 | 17.82 | 3.889 |
| 659 | 21,545 | 140.15 | 17.82 | 3.835 |
| 679 | 21,551 | 140.11 | 17.82 | 3.788 |
| 699 | 21,546 | 140.14 | 17.82 | 3.723 |
| 719 | 21,550 | 140.12 | 17.82 | 3.679 |
| 739 | 21,545 | 140.15 | 17.82 | 3.647 |
| 759 | 21,550 | 140.11 | 17.82 | 3.627 |
| 779 | 21,546 | 140.14 | 17.82 | 3.598 |
| 799 | 21,545 | 140.15 | 17.82 | 3.576 |
| 819 | 21,545 | 140.15 | 17.82 | 3.561 |
| 839 | 21,545 | 140.15 | 17.82 | 3.499 |
| 859 | 21,547 | 140.14 | 17.82 | 3.441 |
| 879 | 21,548 | 140.13 | 17.82 | 3.394 |
| 899 | 21,548 | 140.13 | 17.82 | 3.354 |
| 919 | 21,550 | 140.12 | 17.82 | 3.323 |
| 939 | 21,554 | 140.09 | 17.81 | 3.303 |
| 959 | 21,553 | 140.09 | 17.81 | 3.295 |
| 979 | 21,546 | 140.14 | 17.82 | 3.287 |
| 999 | 21,546 | 140.14 | 17.82 | 3.318 |

### Observations

1. **Stable throughput**: Post-warmup throughput is extremely stable at 140.1–140.2 TFLOPS/GPU with less than 0.1% variance across all 41 logged intervals (iter 199–999).

2. **Warmup**: Iter 19 (warmup) is 98.4 TFLOPS due to CUDA kernel JIT compilation and cache cold-start. From iter 39 onward, performance stabilizes immediately.

3. **Caching overhead**: DynamicEmb caching (HBM cache + host backing) is enabled with 10% cache ratio. The HBM cache stores ~29.3 GB per GPU (item + user_id tables), with full backing in host memory (~36.6 GB).

4. **Memory usage**: Free GPU memory after model init is 65,672 MB (of 98 GB), indicating ~32 GB used for model parameters, optimizer states, and HBM cache.

## Reproducing

```bash
cd recsys-examples/examples/hstu

# Option A: single experiment
./training/benchmark/scripts/run_single_experiment_local.sh exp2_cutlass \
    --exp-args="--balanced_shuffler --kernel_backend cutlass --caching --ratio 0.1 \
                --value_dist zipf --value_dist_alpha 1.05" \
    --nproc=4
```

### Prerequisites

- FBGEMM `hstu` package compiled from `jiayus-nvidia/FBGEMM` fork for sm 8.0
- CUTLASS attention kernels (`hstu_attn`) compiled from `corelib/hstu`
- DynamicEmb compiled and installed
- See `docker/Dockerfile` for full build instructions
