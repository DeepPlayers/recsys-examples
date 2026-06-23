#!/usr/bin/env bash
# ============================================================================
#  HSTU E2E Benchmark 一键测评脚本
#
#  功能：
#    Option A: exp2_cutlass 单实验（快速验证性能）
#    Option B: 全量 6 实验（渐进优化对比，展示 PPU vs H20）
#
#  前置条件：
#    已运行 setup.sh 完成环境部署
#
#  用法：
#    bash run_benchmark.sh [NPROC]
#
#    NPROC: GPU 卡数，默认 8
#
#  示例：
#    bash run_benchmark.sh        # 8 卡
#    bash run_benchmark.sh 4      # 4 卡
# ============================================================================
set -euo pipefail

NPROC="${1:-8}"

# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HSTU_DIR="$REPO_ROOT/examples/hstu"
LOG_DIR="$REPO_ROOT/examples/hstu/training/ppu/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/benchmark_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

echo "============================================" | tee "$LOG_FILE"
echo "  HSTU E2E Benchmark (PPU vs H20)"         | tee -a "$LOG_FILE"
echo "  GPUs:      $NPROC"                     | tee -a "$LOG_FILE"
echo "  Repo:      $REPO_ROOT"                 | tee -a "$LOG_FILE"
echo "  Log file:  $LOG_FILE"                  | tee -a "$LOG_FILE"
echo "  Started:   $(date)"                    | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# ── 前置检查 ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "Checking prerequisites ..." | tee -a "$LOG_FILE"

if ! python3 -c "import hstu_attn_2_cuda" 2>/dev/null; then
    echo "ERROR: hstu_attn not installed. Run setup.sh first." | tee -a "$LOG_FILE"
    exit 1
fi
if ! python3 -c "import torch; import hstu; _ = torch.ops.fbgemm.hstu_varlen_fwd_80" 2>/dev/null; then
    echo "ERROR: hstu (FBGEMM) not installed. Run setup.sh first." | tee -a "$LOG_FILE"
    exit 1
fi
echo "  -> CUTLASS kernels OK." | tee -a "$LOG_FILE"

# PPU 环境变量
export HSTU_ENABLE_EXTENDED_BW_CONFIGS=TRUE

cd "$HSTU_DIR"

# ── Option A: exp2_cutlass 单实验 ─────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "  Option A: exp2_cutlass ($NPROC GPUs)"  | tee -a "$LOG_FILE"
echo "  Started: $(date)"                    | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

./training/benchmark/scripts/run_single_experiment_local.sh exp2_cutlass \
    --exp-args="--balanced_shuffler --kernel_backend cutlass --caching --ratio 0.1 \
                --value_dist zipf --value_dist_alpha 1.05" \
    --nproc="$NPROC" 2>&1 | tee -a "$LOG_FILE"

OPTION_A_EXIT=${PIPESTATUS[0]}
echo "" | tee -a "$LOG_FILE"
echo "  Option A: exit code $OPTION_A_EXIT, $(date)" | tee -a "$LOG_FILE"

if [ $OPTION_A_EXIT -ne 0 ]; then
    echo "  ⚠️  Option A failed! Continuing with Option B ..." | tee -a "$LOG_FILE"
fi

# ── Option B: 全量 6 实验 ─────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "  Option B: All 6 experiments ($NPROC GPUs)" | tee -a "$LOG_FILE"
echo "  Started: $(date)"                        | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

./training/benchmark/scripts/run_all_experiments_local.sh \
    --exp-file=training/benchmark/experiments.txt \
    --nproc="$NPROC" 2>&1 | tee -a "$LOG_FILE"

OPTION_B_EXIT=${PIPESTATUS[0]}
echo "" | tee -a "$LOG_FILE"
echo "  Option B: exit code $OPTION_B_EXIT, $(date)" | tee -a "$LOG_FILE"

# ── PPU vs H20 对比摘要 ──────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "  PPU vs H20 Reference (exp2_cutlass)"        | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
printf "  %-20s %-12s %-12s\n" "Metric" "PPU-ZW810E" "H20 (ref)" | tee -a "$LOG_FILE"
printf "  %-20s %-12s %-12s\n" "--------------------" "------------" "------------" | tee -a "$LOG_FILE"
printf "  %-20s %-12s %-12s\n" "BF16 Peak TFLOPS" "787" "989" | tee -a "$LOG_FILE"
printf "  %-20s %-12s %-12s\n" "SM count" "64" "132" | tee -a "$LOG_FILE"
printf "  %-20s %-12s %-12s\n" "Architecture" "SM 8.0" "SM 9.0" | tee -a "$LOG_FILE"
printf "  %-20s %-12s %-12s\n" "H20 theoretical max" "79.6%" "100%" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  See ppu/docs/PPU_VS_H20.md for full comparison." | tee -a "$LOG_FILE"

# ── 总结 ──────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "  Benchmark Complete!"                        | tee -a "$LOG_FILE"
echo "  Option A exit code: $OPTION_A_EXIT"         | tee -a "$LOG_FILE"
echo "  Option B exit code: $OPTION_B_EXIT"         | tee -a "$LOG_FILE"
echo "  Finished: $(date)"                          | tee -a "$LOG_FILE"
echo "  Full log: $LOG_FILE"                        | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
