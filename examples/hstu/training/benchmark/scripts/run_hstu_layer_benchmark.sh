#!/bin/bash
# ============================================================================
# Quick launcher for the fused HSTU layer benchmark sweep.
# Thin wrapper around the unified run_all_experiments_local.sh flow.
#
# Usage (from examples/hstu/):
#   bash training/benchmark/scripts/run_hstu_layer_benchmark.sh [options]
#
# Options: forwarded verbatim to run_all_experiments_local.sh
#   --exp-file=FILE   Override config list (default: training/benchmark/layer_experiments.txt)
#   --hstu-root=PATH  Specify examples/hstu directory path
#   --dry-run         Print commands only, do not execute
#   --help,-h         Show this help
#
# To customize the sweep, edit training/benchmark/layer_experiments.txt or
# pass --exp-file=<your_list>.
# ============================================================================
set -e

case " $* " in
    *" --help "*|*" -h "*)
        sed -n '2,20p' "$0" | sed 's/^# \?//'
        exit 0
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec bash "${SCRIPT_DIR}/run_all_experiments_local.sh" \
    --benchmark-type=hstu-layer \
    "$@"
