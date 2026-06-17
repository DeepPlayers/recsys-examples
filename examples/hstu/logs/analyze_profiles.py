#!/usr/bin/env python3
"""
Analyze HSTU benchmark profiles across 6 experiments.
Generates comparison tables for kernel time, memory, and communication.
"""

import csv
import os
from pathlib import Path

EXPERIMENTS = [
    "exp0_baseline",
    "exp1_shuffler",
    "exp2_cutlass",
    "exp3_caching",
    "exp4_caching_hr",
    "exp5_prefetch",
]

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

def parse_time_ns(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0

def analyze_kernels(exp_name):
    """Extract top kernel time breakdown."""
    path = f"{exp_name}_stats.csv_hggc_ppu_kern_sum.csv"
    rows = read_csv(path)
    
    total_ns = sum(parse_time_ns(r.get("Total Time (ns)", 0)) for r in rows)
    
    categories = {
        "Attention (fwd+bwd)": 0,
        "GEMM/MatMul": 0,
        "LayerNorm": 0,
        "Embedding/Scatter": 0,
        "PCCL/NCCL": 0,
        "Elementwise": 0,
        "Other": 0,
    }
    
    for r in rows:
        name = r.get("Name", "")
        t = parse_time_ns(r.get("Total Time (ns)", 0))
        pct = parse_time_ns(r.get("Time (%)", 0))
        
        if "hstu_fwd" in name or "hstu_bwd" in name:
            categories["Attention (fwd+bwd)"] += t
        elif "gemm" in name.lower() or "matmul" in name.lower():
            categories["GEMM/MatMul"] += t
        elif "layer_norm" in name or "ln_mul" in name or "_ln_" in name:
            categories["LayerNorm"] += t
        elif "embedding" in name or "dyn_emb" in name or "scatter" in name or "jagged" in name:
            categories["Embedding/Scatter"] += t
        elif "pccl" in name.lower() or "nccl" in name.lower():
            categories["PCCL/NCCL"] += t
        elif "elementwise" in name or "fill" in name or "copy" in name or "add_" in name:
            categories["Elementwise"] += t
        else:
            categories["Other"] += t
    
    return total_ns, categories

def analyze_memory(exp_name):
    """Extract memory stats."""
    path = f"{exp_name}_stats.csv_hggc_ppu_mem_size_sum.csv"
    rows = read_csv(path)
    peak_alloc = 0
    for r in rows:
        val = r.get("Peak Alloc Size (bytes)", r.get("Size (bytes)", "0"))
        try:
            peak_alloc = max(peak_alloc, int(float(val)))
        except:
            pass
    return peak_alloc

# Main analysis
print("=" * 90)
print("HSTU E2E Benchmark - Profile Analysis (4x PPU-ZW810E)")
print("=" * 90)

# Table 1: Kernel Time Breakdown
print("\n## 1. GPU Kernel Time Breakdown (ns)")
print("-" * 90)
header = f"{'Category':<25}"
for exp in EXPERIMENTS:
    header += f" {exp:>12}"
print(header)
print("-" * 90)

all_data = {}
for exp in EXPERIMENTS:
    total, cats = analyze_kernels(exp)
    all_data[exp] = (total, cats)

categories = list(all_data[EXPERIMENTS[0]][1].keys())
for cat in categories:
    row = f"{cat:<25}"
    for exp in EXPERIMENTS:
        val = all_data[exp][1][cat]
        row += f" {val/1e9:>11.2f}s"
    print(row)

print("-" * 90)
row = f"{'TOTAL':<25}"
for exp in EXPERIMENTS:
    total = all_data[exp][0]
    row += f" {total/1e9:>11.2f}s"
print(row)

# Table 2: Percentage Breakdown
print("\n## 2. GPU Kernel Time Distribution (%)")
print("-" * 90)
header = f"{'Category':<25}"
for exp in EXPERIMENTS:
    header += f" {exp:>12}"
print(header)
print("-" * 90)

for cat in categories:
    row = f"{cat:<25}"
    for exp in EXPERIMENTS:
        total = all_data[exp][0]
        val = all_data[exp][1][cat]
        pct = (val / total * 100) if total > 0 else 0
        row += f" {pct:>11.1f}%"
    print(row)

# Table 3: Attention kernel dominance
print("\n## 3. Attention Kernel Detail")
print("-" * 90)
header = f"{'Kernel':<30}"
for exp in EXPERIMENTS:
    header += f" {exp:>12}"
print(header)
print("-" * 90)

for exp in EXPERIMENTS:
    path = f"{exp}_stats.csv_hggc_ppu_kern_sum.csv"
    rows = read_csv(path)
    for r in rows[:5]:
        name = r.get("Name", "")[:50]
        pct = r.get("Time (%)", "0")
        if exp == EXPERIMENTS[0]:
            row = f"{name:<30}"
        else:
            continue

# Simpler: just top 5 per experiment
for exp in EXPERIMENTS:
    path = f"{exp}_stats.csv_hggc_ppu_kern_sum.csv"
    rows = read_csv(path)
    print(f"\n{exp}: Top 3 kernels:")
    for r in rows[:3]:
        name = r.get("Name", "")[:60]
        pct = r.get("Time (%)", "0")
        total_ns = parse_time_ns(r.get("Total Time (ns)", 0))
        print(f"  {pct:>5}% ({total_ns/1e9:.2f}s) {name}")

# Table 4: Communication overhead
print("\n\n## 4. Communication (PCCL/NCCL) Overhead")
print("-" * 70)
for exp in EXPERIMENTS:
    path = f"{exp}_stats.csv_pccl_desync_summary.csv"
    rows = read_csv(path)
    total = all_data[exp][0]
    comm_time = all_data[exp][1]["PCCL/NCCL"]
    pct = (comm_time / total * 100) if total > 0 else 0
    print(f"{exp:<25} {comm_time/1e9:>8.2f}s  ({pct:.1f}%)")

print("\n" + "=" * 90)
print("Analysis complete.")
