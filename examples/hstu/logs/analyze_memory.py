#!/usr/bin/env python3
import csv, os

EXPERIMENTS = [
    "exp0_baseline", "exp1_shuffler", "exp2_cutlass",
    "exp3_caching", "exp4_caching_hr", "exp5_prefetch",
]

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

# Memory Size Summary
print("## Memory Allocation Summary")
print("-" * 80)
print(f"{'Experiment':<25} {'Peak Alloc (MB)':>18} {'Top Allocator':>30}")
print("-" * 80)

for exp in EXPERIMENTS:
    path = f"{exp}_stats.csv_hggc_ppu_mem_size_sum.csv"
    rows = read_csv(path)
    total = 0
    top_name = ""
    top_val = 0
    for r in rows:
        name = r.get("Name", r.get("Op", ""))
        for key in ["Peak Alloc Size (bytes)", "Size (bytes)", "Bytes"]:
            if key in r:
                try:
                    v = int(float(r[key]))
                    total += v
                    if v > top_val:
                        top_val = v
                        top_name = name[:30]
                except:
                    pass
    print(f"{exp:<25} {total/1024/1024:>17.1f} {top_name:>30}")

# Memory Time Summary
print(f"\n## Memory Operation Time Summary")
print("-" * 80)
print(f"{'Experiment':<25} {'Total MemOps (ms)':>18} {'Top Op':>30}")
print("-" * 80)

for exp in EXPERIMENTS:
    path = f"{exp}_stats.csv_hggc_ppu_mem_time_sum.csv"
    rows = read_csv(path)
    total = 0
    top_name = ""
    top_val = 0
    for r in rows:
        name = r.get("Name", r.get("Op", ""))
        for key in ["Total Time (ns)", "Time (ns)"]:
            if key in r:
                try:
                    v = int(float(r[key]))
                    total += v
                    if v > top_val:
                        top_val = v
                        top_name = name[:30]
                except:
                    pass
    print(f"{exp:<25} {total/1e6:>17.1f} {top_name:>30}")

# HGGC API Summary (top 10 API calls by time)
print(f"\n## Top HGGC API Calls (exp2_cutlass)")
print("-" * 80)
path = "exp2_cutlass_stats.csv_hggc_api_sum.csv"
rows = read_csv(path)
print(f"{'API':<50} {'Time%':>6} {'Calls':>8} {'Avg(us)':>10}")
print("-" * 80)
for r in rows[:10]:
    name = r.get("Name", "")[:50]
    pct = r.get("Time (%)", "0")
    calls = r.get("Count", r.get("Instances", "0"))
    avg = r.get("Avg (ns)", "0")
    try:
        avg_us = float(avg) / 1000
    except:
        avg_us = 0
    print(f"{name:<50} {pct:>5}% {calls:>8} {avg_us:>10.1f}")
