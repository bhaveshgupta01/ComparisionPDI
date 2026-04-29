#!/usr/bin/env python3
"""Build PHASE_A_4VARIANT_COMPARISON.csv from sweep outputs.
Run this on HPC at /scratch/$USER/ComparisionPDI/.
"""
import csv, glob, os, re

leaderboards = {"V1": {}, "V2": {}, "V3": {}, "V4": {}}

def add_variant(prefix, lboard):
    for f in glob.glob(f"outputs/sweeps/{prefix}*/results/results.csv"):
        tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
        base = tag
        for p in ["v1_", "v2_", "v3_"]:
            if base.startswith(p):
                base = base[len(p):]
                break
        base = re.sub(r"_fast$", "", base)
        with open(f) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        if len(lines) >= 2:
            lboard[base] = lines[-1].split(",")[-1]

add_variant("v1_", leaderboards["V1"])
add_variant("v2_", leaderboards["V2"])
add_variant("v3_", leaderboards["V3"])

# V4 = anything fast that doesn't start with v1_/v2_/v3_
for f in glob.glob("outputs/sweeps/*_fast/results/results.csv"):
    tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
    if any(tag.startswith(p) for p in ["v1_", "v2_", "v3_"]):
        continue
    base = re.sub(r"_fast$", "", tag)
    with open(f) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) >= 2:
        leaderboards["V4"][base] = lines[-1].split(",")[-1]

all_configs = sorted(set().union(*[lb.keys() for lb in leaderboards.values()]))

with open("PHASE_A_4VARIANT_COMPARISON.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["config", "V1_early_concat", "V2_early_xattn", "V3_late_xattn", "V4_late_concat"])
    for cfg in all_configs:
        w.writerow([cfg] + [leaderboards[v].get(cfg, "") for v in ["V1", "V2", "V3", "V4"]])

print(f"Wrote {len(all_configs)} rows to PHASE_A_4VARIANT_COMPARISON.csv")
print()
print(open("PHASE_A_4VARIANT_COMPARISON.csv").read())
