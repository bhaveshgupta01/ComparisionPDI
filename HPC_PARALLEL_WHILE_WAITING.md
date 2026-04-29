# Parallel work while V1 pilot + Phase D bugfix queue

Both your queued jobs (176569 Phase D bugfix, 176579 V1 pilot) take ~30 min combined. Three productive things to fire NOW.

---

## Block A — Fire V2/V3/V4 pilots in parallel with V1

**Why:** if there's a variant-specific bug in `train.py`, you find out for *all 4* in the same 25 min instead of serially. Cost: 1.25 extra GPU-hours out of your ~280 budget.

```bash
cd /scratch/$USER/ComparisionPDI

sbatch hpc_phase_c/run_phase_c.sbatch early_crossattn random 42
sbatch hpc_phase_c/run_phase_c.sbatch late_crossattn  random 42
sbatch hpc_phase_c/run_phase_c.sbatch late_concat     random 42

squeue -u $USER
```

Now you have 5 jobs queued: phase-d-v2, V1 pilot, V2 pilot, V3 pilot, V4 pilot. When all 4 pilots produce sensible val MSE, fire **Block 2** of [HPC_PHASE_C_RUN.md](HPC_PHASE_C_RUN.md) but skip ALL 4 pilots (not just V1). Use this skip-pilots variant:

```bash
# After all 4 pilots succeed, run this to fire the remaining 32:
SBATCH=hpc_phase_c/run_phase_c.sbatch
VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
SPLITS=(random cold_drug cold_target)
SEEDS=(42 123 456)

count=0
for v in "${VARIANTS[@]}"; do
    for s in "${SPLITS[@]}"; do
        for sd in "${SEEDS[@]}"; do
            # Skip the 4 already-running pilots (variant + random + seed=42)
            if [ "$s" = "random" ] && [ "$sd" = "42" ]; then
                continue
            fi
            sbatch "$SBATCH" "$v" "$s" "$sd"
            count=$((count+1))
        done
    done
done
echo "==> Submitted $count more (32 expected)"
squeue -u $USER | head -10
```

---

## Block B — BindingDB stats on the login node (CPU only, ~30 sec)

This builds a small CSV of pKi histogram bins, drug-length histogram, protein-length histogram, plus N kinase / N other targets for the data panel of the poster.

```bash
cd /scratch/$USER/ComparisionPDI
source .venv/bin/activate

python3 << 'END_OF_BINDINGDB_STATS_AB7392'
import csv, json, os, re
import numpy as np

TSV = "dataset/BindingDB/BindingDB_PDSPKi.tsv"
OUT_DIR = "outputs/binding_db_stats"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Reading {TSV} ...")

# Find columns: pKi (or Ki nM), SMILES, target sequence
# Most BindingDB exports have these columns by these names; adjust if yours differs.
SMILES_COL = "SMILES"
PROT_COL   = "BindingDB Target Chain  Sequence"
KI_COL     = "Ki (nM)"
NAME_COL   = "Target Name Assigned by Curator or DataSource"

pki_vals, drug_lens, prot_lens, target_names = [], [], [], set()
n_rows = 0
n_skipped = 0

with open(TSV, encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f, delimiter="\t")
    cols = reader.fieldnames or []
    print(f"Columns ({len(cols)}): {[c for c in cols if any(k in c for k in ('SMILES','Sequence','Ki','Name'))][:8]}")

    # Try to find the right column names if defaults aren't there
    smiles_col = SMILES_COL if SMILES_COL in cols else next((c for c in cols if "SMILES" in c), None)
    prot_col   = PROT_COL  if PROT_COL  in cols else next((c for c in cols if "Sequence" in c and "Target" in c), None)
    ki_col     = KI_COL    if KI_COL    in cols else next((c for c in cols if c.strip().startswith("Ki")), None)
    name_col   = NAME_COL  if NAME_COL  in cols else next((c for c in cols if "Target Name" in c), None)
    print(f"Using: smiles={smiles_col}, prot={prot_col}, ki={ki_col}, name={name_col}")

    for row in reader:
        n_rows += 1
        try:
            smi = (row.get(smiles_col) or "").strip()
            prot = (row.get(prot_col) or "").strip()
            ki_raw = (row.get(ki_col) or "").strip().lstrip("<>~ ").rstrip()
            if not smi or not prot or not ki_raw:
                n_skipped += 1
                continue
            ki_nm = float(ki_raw)
            if ki_nm <= 0:
                n_skipped += 1
                continue
            pki = -np.log10(ki_nm * 1e-9)
            pki_vals.append(pki)
            drug_lens.append(len(smi))
            prot_lens.append(len(prot))
            if name_col:
                target_names.add((row.get(name_col) or "").strip())
        except (ValueError, TypeError):
            n_skipped += 1

pki_arr = np.array(pki_vals)
drug_arr = np.array(drug_lens)
prot_arr = np.array(prot_lens)

print(f"\nRead {n_rows} rows, kept {len(pki_arr)}, skipped {n_skipped}")
print(f"\npKi:    n={len(pki_arr):>7d}  min={pki_arr.min():.2f}  max={pki_arr.max():.2f}"
      f"  mean={pki_arr.mean():.2f}  median={np.median(pki_arr):.2f}")
print(f"drug:   n={len(drug_arr):>7d}  min={drug_arr.min():>4d}  max={drug_arr.max():>4d}"
      f"  mean={drug_arr.mean():.1f}  median={np.median(drug_arr):.1f}")
print(f"prot:   n={len(prot_arr):>7d}  min={prot_arr.min():>4d}  max={prot_arr.max():>4d}"
      f"  mean={prot_arr.mean():.1f}  median={np.median(prot_arr):.1f}")
print(f"\nUnique target names: {len(target_names)}")
kinase_count = sum(1 for n in target_names if "kinase" in n.lower())
print(f"  containing 'kinase':   {kinase_count}  ({100*kinase_count/max(len(target_names),1):.1f}%)")

# Save histograms (compact, ~few KB)
np.savez_compressed(
    os.path.join(OUT_DIR, "histograms.npz"),
    pki=pki_arr.astype(np.float32),
    drug_len=drug_arr.astype(np.int32),
    prot_len=prot_arr.astype(np.int32),
)

# Save text summary
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump({
        "n_rows_total": n_rows,
        "n_kept":       len(pki_arr),
        "n_skipped":    n_skipped,
        "pki":  {"min": float(pki_arr.min()), "max": float(pki_arr.max()),
                 "mean": float(pki_arr.mean()), "median": float(np.median(pki_arr)),
                 "p5":  float(np.percentile(pki_arr, 5)),
                 "p95": float(np.percentile(pki_arr, 95))},
        "drug_len": {"min": int(drug_arr.min()), "max": int(drug_arr.max()),
                     "mean": float(drug_arr.mean()), "median": float(np.median(drug_arr))},
        "prot_len": {"min": int(prot_arr.min()), "max": int(prot_arr.max()),
                     "mean": float(prot_arr.mean()), "median": float(np.median(prot_arr))},
        "n_unique_targets": len(target_names),
        "n_kinase_targets": kinase_count,
    }, f, indent=2)

print(f"\nWrote: {OUT_DIR}/histograms.npz, {OUT_DIR}/summary.json")
END_OF_BINDINGDB_STATS_AB7392

ls -la outputs/binding_db_stats/
cat outputs/binding_db_stats/summary.json
```

The `histograms.npz` (a few KB) is what I'll download to Mac to build figures 06 (pKi histogram), 07 (length distributions for the data panel of the poster).

---

## Block C — Push everything to GitHub

If you want to back up the work-so-far while HPC grinds, paste the two blocks from [HPC_GITHUB_SYNC.md](HPC_GITHUB_SYNC.md). The first runs on HPC (writes the new sbatch / config / scripts to a feature branch). The second is on Mac (scp + commit on HPC). Skip if you'd rather focus on Phase C / D.

---

## Pulling stats back to Mac (after Block B finishes)

```bash
# from Mac:
mkdir -p ~/CodeFiles/DTI_MLFinalProject/binding_db_stats
scp bg2896@gw.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/outputs/binding_db_stats/* \
    ~/CodeFiles/DTI_MLFinalProject/binding_db_stats/
```

Or via OOD file browser. Once they're on Mac, paste me a confirmation and I'll add diagrams 06 + 07 to `poster_figures/build_all.py`.

---

## Recap — your queue right now

| Job | What | ETA from now |
|---|---|---|
| 176569 phase-d-v2 | Phase D bugfix re-extract | ~6 min once R |
| 176579 phase-c V1 | Phase C pilot, V1 / random / seed 42 | ~25-35 min once R |
| (after Block A) phase-c V2/V3/V4 | 3 more pilots | same |
| (after Block B) BindingDB stats | runs on login node, no queue | ~30 sec |

After everything finishes, you'll have validated all 4 variants' Phase C interfaces + correct Phase D extraction + data-panel figure data — ready to run all 32 remaining Phase C jobs in one shot.
