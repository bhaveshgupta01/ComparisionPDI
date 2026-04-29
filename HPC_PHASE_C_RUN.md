# Phase C — single-block setup + run

Two paste-blocks. **Block 1** writes the config + sbatch + submitter and fires ONE pilot job. **Block 2** (run only after pilot succeeds) fires all 36.

While Phase D bugfix is running, you can paste Block 1 immediately — they don't share GPU slots. Phase C jobs queue on `c12m85-a100-1`; Phase D bugfix is also on that partition but each Phase C job is independent so the queue just lengthens (doesn't conflict).

---

## Block 1 — write files + submit pilot

```bash
cd /scratch/$USER/ComparisionPDI
mkdir -p configs hpc_phase_c logs outputs/phase_c

# ---- 1. configs/phase_c_fair.yaml -----------------------------------------
cat > configs/phase_c_fair.yaml << 'END_OF_PHASE_C_YAML_AB7392'
# Phase C "Fair Config" - locked 2026-04-28 (Bhavesh, soloing Phase B)
d_model: 128
n_heads: 4
d_ff: 512
head_hidden: 256
head_dropout: 0.2
n_layers_shared: 6        # V1, V2 use a 6-layer shared encoder
n_layers_per_side: 3      # V3, V4 use 3 per side (3+3 total = 6)
batch_size: 64
lr: 3.0e-4
dropout: 0.1
weight_decay: 0.01
warmup_steps: 500
optimizer: adamw
lr_schedule: cosine
epochs: 30
early_stopping_patience: 5
val_every_epoch: 1
dataset: bindingdb_pdspki
max_rows: null
max_drug_len: 100
max_prot_len: 1200
seeds: [42, 123, 456]
splits: [random, cold_drug, cold_target]
variants: [early_concat, early_crossattn, late_crossattn, late_concat]
deterministic: true
cudnn_benchmark: false
END_OF_PHASE_C_YAML_AB7392

# ---- 2. hpc_phase_c/run_phase_c.sbatch ------------------------------------
cat > hpc_phase_c/run_phase_c.sbatch << 'END_OF_PHASE_C_SBATCH_AB7392'
#!/bin/bash
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=phase-c
#SBATCH --output=logs/phase_c_%x_%j.out
#SBATCH --error=logs/phase_c_%x_%j.err

set -e

VARIANT="${1:?usage: sbatch run_phase_c.sbatch <variant> <split> <seed>}"
SPLIT="${2:?missing split}"
SEED="${3:?missing seed}"

case "$VARIANT" in
    early_concat|early_crossattn) NLAYERS=6 ;;
    late_crossattn|late_concat)   NLAYERS=3 ;;
    *) echo "Unknown variant: $VARIANT"; exit 1 ;;
esac

TAG="phase_c_${VARIANT}_${SPLIT}_seed${SEED}"

module purge
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_c
source .venv/bin/activate

echo "==================== ENV ===================="
which python
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0))"
echo "==================== RUN ===================="
echo "Variant:  $VARIANT"
echo "Split:    $SPLIT"
echo "Seed:     $SEED"
echo "n_layers: $NLAYERS"
echo "Tag:      $TAG"
echo "============================================="

python scripts/train.py \
    --variant   "$VARIANT" \
    --split     "$SPLIT" \
    --seed      "$SEED" \
    --output_dir "outputs/phase_c/${TAG}" \
    --d_model   128 \
    --n_heads   4 \
    --n_layers  "$NLAYERS" \
    --batch_size 64 \
    --lr        3e-4 \
    --dropout   0.1 \
    --epochs    30

echo "==================== DONE ===================="
echo "Results:  outputs/phase_c/${TAG}/"
END_OF_PHASE_C_SBATCH_AB7392

# ---- 3. hpc_phase_c/submit_phase_c_all.sh ---------------------------------
cat > hpc_phase_c/submit_phase_c_all.sh << 'END_OF_PHASE_C_SUBMITTER_AB7392'
#!/bin/bash
# Phase C master submitter - 36 jobs (4 variants x 3 splits x 3 seeds).
set -e
cd /scratch/$USER/ComparisionPDI
mkdir -p logs outputs/phase_c

SBATCH="hpc_phase_c/run_phase_c.sbatch"
VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
SPLITS=(random cold_drug cold_target)
SEEDS=(42 123 456)

count=0
for v in "${VARIANTS[@]}"; do
    for s in "${SPLITS[@]}"; do
        for sd in "${SEEDS[@]}"; do
            sbatch "$SBATCH" "$v" "$s" "$sd"
            count=$((count+1))
        done
    done
done
echo
echo "==> Submitted $count Phase C jobs."
echo "==> Monitor:  squeue -u \$USER"
END_OF_PHASE_C_SUBMITTER_AB7392

# ---- 4. Make executable ----------------------------------------------------
chmod +x hpc_phase_c/run_phase_c.sbatch hpc_phase_c/submit_phase_c_all.sh

# ---- 5. PILOT: submit ONE job to validate the sbatch + train.py interface --
echo
echo "==> Submitting pilot: early_concat / random / seed 42"
sbatch hpc_phase_c/run_phase_c.sbatch early_concat random 42

# ---- 6. Show queue ---------------------------------------------------------
sleep 2
squeue -u $USER

echo
echo "==> Pilot submitted. Wait until it shows state R, then watch the log:"
echo "    tail -f \$(ls -t logs/phase_c_phase-c_*.out | head -1)"
echo
echo "==> Once you see results.csv land in outputs/phase_c/phase_c_early_concat_random_seed42/results/,"
echo "    run Block 2 to fire the remaining 35 jobs."
```

## What success looks like

After ~25-35 min, the pilot job's log should end with something like:

```
Epoch 30/30  val_mse=1.43  val_ci=0.683  train_mse=1.21
==================== DONE ====================
Results:  outputs/phase_c/phase_c_early_concat_random_seed42/
```

And then:

```bash
ls outputs/phase_c/phase_c_early_concat_random_seed42/results/
# expected: results.csv, predictions.csv, ...
cat outputs/phase_c/phase_c_early_concat_random_seed42/results/results.csv | head
```

If you see a sensible val MSE in the 1.2–1.6 range → ✅ Block 2.

If the pilot crashes:
- **`unrecognized arguments: --epochs`** → your `train.py` uses a different flag name. Tell me which flag name (`--num_epochs`? `--max_epochs`?) and I'll patch the sbatch.
- **`KeyError: 'split'`** → split passing path is different. Paste the error and I'll fix.
- **OOM** → drop `batch_size 64` to `32` in run_phase_c.sbatch (one-line edit).
- **import error / venv issue** → less likely since Phase A used the same venv successfully.

---

## Block 2 — fire the remaining 35 (only after pilot succeeds)

```bash
cd /scratch/$USER/ComparisionPDI

# The submitter would re-submit the pilot too. To avoid duplication, fire only
# the 35 NOT covered by the pilot:
SBATCH=hpc_phase_c/run_phase_c.sbatch
VARIANTS=(early_concat early_crossattn late_crossattn late_concat)
SPLITS=(random cold_drug cold_target)
SEEDS=(42 123 456)

count=0
for v in "${VARIANTS[@]}"; do
    for s in "${SPLITS[@]}"; do
        for sd in "${SEEDS[@]}"; do
            # Skip the pilot
            if [ "$v" = "early_concat" ] && [ "$s" = "random" ] && [ "$sd" = "42" ]; then
                continue
            fi
            sbatch "$SBATCH" "$v" "$s" "$sd"
            count=$((count+1))
        done
    done
done
echo "==> Submitted $count more Phase C jobs (35 expected)."
squeue -u $USER | head -20
```

Or — if you want to be lazy and not skip the pilot (it'll just produce a duplicate result you can ignore):

```bash
bash hpc_phase_c/submit_phase_c_all.sh
```

## Monitoring all 36 progress

```bash
# How many jobs left
squeue -u $USER | grep -c phase-c

# How many finished successfully
find outputs/phase_c -name results.csv | wc -l       # target: 36

# How many failed (exit code != 0)
sacct -u $USER --name=phase-c --format=JobID,State,ExitCode --noheader -X | grep -v COMPLETED | grep -v PENDING | grep -v RUNNING

# Real GPU-hours used so far this month
sacct -u $USER -X --starttime=2026-04-01 --format=Elapsed -P --noheader \
  | awk -F'|' '{n=split($1,t,":");h=(n==3?t[1]+t[2]/60+t[3]/3600:t[1]/60+t[2]/3600);s+=h}END{printf "%.1f GPU-hours\n",s}'
```

When `find ... | wc -l` hits 36, you're done. Zip `outputs/phase_c/` and pull to Mac, or just scp the `results.csv` files (they're tiny, ~kB each):

```bash
# On Mac:
mkdir -p ~/CodeFiles/DTI_MLFinalProject/phase_c_results
rsync -av --include='*/' --include='results.csv' --exclude='*' \
  bg2896@gw.hpc.nyu.edu:/scratch/bg2896/ComparisionPDI/outputs/phase_c/ \
  ~/CodeFiles/DTI_MLFinalProject/phase_c_results/
```

Or via OOD file browser: navigate to `outputs/phase_c/`, tick all 36 dirs, Download (it'll zip).

Then tell me — diagrams 10b (MSE per split) and 11 (CI per split) auto-populate from the results.csv files; I'll regenerate `poster_figures/` and we'll have real Phase C numbers in the poster.
