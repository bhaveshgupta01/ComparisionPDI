#!/bin/bash
set -e
cd /scratch/$USER/ComparisionPDI
SB=hpc_early_crossattn/run_ec_a100_fast.sbatch

sbatch "$SB" baseline
sbatch "$SB" lr5e5    --lr 5e-5
sbatch "$SB" lr1e4    --lr 1e-4
sbatch "$SB" lr3e4    --lr 3e-4
sbatch "$SB" dm64     --d_model 64  --n_heads 2
sbatch "$SB" dm128    --d_model 128 --n_heads 4
sbatch "$SB" dm256    --d_model 256 --n_heads 8
sbatch "$SB" l4       --n_layers 4
sbatch "$SB" l6       --n_layers 6
sbatch "$SB" l8       --n_layers 8
sbatch "$SB" bs32     --batch_size 32
sbatch "$SB" bs64     --batch_size 64
sbatch "$SB" bs128    --batch_size 128
sbatch "$SB" do01     --dropout 0.1
sbatch "$SB" do02     --dropout 0.2
sbatch "$SB" do03     --dropout 0.3
sbatch "$SB" h2       --n_heads 2
sbatch "$SB" h8       --n_heads 8
sbatch "$SB" seed42   --seed 42
sbatch "$SB" seed123  --seed 123
sbatch "$SB" seed456  --seed 456
sbatch "$SB" cold_drug    --split cold_drug
sbatch "$SB" cold_target  --split cold_target

squeue -u $USER
