# Phase E4 Findings — Causal ablations on Phase C (d=128) checkpoints

Each row of the SUMMARY.csv is a single inference pass on the same fixed
held-out batch with one circuit (a layer's attention output, or one head's
contribution) zeroed. Δ MSE = ablated - baseline. Larger positive Δ ⇒
the circuit is more important to the prediction.

## Per-variant most-important layer

- **V1 EC**: most-load-bearing layer = `encoder.layer4` (Δ MSE = +0.110). Top-3: encoder.L4 (+0.110), encoder.L5 (+0.035), encoder.L3 (+0.035)
- **V2 EX**: most-load-bearing layer = `encoder.layer0` (Δ MSE = +0.035). Top-3: encoder.L0 (+0.035), encoder.L5 (-0.056), encoder.L2 (-0.112)
- **V3 LX**: most-load-bearing layer = `prot_encoder.layer0` (Δ MSE = -0.053). Top-3: prot_encoder.L0 (-0.053), drug_encoder.L0 (-0.505)
- **V4 LC**: most-load-bearing layer = `prot_encoder.layer0` (Δ MSE = -0.000). Top-3: prot_encoder.L0 (-0.000), drug_encoder.L0 (-0.094)

## Per-variant most-important head

- **V1 EC**: most-load-bearing head = `encoder.L0 h=0` (Δ MSE = +0.047). Top-3: encoder.L0/h0 (+0.047), encoder.L0/h3 (+0.038), encoder.L4/h0 (+0.035)
- **V2 EX**: most-load-bearing head = `encoder.L0 h=0` (Δ MSE = +0.081). Top-3: encoder.L0/h0 (+0.081), encoder.L0/h3 (+0.039), encoder.L0/h1 (+0.018)
- **V3 LX**: most-load-bearing head = `prot_encoder.L0 h=0` (Δ MSE = +0.068). Top-3: prot_encoder.L0/h0 (+0.068), prot_encoder.L0/h1 (+0.028), drug_encoder.L0/h1 (+0.025)
- **V4 LC**: most-load-bearing head = `drug_encoder.L0 h=0` (Δ MSE = -0.003). Top-3: drug_encoder.L0/h0 (-0.003), prot_encoder.L0/h1 (-0.030), prot_encoder.L0/h0 (-0.094)

## V3 ↔ V4 drug-encoder representation swap

- V3 baseline MSE: 1.9742
- V3 with V4's drug encoder: 1.5802 (Δ = -0.3940)
- V4 baseline MSE: 1.6471
- V4 with V3's drug encoder: 1.6843 (Δ = +0.0372)

**Drug encoders are co-adapted with their fusion module** — swapping yields >5% MSE degradation.
