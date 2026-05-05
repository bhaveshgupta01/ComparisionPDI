# Phase E5 — Mask-aware attention entropy

Recomputes attention entropy on the existing Phase D artifacts, but averages
over *valid* query positions only (excludes pad tokens that artificially
inflated the original poster numbers).

## Per-variant per-layer entropy (nats)

| Variant | Component | Layer | Pad-contaminated (old) | Mask-aware (new) | Δ (new − old) |
|---|---|---|---|---|---|
| V1 Early Concat | encoder | 0 | 6.068 | 6.057 | -0.011 |
| V1 Early Concat | encoder | 1 | 6.003 | 5.978 | -0.025 |
| V1 Early Concat | encoder | 2 | 5.905 | 5.897 | -0.009 |
| V1 Early Concat | encoder | 3 | 5.820 | 5.699 | -0.121 |
| V1 Early Concat | encoder | 4 | 5.166 | 5.148 | -0.019 |
| V1 Early Concat | encoder | 5 | 5.965 | 5.941 | -0.025 |
| V2 Early X-Attn | encoder | 0 | 6.052 | 6.024 | -0.028 |
| V2 Early X-Attn | encoder | 1 | 5.912 | 5.941 | +0.028 |
| V2 Early X-Attn | encoder | 2 | 5.956 | 5.945 | -0.012 |
| V2 Early X-Attn | encoder | 3 | 5.924 | 5.888 | -0.036 |
| V2 Early X-Attn | encoder | 4 | 5.938 | 5.924 | -0.014 |
| V2 Early X-Attn | encoder | 5 | 5.961 | 5.967 | +0.006 |
| V3 Late X-Attn | drug_encoder | 0 | 3.160 | 3.021 | -0.139 |
| V3 Late X-Attn | prot_encoder | 0 | 5.600 | 5.260 | -0.340 |
| V4 Late Concat | drug_encoder | 0 | 3.148 | 3.098 | -0.050 |
| V4 Late Concat | prot_encoder | 0 | 5.051 | 4.615 | -0.436 |

## V1 vs V2 ranking check (poster claim: V1 < V2 in entropy preserved under mask-aware)

- V1 mean entropy across layers: 5.821 (old) → 5.787 (new)
- V2 mean entropy across layers: 5.957 (old) → 5.948 (new)
- V1 minimum entropy (the layer-4 dip claim): 5.166 (old) → 5.148 (new)
- V2 minimum entropy: 5.912 (old) → 5.888 (new)

**Ranking preserved**: V1 < V2 in mean entropy (V1=5.787 < V2=5.948). The poster's central claim — V1 attention is more concentrated than V2 — survives mask-aware reanalysis.
