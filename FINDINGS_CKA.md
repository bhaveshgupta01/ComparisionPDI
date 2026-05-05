# CKA findings — d=128 vs d=256

## Phase C (d=128) CKA matrix

| | V1 EC | V2 EX | V3 LX | V4 LC |
|---|---|---|---|---|
| V1 EC | 1.000 | 0.976 | 0.663 | 0.575 |
| V2 EX | 0.976 | 1.000 | 0.682 | 0.591 |
| V3 LX | 0.663 | 0.682 | 1.000 | 0.928 |
| V4 LC | 0.575 | 0.591 | 0.928 | 1.000 |

## Phase E1 (d=256) CKA matrix

| | V1 EC | V2 EX | V3 LX | V4 LC |
|---|---|---|---|---|
| V1 EC | 1.000 | 0.761 | 0.635 | 0.518 |
| V2 EX | 0.761 | 1.000 | 0.735 | 0.535 |
| V3 LX | 0.635 | 0.735 | 1.000 | 0.869 |
| V4 LC | 0.518 | 0.535 | 0.869 | 1.000 |

## Cluster check

- Within-early {V1, V2} CKA = 0.761
- Within-late  {V3, V4} CKA = 0.869
- Across stages mean CKA      = 0.606

**Fusion-stage clustering preserved at d=256.** Within-cluster similarity exceeds cross-cluster by >0.05.
