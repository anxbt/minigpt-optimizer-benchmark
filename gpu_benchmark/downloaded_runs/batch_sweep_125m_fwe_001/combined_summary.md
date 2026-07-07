# 125M FineWeb-Edu batch sweep

Model: MiniGPT-Dense-125M-v1. Dataset cache: FineWeb-Edu 500M extracted cache. Effective batch fixed at 65,536 tokens/update.

| Optimizer | Micro batch | Grad accum | Status | Tok/s | Max GPU GB |
|---|---:|---:|---|---:|---:|
| adamw | 2 | 64 | completed | 52921 | 3.13 |
| adamw | 4 | 32 | completed | 75633 | 4.03 |
| adamw | 8 | 16 | completed | 97032 | 5.87 |
| adamw | 16 | 8 | completed | 103450 | 9.63 |
| adamw | 32 | 4 | completed | 101052 | 17.03 |
| muon | 2 | 64 | completed | 49121 | 2.81 |
| muon | 4 | 32 | completed | 67333 | 3.71 |
| muon | 8 | 16 | completed | 83957 | 5.55 |
| muon | 16 | 8 | completed | 88335 | 9.29 |
| muon | 32 | 4 | completed | 86630 | 16.72 |

Decision: use micro_batch_size=16 and grad_accum_steps=8 for the first 125M benchmark. It is fastest for both optimizers in this sweep and keeps a large VRAM margin.
