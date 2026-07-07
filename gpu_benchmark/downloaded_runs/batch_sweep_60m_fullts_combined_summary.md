# Full TinyStories batch sweep summary

Effective batch was fixed at 65,536 tokens/update. Higher micro-batch means fewer gradient accumulation passes.

| Optimizer | Micro batch | Grad accum | Status | Tok/s | Max GPU GB | Return code |
|---|---:|---:|---|---:|---:|---:|
| adamw | 4 | 32 | completed | 110481 | 2.53 | 0 |
| adamw | 8 | 16 | completed | 145612 | 4.08 | 0 |
| adamw | 16 | 8 | completed | 165869 | 7.17 | 0 |
| adamw | 32 | 4 | completed | 162943 | 13.36 | 0 |
| adamw | 64 | 2 | completed | 152056 | 25.74 | 0 |
| adamw | 128 | 1 | running |  | 0.00 | 1 |
| muon | 4 | 32 | completed | 101895 | 2.41 | 0 |
| muon | 8 | 16 | completed | 130642 | 3.96 | 0 |
| muon | 16 | 8 | completed | 147050 | 7.05 | 0 |
| muon | 32 | 4 | completed | 149177 | 13.25 | 0 |
| muon | 64 | 2 | completed | 140502 | 25.63 | 0 |

Decision: use micro_batch_size=16 and grad_accum_steps=8 for the first full-TinyStories AdamW vs Muon benchmark. It is fastest for AdamW, nearly fastest for Muon, and leaves large VRAM safety margin.
