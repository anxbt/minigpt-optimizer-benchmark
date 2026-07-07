# Multi-seed MiniGPT 57M benchmark summary

All listed optimizer pairs use the same seed, same model, same data cache, same batch settings, and same token budget within each row.

## Individual runs

| Dataset | Steps | Seed | Optimizer | Status | Tokens | Best val loss | Last val loss | Median tok/s |
|---|---:|---:|---|---|---:|---:|---:|---:|
| FineWeb-Edu-500M | 4000 | 1337 | AdamW | completed | 262,144,000 | 3.9691 | 3.9691 | 166088 |
| FineWeb-Edu-500M | 4000 | 1337 | Hybrid Muon | completed | 262,144,000 | 3.8571 | 3.8571 | 149368 |
| FineWeb-Edu-500M | 4000 | 2024 | AdamW | completed | 262,144,000 | 3.9634 | 3.9634 | 166129 |
| FineWeb-Edu-500M | 4000 | 2024 | Hybrid Muon | completed | 262,144,000 | 3.8553 | 3.8553 | 149530 |
| FineWeb-Edu-500M | 4000 | 2025 | AdamW | completed | 262,144,000 | 3.9918 | 4.0034 | 166100 |
| FineWeb-Edu-500M | 4000 | 2025 | Hybrid Muon | completed | 262,144,000 | 3.8810 | 3.8965 | 149515 |
| FineWeb-Edu-500M | 8000 | 1337 | AdamW | completed | 524,288,000 | 3.7377 | 3.7377 | 166069 |
| FineWeb-Edu-500M | 8000 | 1337 | Hybrid Muon | completed | 524,288,000 | 3.6614 | 3.6614 | 149741 |
| TinyStories | 4000 | 1337 | AdamW | completed | 262,144,000 | 1.4629 | 1.4629 | 166037 |
| TinyStories | 4000 | 1337 | Hybrid Muon | completed | 262,144,000 | 1.4129 | 1.4129 | 149293 |
| TinyStories | 4000 | 2024 | AdamW | completed | 262,144,000 | 1.4685 | 1.4797 | 166062 |
| TinyStories | 4000 | 2024 | Hybrid Muon | completed | 262,144,000 | 1.4122 | 1.4239 | 149748 |
| TinyStories | 4000 | 2025 | AdamW | completed | 262,144,000 | 1.4660 | 1.4884 | 166211 |
| TinyStories | 4000 | 2025 | Hybrid Muon | completed | 262,144,000 | 1.4098 | 1.4312 | 149516 |

## Paired optimizer deltas

Positive delta means Muon had lower validation loss than AdamW.

| Dataset | Steps | Seed | AdamW best | Muon best | AdamW - Muon | Relative improvement |
|---|---:|---:|---:|---:|---:|---:|
| FineWeb-Edu-500M | 4000 | 1337 | 3.9691 | 3.8571 | 0.1120 | 2.82% |
| FineWeb-Edu-500M | 4000 | 2024 | 3.9634 | 3.8553 | 0.1080 | 2.73% |
| FineWeb-Edu-500M | 4000 | 2025 | 3.9918 | 3.8810 | 0.1107 | 2.77% |
| FineWeb-Edu-500M | 8000 | 1337 | 3.7377 | 3.6614 | 0.0763 | 2.04% |
| TinyStories | 4000 | 1337 | 1.4629 | 1.4129 | 0.0500 | 3.41% |
| TinyStories | 4000 | 2024 | 1.4685 | 1.4122 | 0.0563 | 3.84% |
| TinyStories | 4000 | 2025 | 1.4660 | 1.4098 | 0.0562 | 3.83% |

## Aggregate by dataset/token budget

| Dataset | Steps | Seeds | AdamW mean±std | Muon mean±std | Mean delta | Mean relative improvement |
|---|---:|---:|---:|---:|---:|---:|
| FineWeb-Edu-500M | 4000 | 3 | 3.9747 ± 0.0150 | 3.8645 ± 0.0144 | 0.1103 ± 0.0020 | 2.77% |
| FineWeb-Edu-500M | 8000 | 1 | 3.7377 ± 0.0000 | 3.6614 ± 0.0000 | 0.0763 ± 0.0000 | 2.04% |
| TinyStories | 4000 | 3 | 1.4658 ± 0.0028 | 1.4116 ± 0.0016 | 0.0542 ± 0.0036 | 3.69% |

## Notes

- TinyStories now has a 3-seed result at 4000 steps.
- FineWeb-Edu has a 3-seed result at 4000 steps and a 1-seed longer 8000-step result.
- The FineWeb-Edu cache is a 500M-token extracted cache from sample-10BT, not the full 10B-token sample and not the full FineWeb-Edu dataset.
