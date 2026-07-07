# 125M MiniGPT AdamW vs Hybrid Muon — 3-seed report

## Setup

```text
model: MiniGPT-Dense-125M-v1
parameters: 123,963,648
dataset: HuggingFaceFW/fineweb-edu
fineweb_config: sample-10BT
train_tokens_cache: 500,000,000
validation_tokens_cache: 5,000,000
steps_per_run: 4,000
tokens_per_run: 262,144,000
micro_batch_size: 16
grad_accum_steps: 8
effective_batch_tokens: 65,536
```

## Individual runs

| Seed | Optimizer | Status | Best val loss | Last val loss | Median tok/s | Max GPU GB |
|---:|---|---|---:|---:|---:|---:|
| 1337 | AdamW | completed | 3.7986 | 3.7986 | 102927 | 9.63 |
| 1337 | Hybrid Muon | completed | 3.6944 | 3.6944 | 87915 | 9.29 |
| 2024 | AdamW | completed | 3.7958 | 3.7958 | 102807 | 9.63 |
| 2024 | Hybrid Muon | completed | 3.6946 | 3.6946 | 87901 | 9.29 |
| 2025 | AdamW | completed | 3.8224 | 3.8357 | 103040 | 9.63 |
| 2025 | Hybrid Muon | completed | 3.7258 | 3.7376 | 87897 | 9.29 |

## Paired deltas

Positive delta means Muon had lower validation loss than AdamW.

| Seed | AdamW best | Muon best | Delta | Relative improvement |
|---:|---:|---:|---:|---:|
| 1337 | 3.7986 | 3.6944 | 0.1042 | 2.74% |
| 2024 | 3.7958 | 3.6946 | 0.1012 | 2.67% |
| 2025 | 3.8224 | 3.7258 | 0.0966 | 2.53% |

## Aggregate

| Metric | AdamW | Hybrid Muon / Delta |
|---|---:|---:|
| Best validation loss mean±std | 3.8056 ± 0.0146 | 3.7049 ± 0.0181 |
| Delta mean±std | n/a | 0.1007 ± 0.0038 |
| Relative improvement mean | n/a | 2.65% |

## Main conclusion

Across 3 seed-matched 125M runs, Hybrid Muon reached lower best validation loss than AdamW at the same model, dataset cache, batch settings, and token budget. AdamW remained faster in raw tokens/sec.

## Artifacts

```text
gpu_benchmark/downloaded_runs/compare_125m_finewebedu_3seed_4000s/runs.csv
gpu_benchmark/downloaded_runs/compare_125m_finewebedu_3seed_4000s/pairs.csv
gpu_benchmark/downloaded_runs/compare_125m_finewebedu_3seed_4000s/best_val_loss_by_seed.png
gpu_benchmark/downloaded_runs/compare_125m_finewebedu_3seed_4000s/delta_by_seed.png
```
