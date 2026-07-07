# 125M MiniGPT AdamW vs Hybrid Muon Result — Seed 1337

This report freezes the first completed 125M-scale comparison.

## Model

```text
name: MiniGPT-Dense-125M-v1
parameters: 123,963,648
layers: 12
hidden size: 768
attention heads: 12
head dim: 64
context length: 512
tokenizer: openai-community/gpt2
precision: bf16
```

## Dataset

```text
source: HuggingFaceFW/fineweb-edu
config: sample-10BT
cache: ~/pretrain_data/finewebedu_sample10bt_gpt2_512_500m
train_tokens_in_cache: 500,000,000
validation_tokens_in_cache: 5,000,000
```

This is a 500M-token extracted cache from `sample-10BT`, not the full 10B-token sample and not the full FineWeb-Edu dataset.

## Training setup

```text
seed: 1337
max_steps: 4,000
micro_batch_size: 16
grad_accum_steps: 8
effective_batch_tokens: 65,536
tokens_seen_per_run: 262,144,000
validation_tokens_per_eval: 262,144
```

## Result

| Optimizer | Status | Step | Tokens | Best val loss | Last val loss | Median tok/s | Max GPU GB |
|---|---|---:|---:|---:|---:|---:|---:|
| AdamW | completed | 4,000 | 262,144,000 | 3.7986 | 3.7986 | 102,926 | 9.63 |
| Hybrid Muon | completed | 4,000 | 262,144,000 | 3.6944 | 3.6944 | 87,914 | 9.29 |

Positive delta means `AdamW best validation loss - Muon best validation loss`.

```text
best_val_loss_delta: 0.1042
relative_improvement: 2.74%
```

## Interpretation

At 125M parameters and the same seed/data/batch/token budget, Hybrid Muon reached lower validation loss than AdamW.

AdamW was faster in raw throughput. Muon's advantage here is lower validation loss per token, not faster wall-clock speed.

## Artifacts

```text
gpu_benchmark/downloaded_runs/finewebedu_125m_adamw_sample10bt_seed1337_4000s_001/
gpu_benchmark/downloaded_runs/finewebedu_125m_muon_sample10bt_seed1337_4000s_001/
gpu_benchmark/downloaded_runs/compare_125m_finewebedu_seed1337_4000s/summary.md
gpu_benchmark/downloaded_runs/compare_125m_finewebedu_seed1337_4000s/summary.csv
gpu_benchmark/downloaded_runs/compare_125m_finewebedu_seed1337_4000s/train_loss.png
gpu_benchmark/downloaded_runs/compare_125m_finewebedu_seed1337_4000s/val_loss.png
```

## Limitation

This is one seed. The next correctness step is to repeat the same 125M pair for seeds `2024` and `2025`.
