# 125M MiniGPT AdamW vs Hybrid Muon comparison

Positive delta means `AdamW best validation loss - Muon best validation loss`.
Positive is good for Muon; negative is bad for Muon.

## Setup

```text
model_config: MiniGPT-Dense-125M-v1
parameter_count: 123963648
dataset: HuggingFaceFW/fineweb-edu
fineweb_config: sample-10BT
train_tokens_in_cache: 500000000
validation_tokens_in_cache: 5000000
seed: 2025
max_steps: 4000
micro_batch_size: 16
grad_accum_steps: 8
effective_batch_tokens: 65536
block_size: 512
```

## Run table

| Optimizer | Status | Step | Tokens | Best val loss | Last val loss | Median tok/s | Max GPU GB |
|---|---|---:|---:|---:|---:|---:|---:|
| AdamW | completed | 4,000 | 262,144,000 | 3.8224 | 3.8357 | 103,040 | 9.63 |
| Hybrid Muon | completed | 4,000 | 262,144,000 | 3.7258 | 3.7376 | 87,896 | 9.29 |

## Paired result

- Best validation-loss delta: `0.0966`
- Relative improvement: `2.53%`

## Plots

- `train_loss.png`
- `val_loss.png`

## Interpretation guardrail

This is a one-seed 125M check. It tells us whether the 57M result still appears at GPT-2-small scale, but it is not yet a statistically stable claim. For a stronger claim, repeat this exact pair for seeds `2024` and `2025`.
