# Seed-matched 57M benchmark: AdamW vs Hybrid Muon

This is the controlled rerun requested by the user. Both runs used the same seed and the same fixed token budget.

## Locked setup

```text
seed: 1337
model: MiniGPT-Dense-60M-v1, ~57.47M params
dataset: full TinyStories token cache
train cache tokens: 473,992,236
context length: 512
micro_batch_size: 16
grad_accum_steps: 8
effective_batch_tokens: 65,536
max_steps: 4,000
tokens per run: 262,144,000
precision: bf16
validation_tokens per eval: 262,144
```

| Run | Status | Seed | Steps | Tokens seen | Best val loss | Best step | Final val loss | Final train loss | Median tok/s | Wall-clock min | Max GPU GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | completed | 1337 | 4000 | 262,144,000 | 1.4629 | 4000 | 1.4629 | 1.4713 | 166037 | 27.0 | 7.17 |
| Hybrid Muon | completed | 1337 | 4000 | 262,144,000 | 1.4129 | 4000 | 1.4129 | 1.4112 | 149293 | 30.0 | 7.05 |

## Result

- Muon best validation loss was lower by **0.0500** absolute loss.
- Relative to AdamW best loss, that is about **3.41% lower**.
- AdamW throughput was about **10.1% faster** in tokens/sec.
- Both runs processed exactly the same token budget, so this comparison is cleaner than the previous fixed-time run.
- This is one seed. For a publishable claim, repeat with at least 3 seeds and report mean/std.
