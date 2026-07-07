# 57M MiniGPT AdamW vs Hybrid Muon Results

This report freezes the first completed benchmark before changing model size.

## Model

```text
name: MiniGPT-Dense-60M-v1
parameters: ~57.47M
layers: 10
hidden size: 512
attention heads: 8
head dim: 64
context length: 512
tokenizer: openai-community/gpt2
precision: bf16
```

## Optimizers

```text
AdamW:
  standard torch.optim.AdamW

Hybrid Muon:
  Muon for hidden 2D matrix weights:
    attention q/k/v/o matrices
    MLP up/down matrices
  AdamW fallback for:
    token embeddings
    position embeddings
    LayerNorm weights/biases
    non-2D params
```

## Batch and training setup

```text
micro_batch_size: 16
grad_accum_steps: 8
effective_batch_tokens: 65,536
steps per 4000-step run: 4,000
tokens per 4000-step run: 262,144,000
validation tokens per eval: 262,144
seeds: 1337, 2024, 2025
```

## Datasets

### TinyStories

```text
source: roneneldan/TinyStories
cache: ~/pretrain_data/tinystories_gpt2_512_full
train_tokens: 473,992,236
validation_tokens: 4,765,918
```

### FineWeb-Edu extracted cache

```text
source: HuggingFaceFW/fineweb-edu
config: sample-10BT
cache: ~/pretrain_data/finewebedu_sample10bt_gpt2_512_500m
train_tokens: 500,000,000
validation_tokens: 5,000,000
```

This is a 500M-token extracted cache from `sample-10BT`, not the full 10B-token sample and not the full FineWeb-Edu dataset.

## Aggregate results

| Dataset | Steps | Seeds | AdamW mean±std | Muon mean±std | Mean delta | Mean relative improvement |
|---|---:|---:|---:|---:|---:|---:|
| TinyStories | 4000 | 3 | 1.4658 ± 0.0028 | 1.4116 ± 0.0016 | 0.0542 ± 0.0036 | 3.69% |
| FineWeb-Edu-500M | 4000 | 3 | 3.9747 ± 0.0150 | 3.8645 ± 0.0144 | 0.1103 ± 0.0020 | 2.77% |
| FineWeb-Edu-500M | 8000 | 1 | 3.7377 | 3.6614 | 0.0763 | 2.04% |

Positive delta means AdamW validation loss minus Muon validation loss, so positive values mean Muon was lower.

## Artifacts

```text
gpu_benchmark/downloaded_runs/multi_seed_benchmark_summary.md
gpu_benchmark/downloaded_runs/multi_seed_runs.csv
gpu_benchmark/downloaded_runs/multi_seed_pairs.csv
gpu_benchmark/downloaded_runs/multi_seed_aggregate.csv
gpu_benchmark/downloaded_runs/multi_seed_delta_bars.png
gpu_benchmark/downloaded_runs/multi_seed_mean_val_loss.png
```

## Main conclusion

Across both TinyStories and the 500M-token FineWeb-Edu cache, Hybrid Muon consistently reached lower validation loss than AdamW at the same model size, seed, data cache, batch settings, and token budget.

AdamW remained faster in raw tokens/sec. Muon's signal here is better validation loss per token, not faster wall-clock speed.

## Limitations

- Model is only ~57M parameters.
- FineWeb-Edu experiment used a 500M-token extracted cache, not full FineWeb-Edu.
- The 8000-step FineWeb-Edu result is only one seed.
- This is single-GPU, small-scale evidence, not a frontier-scale reproduction.

## Next step

Move to `MiniGPT-Dense-125M-v1` to test whether the Muon advantage survives a GPT-2-small-scale model.
