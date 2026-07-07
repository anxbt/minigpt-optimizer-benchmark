# minigpt-muon-benchmark

Single-GPU from-scratch GPT pretraining experiments comparing **AdamW** against **Hybrid Muon**.

This repository is both:

1. a learning project for understanding LLM pretraining from scratch, and
2. a small controlled benchmark of AdamW vs Hybrid Muon on MiniGPT models.

The benchmark is intentionally modest: one GPU, public datasets, small GPT-style models, seed-matched runs, and validation-loss comparisons. It is **not** a frontier-model reproduction.

---

## What this repo does

This repo trains small decoder-only GPT models from random initialization using next-token prediction.

It compares:

```text
AdamW
vs
Hybrid Muon
```

Hybrid Muon means:

```text
Muon for hidden 2D matrix weights:
  - attention q/k/v/o matrices
  - MLP matrices

AdamW fallback for everything else:
  - token embeddings
  - position embeddings
  - LM head
  - LayerNorm weights
  - biases
  - vectors/scalars
  - non-2D parameters
```

The main question:

> At the same model, seed, dataset cache, batch setting, and token budget, does Hybrid Muon reach lower validation loss than AdamW?

---

## Current results

### 57M MiniGPT

Model:

```text
MiniGPT-Dense-60M-v1
parameters: ~57.47M
context length: 512
tokenizer: openai-community/gpt2
precision: bf16
```

Training setup:

```text
micro_batch_size: 16
grad_accum_steps: 8
effective_batch_tokens: 65,536
steps per run: 4,000
tokens per run: 262,144,000
seeds: 1337, 2024, 2025
```

Aggregate results:

| Dataset | Steps | Seeds | AdamW mean ± std | Hybrid Muon mean ± std | Mean delta | Relative improvement |
|---|---:|---:|---:|---:|---:|---:|
| TinyStories | 4000 | 3 | 1.4658 ± 0.0028 | 1.4116 ± 0.0016 | 0.0542 | 3.69% |
| FineWeb-Edu 500M cache | 4000 | 3 | 3.9747 ± 0.0150 | 3.8645 ± 0.0144 | 0.1103 | 2.77% |
| FineWeb-Edu 500M cache | 8000 | 1 | 3.7377 | 3.6614 | 0.0763 | 2.04% |

Positive delta means AdamW validation loss minus Muon validation loss. Positive values mean Muon was lower.

### 125M MiniGPT

Model:

```text
MiniGPT-Dense-125M-v1
parameters: 123,963,648
context length: 512
tokenizer: openai-community/gpt2
precision: bf16
```

Dataset:

```text
source: HuggingFaceFW/fineweb-edu
config: sample-10BT
train cache: 500,000,000 tokens
validation cache: 5,000,000 tokens
```

Aggregate result over 3 seed-matched runs:

| Metric | AdamW | Hybrid Muon |
|---|---:|---:|
| Best validation loss mean ± std | 3.8056 ± 0.0146 | 3.7049 ± 0.0181 |
| Mean validation-loss delta | n/a | 0.1007 ± 0.0038 |
| Mean relative improvement | n/a | 2.65% |

Main observation so far:

> In these small single-GPU experiments, Hybrid Muon consistently reached lower validation loss than AdamW at the same token budget. AdamW remained faster in raw tokens/sec.

---

## What this does not prove

This repo does **not** prove that Muon is universally better than AdamW.

Limitations:

- models are small: ~57M and ~124M parameters
- runs are single-GPU
- context length is only 512 tokens
- FineWeb-Edu uses a 500M-token extracted cache, not the full dataset
- the implementation is educational/research-oriented, not production training infrastructure

A defensible claim is:

> On these seed-matched MiniGPT pretraining benchmarks, Hybrid Muon achieved lower validation loss than AdamW at the same model size, dataset cache, and token budget, while AdamW was faster in raw throughput.

---

## Repository structure

```text
.
├── gpu_benchmark/
│   ├── train_gpt.py                  # main pretraining script
│   ├── batch_sweep.py                # batch-size/memory sweep helper
│   ├── compare_optimizer_runs.py     # comparison/plotting helper
│   ├── README.md                     # GPU benchmark usage notes
│   ├── notebooks/                    # notebook runners for EC2 experiments
│   └── downloaded_runs/              # downloaded summaries, CSVs, plots
│
├── visual_labs/
│   ├── PRETRAINING_HANDHOLDING_GUIDE.md
│   ├── REAL_WORLD_ANALOGIES.md
│   ├── PROJECT_ROADMAP.md
│   ├── scripts/                      # VS Code / Jupyter-style # %% labs
│   └── notebooks/                    # notebook versions of the visual labs
│
├── reports/
│   ├── 57m_adamw_vs_muon_results.md
│   ├── 125m_adamw_vs_muon_seed1337_results.md
│   └── 125m_adamw_vs_muon_3seed_results.md
│
├── papers/
│   └── 2502.16982v1-muon-scalable-llm-training.*
│
└── spec.md                           # training process spec
```

---

## Core concepts

### Pretraining

Pretraining trains a randomly initialized model to predict the next token.

Example:

```text
input:  The capital of France is
target: Paris
```

### Token budget

We usually measure LLM training by tokens processed, not only by epochs.

For the main runs:

```text
context_length = 512
micro_batch_size = 16
grad_accum_steps = 8
```

So:

```text
tokens_per_step = 512 × 16 × 8 = 65,536
```

For 4000 steps:

```text
tokens_seen = 65,536 × 4000 = 262,144,000
```

### Seed-matched benchmark

A seed controls randomness: model initialization, data sampling, dropout, and generation randomness.

For optimizer comparisons, seed matching matters:

```text
AdamW seed 1337
Muon  seed 1337
```

This makes the comparison cleaner because both optimizers start from the same random setup.

---

## Datasets

### TinyStories

```text
source: roneneldan/TinyStories
train tokens: 473,992,236
validation tokens: 4,765,918
```

Used for early pipeline validation and small-model optimizer comparison.

### FineWeb-Edu

```text
source: HuggingFaceFW/fineweb-edu
config: sample-10BT
train cache: 500,000,000 tokens
validation cache: 5,000,000 tokens
```

This is an extracted local token cache from the larger `sample-10BT` source. It is not the full FineWeb-Edu dataset.

---

## Quick start

### Local toy smoke test

```bash
python3 gpu_benchmark/train_gpt.py \
  --dataset toy \
  --data-dir /tmp/minigpt_toy_data \
  --run-dir /tmp/minigpt_toy_adamw \
  --optimizer adamw \
  --model-config tiny_debug \
  --precision bf16 \
  --micro-batch-size 2 \
  --grad-accum-steps 2 \
  --max-steps 2 \
  --max-minutes 0 \
  --eval-every-steps 1 \
  --save-every-steps 1 \
  --validation-tokens 4096
```

### Muon toy smoke test

```bash
python3 gpu_benchmark/train_gpt.py \
  --dataset toy \
  --data-dir /tmp/minigpt_toy_data \
  --run-dir /tmp/minigpt_toy_muon \
  --optimizer muon \
  --model-config tiny_debug \
  --precision bf16 \
  --micro-batch-size 2 \
  --grad-accum-steps 2 \
  --max-steps 2 \
  --max-minutes 0 \
  --eval-every-steps 1 \
  --save-every-steps 1 \
  --validation-tokens 4096
```

---

## Safe long-running training

The training script supports:

- checkpoint saving
- resume from checkpoint
- STOP-file safe exit
- metrics JSONL logging
- periodic validation
- sample generation
- plot generation

To stop a run safely:

```bash
touch <run-dir>/STOP
```

The script should finish the current safe point, save the latest checkpoint, write summaries, and exit cleanly.

---

## Learning materials

If you are learning pretraining from scratch, start here:

```text
visual_labs/PRETRAINING_HANDHOLDING_GUIDE.md
```

Then read:

```text
visual_labs/REAL_WORLD_ANALOGIES.md
visual_labs/PROJECT_ROADMAP.md
visual_labs/notebooks/
visual_labs/scripts/
```

The visual labs cover:

- embeddings
- attention projections
- MLPs
- LM heads
- KV cache intuition
- MoE routing
- AdamW vs Muon toy experiments
- tiny GPT training

---

## Reports

Main result reports:

```text
reports/57m_adamw_vs_muon_results.md
reports/125m_adamw_vs_muon_3seed_results.md
```

Generated artifacts include CSVs and plots under:

```text
gpu_benchmark/downloaded_runs/
```

---

## Recommended repo name

Recommended GitHub repository name:

```text
minigpt-muon-benchmark
```

Reason:

- accurate: it is a MiniGPT benchmark
- specific: it is about Muon
- not overclaiming: it does not imply frontier-scale reproduction
- searchable: includes the relevant technical keywords

Alternative names:

```text
single-gpu-muon-pretraining
adamw-vs-muon-minigpt
muon-pretraining-lab
```

