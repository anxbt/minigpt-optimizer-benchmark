# Training Process Spec

This document defines how we will move from toy local experiments to a real from-scratch pretraining benchmark on the EC2 L40S GPU.

The purpose of this spec is to prevent confusion, scope creep, and unsafe long-running training runs.

---

## 1. Overall goal

We want to train a small GPT-style language model **from scratch** and compare:

```text
AdamW vs Hybrid Muon
```

This is a single-GPU, small-scale benchmark.

We are **not** trying to reproduce Moonlight or train a frontier model.

The honest claim we want to support is:

> We trained small GPT models from scratch on public data on a single L40S and compared AdamW vs Hybrid Muon using validation loss, tokens seen, approximate FLOPs, update RMS, and checkpoint-safe training.

---

## 2. Key definitions

### Pretraining

Pretraining means training a model on raw text with the objective:

```text
Given previous tokens, predict the next token.
```

Example:

```text
Input:  The cat sat on the
Target: mat
```

### From scratch

From scratch means:

```text
model weights start random
```

We are not starting from Qwen, TinyLlama, or any other pretrained checkpoint.

### AdamW

AdamW is the standard optimizer. It updates parameters element-by-element.

### Hybrid Muon

Hybrid Muon means:

```text
Muon for hidden matrix weights
AdamW for everything else
```

Muon parameters:

```text
attention q/k/v/o matrices
MLP up/gate/down matrices
```

AdamW fallback parameters:

```text
token embeddings
position embeddings
LM head
norm weights
biases
scalars
vectors
any non-2D parameter
```

---

## 3. Hardware target

Primary machine:

```text
EC2 instance with NVIDIA L40S
~44 GiB usable GPU memory in PyTorch
~300GB root disk, with limited free space
```

Local machine:

```text
used for code editing, small tests, plotting, and backup inspection
```

Workflow:

```text
local = development and small smoke tests
EC2 L40S = serious GPU training
```

---

## 4. Dataset plan

We will use public datasets in phases.

### Phase 1: TinyStories

Dataset:

```text
roneneldan/TinyStories
```

Why:

```text
small enough to handle easily
good for small language model training
has simple generated stories
useful for debugging the pretraining pipeline
```

Purpose:

```text
smoke tests
first AdamW vs Muon benchmark
checkpoint/resume testing
```

Expected limitation:

```text
TinyStories is not a general web-scale dataset.
Results are useful for small-model behavior, not frontier claims.
```

### Phase 2: FineWeb-Edu sample

Dataset:

```text
HuggingFaceFW/fineweb-edu
```

Likely subset/config:

```text
sample-10BT or a smaller streamed/sample shard
```

Why:

```text
more realistic pretraining text
educational/web-style data
better benchmark signal than toy data
```

Risk:

```text
larger download/cache size
more disk pressure
slower preprocessing
```

Use only after Phase 1 is stable.

---

## 5. Tokenization plan

Tokenizer converts raw text to token IDs.

Initial tokenizer:

```text
GPT-2 tokenizer
```

Why:

```text
common baseline
works on English text
easy to use from Hugging Face
```

We will log:

```text
tokenizer name
vocabulary size
special tokens
number of train tokens
number of validation tokens
```

---

## 6. Model plan

We will use a decoder-only GPT-style Transformer.

### Smoke model

Purpose:

```text
verify code, data, checkpointing, STOP file, resume
```

Approximate size:

```text
10M–20M parameters
context length 256 or 512
```

### First serious model

Purpose:

```text
first meaningful AdamW vs Muon benchmark
```

Approximate size:

```text
50M–70M parameters
context length 512
```

### Stretch model

Only after smaller models are stable.

Approximate size:

```text
100M–125M parameters
context length 1024
```

We should not start with the stretch model.


---

## 6A. Locked architecture for first benchmark: `MiniGPT-Dense-60M-v1`

We will use one fixed architecture for the first serious from-scratch benchmark.

Name:

```text
MiniGPT-Dense-60M-v1
```

Model family:

```text
GPT-style decoder-only Transformer
```

This is not exactly GPT-2, LLaMA, Qwen, or TinyLlama. It is a small custom GPT-style model designed for a clean AdamW vs Muon benchmark.

### Fixed architecture values

```text
tokenizer: GPT-2 byte-level BPE tokenizer
vocab_size: 50,257
context_length / block_size: 512 tokens
number_of_layers / n_layer: 10
hidden_size / n_embd: 512
attention_heads / n_head: 8
head_dim: 64
attention_type: causal multi-head self-attention
position_embedding: learned absolute position embedding
normalization: pre-LayerNorm
MLP type: GELU MLP
MLP expansion: 4x
MLP hidden size: 2,048
linear_bias: false for attention and MLP linear layers
LayerNorm affine: true
LM head: tied to token embedding weight
dropout: 0.0 for benchmark simplicity
precision: bf16 on L40S
```

### Layer stack

The model is:

```text
token IDs
  ↓
token embedding + position embedding
  ↓
Transformer Block × 10
  ↓
final LayerNorm
  ↓
LM head using tied token embedding weight
  ↓
next-token logits
```

Each Transformer block is:

```text
input hidden state
  ↓
LayerNorm
  ↓
causal self-attention:
    q_proj: 512 → 512
    k_proj: 512 → 512
    v_proj: 512 → 512
    o_proj: 512 → 512
  ↓
residual add
  ↓
LayerNorm
  ↓
MLP:
    up_proj:   512 → 2048
    GELU
    down_proj: 2048 → 512
  ↓
residual add
```

### Expected parameter scale

Approximate total parameters with tied LM head:

```text
~57M–58M total parameters
~31M–32M non-embedding parameters
```

The exact count must be logged by the training script.

### Muon parameter selection for this architecture

Muon will update these hidden matrix weights:

```text
blocks.*.attn.q_proj.weight  shape [512, 512]
blocks.*.attn.k_proj.weight  shape [512, 512]
blocks.*.attn.v_proj.weight  shape [512, 512]
blocks.*.attn.o_proj.weight  shape [512, 512]
blocks.*.mlp.up_proj.weight  shape [2048, 512]
blocks.*.mlp.down_proj.weight shape [512, 2048]
```

AdamW will update everything else:

```text
token_embedding.weight / tied LM head
position_embedding.weight
LayerNorm weights and biases
any scalar/vector/non-2D parameter
```

### Why this architecture

We choose this architecture because:

```text
small enough for one L40S
large enough to be more meaningful than toy notebooks
simple enough to debug
has clear hidden matrix weights for Muon
uses standard GPT-style components
lets us run both AdamW and Muon comparisons within limited GPU time
```

### What we are not using initially

For the first benchmark, we are not using:

```text
MoE
GQA/MQA
SwiGLU
RoPE
Qwen architecture
LLaMA architecture
TinyLlama architecture
long context >512
```

Those can be added later only after the dense benchmark works.

---

## 7. Smoke test plan

No long run may start until smoke tests pass.

### 7.1 Local smoke test

Run locally with a tiny config.

Goal:

```text
script imports
model builds
data loads
one training step runs
loss is finite
metrics file is written
```

Acceptance:

```text
no crash
no NaN loss
config/log files created
```

### 7.2 EC2 5–10 minute smoke test

Run on L40S with a small model and small batch.

Checks:

```text
GPU detected
model fits in GPU memory
forward pass works
backward pass works
optimizer step works
loss is finite
metrics.jsonl is written
checkpoint latest.pt is written
STOP file works
resume works
```

Acceptance:

```text
training reaches at least a few hundred steps or a fixed short time
loss does not become NaN
GPU memory does not grow uncontrollably
safe STOP saves checkpoint and exits
resume continues from the checkpoint
```

### 7.3 EC2 1-hour smoke test

Run once with AdamW and once with Hybrid Muon.

Checks:

```text
throughput is stable
validation runs
checkpointing runs
plots can be generated
rsync backup works
Muon update RMS is logged for Muon run
```

Acceptance:

```text
no OOM
no NaN
validation loss exists
local backup contains config, metrics, summary, checkpoint
```

### 7.4 Long run gate

Only after all smoke tests pass can we start:

```text
12h / 24h / 48h / 72h runs
```

---

## 8. Out-of-memory prevention plan

OOM means out-of-memory: the GPU cannot fit the model, activations, gradients, and optimizer state.

We prevent OOM in several ways.

### 8.1 Start conservative

Start with smaller values:

```text
smaller model
shorter context length
smaller micro-batch size
```

Increase only after stable runs.

### 8.2 Use micro-batches and gradient accumulation

Definitions:

```text
micro_batch_size = sequences processed at once
grad_accum_steps = number of micro-batches before optimizer update
block_size = context length in tokens
```

Effective tokens per optimizer step:

```text
micro_batch_size × block_size × grad_accum_steps
```

This lets us keep GPU memory low while still training with a larger effective batch.

### 8.3 Use mixed precision

Preferred precision:

```text
bf16
```

bf16 uses less memory than fp32 and is usually stable on modern GPUs.

Fallback:

```text
fp16 with gradient scaler
```

### 8.4 Automatic micro-batch fallback

Training script should be able to try a safe batch size.

Example strategy:

```text
try micro_batch_size = 8
if OOM, try 4
if OOM, try 2
if OOM, try 1
```

After an OOM, the script should clear CUDA cache before retrying.

### 8.5 Memory logging

Every run should log:

```text
GPU memory allocated
GPU memory reserved
max GPU memory allocated
```

### 8.6 Avoid huge checkpoint explosion

Checkpoints include optimizer state and can be large.

Policy:

```text
always keep latest checkpoint
optionally keep last N numbered checkpoints
prune older frequent checkpoints
```

### 8.7 Do not start with long context

Do not start at:

```text
context length 2048+
```

Start with:

```text
512
```

Move to 1024 only after stable.

---

## 9. Checkpoint and resume plan

### 9.1 Checkpoint contents

Each checkpoint must contain:

```text
model weights
optimizer state
scheduler state if used
step number
tokens seen
config
random number generator state
precision scaler state if fp16 is used
best validation loss if tracked
```

### 9.2 Checkpoint location

Remote EC2 run directory:

```text
~/pretrain_runs/<run_id>/checkpoints/latest.pt
```

Optional numbered checkpoints:

```text
~/pretrain_runs/<run_id>/checkpoints/step_00010000.pt
```

### 9.3 Resume behavior

Resume command should look like:

```bash
python train_gpt.py --run-dir ~/pretrain_runs/<run_id> --resume latest
```

Expected behavior:

```text
load config
load model weights
load optimizer state
load scheduler/scaler if present
restore step and tokens_seen
append to existing metrics.jsonl
continue training
```

Resume must not overwrite previous logs.

---

## 10. Safe pause / STOP file plan

We need a safe way to pause training and free the GPU.

Bad pause:

```text
kill -STOP <pid>
```

This freezes the process but keeps GPU memory occupied.

Good pause:

```bash
touch ~/pretrain_runs/<run_id>/STOP
```

Training script must check for this file regularly.

When STOP exists, the script must:

```text
finish current optimizer step
save latest checkpoint
write run_summary.json
write artifact_manifest.json
print safe-stop message
exit with code 0
free GPU memory
```

This allows us to:

```text
pause training
use GPU for another task
resume later
sync data locally
```

---

## 11. Local rsync backup plan

We need local backups because EC2 data can be lost, deleted, or reset.

### 11.1 Sync run artifacts

From local repo root:

```bash
mkdir -p gpu_benchmark/downloaded_runs
rsync -avz --partial --progress \
  anubrat@13.234.232.112:/home/anubrat/pretrain_runs/<run_id>/ \
  ./gpu_benchmark/downloaded_runs/<run_id>/
```

This should download:

```text
config.json
metrics.jsonl
run_summary.json
artifact_manifest.json
plots
sample generations
checkpoints
logs
```

### 11.2 Sync tokenized data

If we build tokenized dataset shards, sync them separately:

```bash
mkdir -p gpu_benchmark/downloaded_data
rsync -avz --partial --progress \
  anubrat@13.234.232.112:/home/anubrat/pretrain_data/<dataset_cache_name>/ \
  ./gpu_benchmark/downloaded_data/<dataset_cache_name>/
```

### 11.3 Minimum backup requirement

After every smoke or serious run, local machine must have:

```text
config.json
metrics.jsonl
run_summary.json
latest checkpoint if resume is needed
plots if generated
```

Before terminating EC2:

```text
safe stop all runs
sync all important run directories
verify local files exist
optionally sync tokenized data shards
```

---

## 12. Logging plan

Each run must write:

```text
config.json
metrics.jsonl
run_summary.json
artifact_manifest.json
sample_generations.txt
```

### 12.1 config.json

Contains:

```text
run id
hardware
model config
dataset config
tokenizer config
optimizer config
batch settings
precision
```

### 12.2 metrics.jsonl

One JSON record per event.

Training event fields:

```text
step
tokens_seen
estimated_flops
train_loss
learning_rate
tokens_per_second
gpu_memory_allocated
gpu_memory_reserved
wall_clock_seconds
```

Validation event fields:

```text
step
tokens_seen
validation_loss
validation_tokens
```

Checkpoint event fields:

```text
step
checkpoint_path
checkpoint_size_bytes
```

Safe stop event fields:

```text
step
reason
checkpoint_path
```

### 12.3 Muon-specific logs

For Muon runs:

```text
muon matrix count
muon parameter count
AdamW fallback parameter count
average update RMS
min update RMS
max update RMS
selected Muon parameter names
```

---

## 13. Comparison plan

For AdamW vs Muon, keep the following identical:

```text
same dataset
same tokenizer
same validation split
same model config
same context length
same effective batch tokens if possible
same number of training tokens or same wall-clock budget, depending experiment
```

Primary plots:

```text
validation loss vs tokens seen
validation loss vs approximate FLOPs
training loss vs step
Muon update RMS vs step
```

Approximate FLOPs:

```text
training_flops ≈ 6 × non_embedding_parameters × tokens_seen
```

This is approximate and must be labeled as approximate.

---

## 14. Run stages

### Stage A: Implementation

Build training script with:

```text
AdamW
Hybrid Muon
checkpoint/resume
STOP file
logging
rsync-compatible run dirs
```

### Stage B: Local smoke

Tiny model, tiny data, local run.

### Stage C: EC2 smoke

Short L40S run.

### Stage D: One-hour runs

```text
1h AdamW
1h Muon
```

### Stage E: First serious benchmark

```text
12–24h AdamW
12–24h Muon
```

### Stage F: Extended benchmark

Only if Stage E is stable:

```text
48–72h total budget split across AdamW, Muon, and possibly ablations
```

---

## 15. Claim boundaries

Allowed claim:

> We ran a from-scratch small GPT pretraining benchmark on one L40S and compared AdamW vs Hybrid Muon with validation loss, tokens/FLOPs, checkpoint-safe training, and update RMS logging.

Not allowed claim:

> We reproduced Moonlight.

Not allowed claim:

> We proved Muon is 2x better at LLM scale.

Not allowed claim:

> This model is competitive with modern open LLMs.

---

## 16. Next implementation step

After this spec is accepted, implement:

```text
gpu_benchmark/train_gpt.py
```

Minimum first version:

```text
TinyStories loader
small GPT model
AdamW optimizer
Hybrid Muon optimizer
checkpoint save/load
STOP file
metrics.jsonl
run_summary.json
local/remote-friendly run directory
```

No long run starts until smoke tests pass.

---

## 17. Locked first benchmark configuration: `MiniGPT-Dense-60M-v1 on TinyStories`

This section freezes all first-benchmark choices so implementation does not drift.

### 17.1 Dataset

First benchmark dataset:

```text
roneneldan/TinyStories
```

Training split:

```text
train
```

Validation split:

```text
validation
```

Data access mode for smoke tests:

```text
Hugging Face datasets streaming or direct download cache
```

Data access mode for long runs:

```text
pre-tokenized local shards under ~/pretrain_data/tinystories_gpt2_512/
```

Shard format:

```text
.bin files containing uint16 or int32 token IDs, depending vocab requirements
metadata.json with token counts and source dataset info
```

Since GPT-2 vocab size is 50,257, uint16 is sufficient because max token id is below 65,535.

Required dataset logs:

```text
dataset_name: roneneldan/TinyStories
train_split: train
validation_split: validation
tokenized_cache_dir: ~/pretrain_data/tinystories_gpt2_512/
train_tokens
validation_tokens
num_train_shards
num_validation_shards
```

### 17.2 Tokenizer

Locked tokenizer for first benchmark:

```text
openai-community/gpt2
```

Tokenizer type:

```text
byte-level BPE
```

Vocabulary size:

```text
50,257
```

Reason for first benchmark:

```text
simple, common, stable, English-friendly, works well enough for TinyStories
```

Tokenizer comparisons are explicitly postponed until after AdamW vs Muon training works.

### 17.3 Model architecture

Locked model:

```text
MiniGPT-Dense-60M-v1
```

Architecture values are defined in Section 6A and must not be changed for the first AdamW vs Muon comparison.

### 17.4 Precision

Locked precision on L40S:

```text
bf16 mixed precision
```

Fallback only if bf16 fails unexpectedly:

```text
fp32 for debugging only
```

We will not use fp16 for the first benchmark unless forced, because bf16 is simpler and usually more stable.

### 17.5 Batch settings

Smoke batch settings:

```text
micro_batch_size: 4
grad_accum_steps: 8
block_size: 512
effective_batch_tokens: 4 × 8 × 512 = 16,384 tokens/update
```

Verified serious batch settings after L40S sweep:

```text
micro_batch_size: 16
grad_accum_steps: 8
block_size: 512
effective_batch_tokens: 16 × 8 × 512 = 65,536 tokens/update
```

Why this setting is locked for the first full TinyStories benchmark:

```text
AdamW: fastest tested setting, ~166k tokens/sec, ~7.17GB max allocated
Hybrid Muon: nearly fastest tested setting, ~147k tokens/sec, ~7.05GB max allocated
VRAM safety margin: large; L40S has ~44GiB usable in PyTorch
```

Rejected settings:

```text
micro_batch_size 4: works but underfills GPU
micro_batch_size 8: works but slower than 16
micro_batch_size 32: works and is close, but uses ~13GB and is not clearly better
micro_batch_size 64: works but slower and uses ~25GB
micro_batch_size 128: OOM for AdamW
```

OOM fallback order for future larger models:

```text
micro_batch_size 16 → 8 → 4 → 2 → 1
```

If micro_batch_size is reduced, increase grad_accum_steps when possible to keep effective batch tokens close to 65,536.

### 17.6 Optimizer settings

AdamW first benchmark settings:

```text
optimizer: AdamW
learning_rate: 6e-4
weight_decay: 0.1
betas: (0.9, 0.95)
epsilon: 1e-8
```

Hybrid Muon first benchmark settings:

```text
optimizer: Hybrid Muon
learning_rate: 6e-4
weight_decay: 0.1
muon_momentum: 0.95
newton_schulz_steps: 5
rms_scale: 0.2 * sqrt(max(rows, cols))
adamw_fallback_betas: (0.9, 0.95)
adamw_fallback_epsilon: 1e-8
```

Important:

```text
The first Muon run uses the same learning rate and weight decay as AdamW.
```

This follows the Muon paper's idea that scaled Muon can reuse AdamW-style hyperparameters.

### 17.7 Learning-rate schedule

Locked first schedule:

```text
warmup_steps: 500
schedule: cosine decay
min_lr_ratio: 0.1
```

Meaning:

```text
LR starts near 0
linearly warms to 6e-4 over 500 steps
then cosine decays toward 6e-5
```

For very short smoke tests, schedule still runs but may only cover warmup.

### 17.8 Evaluation settings

Validation batch settings:

```text
validation_micro_batch_size: same as training micro_batch_size if memory allowhttps://file+.vscode-resource.vscode-cdn.net/Users/rishav/Desktop/lean-ml/figures/final_figure_contact_sheet.png?version%3D1783352329257s
validation_tokens_per_eval: 262,144 tokens for smoke/early runs
validation_tokens_per_eval_long: 1,048,576 tokens for serious runs
```

Evaluation cadence:

```text
smoke: every 100 steps
serious: every 500 steps
```

Validation split must be identical for AdamW and Muon.

### 17.9 Checkpoint settings

Smoke checkpoint cadence:

```text
save_every_steps: 200
```

Serious checkpoint cadence:

```text
save_every_steps: 1000
```

Always save:

```text
checkpoints/latest.pt
```

Optional numbered checkpoints:

```text
checkpoints/step_<step>.pt every save_every_steps
```

Checkpoint pruning:

```text
keep latest.pt
keep last 2 numbered checkpoints
keep best validation checkpoint if enabled
```

### 17.10 Logging cadence

Training metrics cadence:

```text
log_every_steps: 10
```

Muon update RMS cadence:

```text
log every optimizer step internally, write aggregate every log_every_steps
```

Sample generation cadence:

```text
smoke: every evaluation
serious: every 1000 steps
```

Prompts for sample generation:

```text
"Once upon a time"
"The little girl"
"In a small village"
```

### 17.11 Run IDs

First smoke run IDs:

```text
tinystories_60m_adamw_smoke_001
tinystories_60m_muon_smoke_001
```

First one-hour run IDs:

```text
tinystories_60m_adamw_1h_001
tinystories_60m_muon_1h_001
```

First serious run IDs:

```text
tinystories_60m_adamw_main_001
tinystories_60m_muon_main_001
```

### 17.12 Run duration gates

Order:

```text
1. local import/unit smoke
2. EC2 AdamW 10-minute smoke
3. EC2 Muon 10-minute smoke
4. STOP file test
5. resume test
6. rsync backup test
7. EC2 AdamW 1-hour run
8. EC2 Muon 1-hour run
9. inspect curves and logs
10. launch 12–24h AdamW main
11. launch 12–24h Muon main
```

No 48–72h run starts until the 12–24h runs look stable.

### 17.13 Exact first command shape

AdamW smoke command shape:

```bash
python gpu_benchmark/train_gpt.py \
  --run-id tinystories_60m_adamw_smoke_001 \
  --run-dir ~/pretrain_runs/tinystories_60m_adamw_smoke_001 \
  --dataset tinystories \
  --tokenizer openai-community/gpt2 \
  --model-config minigpt_dense_60m_v1 \
  --optimizer adamw \
  --precision bf16 \
  --micro-batch-size 4 \
  --grad-accum-steps 8 \
  --max-minutes 10 \
  --eval-every-steps 100 \
  --save-every-steps 200
```

Muon smoke command shape:

```bash
python gpu_benchmark/train_gpt.py \
  --run-id tinystories_60m_muon_smoke_001 \
  --run-dir ~/pretrain_runs/tinystories_60m_muon_smoke_001 \
  --dataset tinystories \
  --tokenizer openai-community/gpt2 \
  --model-config minigpt_dense_60m_v1 \
  --optimizer muon \
  --precision bf16 \
  --micro-batch-size 4 \
  --grad-accum-steps 8 \
  --max-minutes 10 \
  --eval-every-steps 100 \
  --save-every-steps 200
```

### 17.14 First benchmark success criteria

The first benchmark is successful if we produce:

```text
AdamW metrics.jsonl
Muon metrics.jsonl
AdamW run_summary.json
Muon run_summary.json
AdamW latest checkpoint
Muon latest checkpoint
loss vs tokens plot
loss vs approximate FLOPs plot
Muon update RMS plot
muon_param_groups.json
local rsync backup of both runs
```

And both runs satisfy:

```text
no OOM
no NaN
validation loss logged
STOP/resume tested before long run
```


---

## Verified run notes: 2026-07-05 full TinyStories 30-minute runs

Full TinyStories cache built on EC2:

```text
cache_dir: ~/pretrain_data/tinystories_gpt2_512_full
train_tokens: 473,992,236
validation_tokens: 4,765,918
tokenizer: openai-community/gpt2
block_size: 512
```

First operational 30-minute comparison, not final paper claim because explicit seed control was added after these two runs:

```text
AdamW:
  run_dir: ~/pretrain_runs/tinystories_60m_adamw_fullts_30m_001
  steps: 4,439
  tokens_seen: 290,914,304
  best_val_loss: 1.4898
  median throughput: ~165,960 tokens/sec

Hybrid Muon:
  run_dir: ~/pretrain_runs/tinystories_60m_muon_fullts_30m_001
  steps: 4,009
  tokens_seen: 262,733,824
  best_val_loss: 1.4282
  median throughput: ~149,605 tokens/sec
```

Seed-control requirement for paper-style claims:

```text
Use --seed with the same value for AdamW and Muon reruns.
Recommended seed for next controlled benchmark: 1337.
```

---

## Verified run notes: 2026-07-05 seed-matched 57M control benchmark

Controlled benchmark command shape:

```text
same seed: 1337
same model: MiniGPT-Dense-60M-v1
same dataset: full TinyStories token cache
same batch: micro_batch_size=16, grad_accum_steps=8
same effective batch: 65,536 tokens/update
same max_steps: 4,000
same token budget: 262,144,000 tokens/run
same precision: bf16
```

Results:

```text
AdamW:
  run_dir: ~/pretrain_runs/tinystories_60m_adamw_fullts_seed1337_4000s_001
  status: completed
  steps: 4,000
  tokens_seen: 262,144,000
  best_val_loss: 1.4629
  final_val_loss: 1.4629
  median throughput: ~166,037 tokens/sec
  wall_clock: ~27.0 min

Hybrid Muon:
  run_dir: ~/pretrain_runs/tinystories_60m_muon_fullts_seed1337_4000s_001
  status: completed
  steps: 4,000
  tokens_seen: 262,144,000
  best_val_loss: 1.4129
  final_val_loss: 1.4129
  median throughput: ~149,293 tokens/sec
  wall_clock: ~30.0 min
```

Interpretation:

```text
Muon had ~0.0500 lower absolute validation loss.
That is ~3.41% lower relative to AdamW best validation loss.
AdamW was ~10.1% faster in tokens/sec.
This is one seed only. For publishable claims, repeat with at least 3 seeds and report mean/std.
```

---

## Locked next model: MiniGPT-Dense-125M-v1

Purpose:

```text
Test whether the AdamW vs Hybrid Muon signal survives a GPT-2-small-scale model.
```

Architecture:

```text
name: MiniGPT-Dense-125M-v1
layers: 12
hidden size: 768
attention heads: 12
head dim: 64
MLP hidden size: 3072
context length: 512 initially
tokenizer: openai-community/gpt2
linear bias: false
tied LM head: true
```

Parameter count with GPT-2 vocab:

```text
total parameters: 123,963,648
embedding parameters: 38,990,592
non-embedding parameters: 84,973,056
```

First validation sequence:

```text
1. 5-step smoke test on EC2
2. batch sweep on FineWeb-Edu 500M cache
3. seed 1337 AdamW vs Muon 4000-step run
4. if clean, seeds 2024 and 2025
```
