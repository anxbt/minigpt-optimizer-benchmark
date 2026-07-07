# Pretraining Handholding Guide

This guide is the reference note for our from-scratch LLM pretraining work.

It is intentionally plain-language. The goal is not to sound academic. The goal
is that you can explain pretraining, seeds, steps, epochs, token budgets, batch
size, optimizer comparisons, and our current TinyStories/FineWeb-Edu experiments
without hand-waving.

---

## 1. The shortest correct definition of pretraining

**Pretraining** is the stage where a randomly initialized model learns general
patterns from large data using a self-supervised objective.

For a GPT-style text model, the objective is:

```text
Given previous tokens, predict the next token.
```

Example:

```text
Input:  The capital of France is
Target: Paris
```

The model is not explicitly given a labeled table like:

```text
France -> Paris
```

It learns by repeatedly predicting next tokens across huge amounts of text.

Good short answer:

> GPT pretraining trains a model to predict the next token from previous tokens.

---

## 2. Pretraining vs post-training

A useful mental model:

```text
pretraining  = learn broad language/world/code/math patterns
post-training = shape the model into a useful assistant
```

Pretraining teaches:

```text
grammar
facts
style
code patterns
math patterns
translation patterns
some reasoning patterns
world associations
```

Post-training teaches behavior:

```text
follow instructions
answer as an assistant
format responses
be concise or detailed when asked
refuse unsafe requests
prefer better answers over worse answers
use reasoning traces when useful
```

Post-training methods include:

```text
SFT   = supervised fine-tuning on instruction/answer examples
DPO   = preference optimization from chosen/rejected answers
PPO   = reinforcement learning style preference optimization
RLHF  = reinforcement learning from human feedback
QLoRA = memory-efficient fine-tuning method
```

Important correction:

```text
Post-training does not create all reasoning from zero.
```

Better:

```text
Pretraining learns many latent abilities.
Post-training elicits, shapes, and rewards useful behavior.
```

Base models are often autocomplete-like. Instruct/chat models are post-trained
to act like assistants.

---

## 3. Training from scratch vs continued pretraining

### Training from scratch

Training from scratch means:

```text
start with random model weights
train on next-token prediction
```

This is what our MiniGPT experiments do.

### Continued pretraining

Continued pretraining means:

```text
start with an already pretrained base model
continue next-token training on more text
```

Example:

```text
start from Qwen Base
continue train on domain-specific legal/medical/code text
```

Our current benchmark is **from scratch**, not continued pretraining:

```text
MiniGPT-Dense-60M-v1
~57.47M parameters
random initialization
trained on TinyStories and FineWeb-Edu token caches
```

---

## 4. Token, token ID, and embedding

This is a common confusion point.

### Token

A **token** is a text chunk.

Examples:

```text
"dog"
"ing"
" France"
"."
" pre"
```

A token is not always a full word. It can be a word, subword, punctuation mark,
space-prefixed word, or byte-like unit.

### Token ID

The tokenizer maps each token to an integer ID.

Example:

```text
"dog" -> token ID 121
```

The model does not read raw strings. It reads token IDs.

### Embedding

The token ID is then used to look up a vector from the embedding matrix.

```text
token ID 121 -> row 121 in embedding matrix -> vector
```

Think of the embedding matrix as a giant table:

```text
token ID 0     -> vector row 0
token ID 1     -> vector row 1
token ID 121   -> vector row 121
token ID 50256 -> vector row 50256
```

Correct distinction:

```text
token/token ID = discrete text unit / integer
embedding      = vector representation looked up from a matrix
```

Do **not** say:

```text
the token is divided into matrices
```

Say:

```text
the token ID indexes into the embedding matrix.
```

---

## 5. Tokenizer

A **tokenizer** converts text to token IDs.

Example:

```text
"hello world" -> [15339, 995]
```

Tokenizer choice affects:

```text
how efficiently text is compressed
how many tokens a document becomes
multilingual handling
code handling
rare character handling
vocabulary size
```

For our current benchmark we use:

```text
openai-community/gpt2 tokenizer
vocab size: 50,257
```

Reason:

```text
simple
stable
common baseline
fits in uint16 storage
works well enough for English text experiments
```

It is not necessarily the best modern tokenizer.

---

## 6. Context length

**Context length** is how many tokens the model can see at once.

Our current context length:

```text
512 tokens
```

The model trains on chunks like:

```text
x = tokens[i : i + 512]
y = tokens[i + 1 : i + 513]
```

So the model predicts the next token at every position.

Longer context means:

```text
more history visible
more memory use
more compute
slower training
```

---

## 7. One GPT training example

Suppose text is:

```text
The cat sat on the mat
```

Training input/target are shifted:

```text
x: The cat sat on the
y: cat sat on the mat
```

The model learns:

```text
The -> cat
cat -> sat
sat -> on
on -> the
the -> mat
```

During training, all positions are predicted in parallel. During generation,
tokens are produced one at a time.

---

## 8. Loss and validation loss

### Training loss

Loss measures how wrong the model was.

For LLMs, the usual loss is **cross-entropy loss**.

Simple meaning:

```text
Did the model assign high probability to the correct next token?
```

Lower loss is better.

### Validation loss

Validation loss is measured on held-out text that does not update the model.

It tells us whether the model is generalizing.

Important pattern:

```text
training loss down + validation loss down = learning
training loss down + validation loss up   = overfitting
```

Overfitting means:

```text
the model is memorizing training data instead of learning general patterns
```

Our 1M-token smoke cache overfit because:

```text
1M tokens was too small
model saw the same tiny cache many times
training loss fell
validation loss got worse
```

---

## 9. Seed

A **seed** is a number that initializes random number generators.

Example:

```text
seed = 1337
```

It controls randomness in the run:

```text
random initial model weights
random data chunk sampling
random generation samples
dropout randomness, if dropout is enabled
```

A seed is not a dataset.

Correct:

```text
seed 1337 = randomness-control value
FineWeb-Edu sample-10BT = dataset configuration/source
```

Why seeds matter:

If AdamW and Muon use different random starts, one may win by luck. So we run:

```text
AdamW seed 1337
Muon  seed 1337
```

Then repeat:

```text
seed 2024
seed 2025
```

If Muon wins across multiple seeds, the result is more credible.

---

## 10. Training step

A **training step** means one optimizer update.

During one step:

```text
1. sample token chunks
2. run model forward pass
3. compute loss
4. run backward pass to compute gradients
5. optimizer updates weights once
```

In our logs:

```text
step 4000
```

means:

```text
the model weights have been updated 4000 times
```

---

## 11. Batch size, micro-batch, and gradient accumulation

This is the most important math for reading training logs.

### Micro-batch size

Micro-batch size is how many sequences the GPU processes at once.

Our setting:

```text
micro_batch_size = 16
context_length = 512
```

Tokens per micro-batch:

```text
16 × 512 = 8192 tokens
```

### Gradient accumulation

Gradient accumulation means doing several micro-batches before one optimizer
update.

Our setting:

```text
grad_accum_steps = 8
```

So one optimizer step uses:

```text
8192 × 8 = 65,536 tokens
```

Full formula:

```text
tokens_per_step = context_length × micro_batch_size × grad_accum_steps
```

Our formula:

```text
512 × 16 × 8 = 65,536 tokens/step
```

### Total tokens after N steps

```text
tokens_seen = tokens_per_step × steps
```

For 4000 steps:

```text
65,536 × 4000 = 262,144,000 tokens
```

So every 4000-step run processes:

```text
~262.144M tokens
```

---

## 12. Epoch

An **epoch** means one full pass through the training dataset/cache.

Formula:

```text
epochs = tokens_seen / train_tokens_in_cache
```

Example with FineWeb-Edu cache:

```text
train cache = 500,000,000 tokens
tokens seen = 262,144,000 tokens
```

Epochs:

```text
262,144,000 / 500,000,000 ≈ 0.52 epochs
```

So the model saw about half of the 500M-token cache.

Why LLM people often talk about tokens/steps instead of epochs:

```text
web datasets are huge
training data is often streamed
one full epoch can be enormous
experiments are usually planned by token budget and compute budget
```

---

## 13. Token cache and token budget

### Raw dataset

Raw dataset is text documents.

Example:

```text
FineWeb-Edu sample-10BT
```

### Token cache

Token cache is pre-tokenized data saved to disk.

Example:

```text
train.bin
validation.bin
metadata.json
```

Why create a token cache?

```text
tokenization is CPU-heavy
GPU training wants fast token IDs
cache keeps GPU fed
```

Restaurant analogy:

> Token cache is like chopping vegetables before dinner service. You do not chop
> each onion after every order.

### Token budget

Token budget means how many tokens we decide to train on.

Example:

```text
4000 steps × 65,536 tokens/step = 262,144,000 training tokens
```

Token budget is independent of source dataset size.

---

## 14. TinyStories vs FineWeb-Edu in our experiments

### TinyStories full cache

We built:

```text
train_tokens:      473,992,236
validation_tokens:   4,765,918
```

A 4000-step run sees:

```text
262,144,000 tokens
```

Epoch fraction:

```text
262M / 474M ≈ 0.55 epochs
```

### FineWeb-Edu sample-10BT source

FineWeb-Edu `sample-10BT` is a source pool of about:

```text
10B GPT-2 tokens
```

We do not need to train on all 10B for the first experiment.

### Our FineWeb-Edu 500M-token cache

We extracted:

```text
train_tokens:      500,000,000
validation_tokens:   5,000,000
source: HuggingFaceFW/fineweb-edu / sample-10BT
```

This means:

```text
source dataset/config: ~10B tokens
our experiment cache:  500M train tokens
```

The 500M cache is a subset/cache made from the larger source.

It is **not** the full sample-10BT dataset.

---

## 15. AdamW vs Muon

### AdamW

AdamW is the standard optimizer baseline.

Simple mental model:

```text
updates parameters element-by-element
```

### Muon

Muon is matrix-aware for selected hidden matrix weights.

Hybrid Muon means:

```text
Muon for hidden matrix weights
AdamW for everything else
```

Muon gets:

```text
attention q/k/v/o matrices
MLP up/down matrices
```

AdamW fallback gets:

```text
token embeddings
position embeddings
LM head
LayerNorm weights
biases
vectors
scalars
non-2D parameters
```

### How to interpret our result

If:

```text
AdamW has higher tokens/sec
Muon has lower validation loss at same token count
```

then:

```text
AdamW is faster per wall-clock second
Muon is more token-efficient / sample-efficient
```

Do not call this overfitting.

Overfitting is:

```text
training loss improves while validation loss worsens
```

---

## 16. Current experiment ladder

We are following this ladder:

```text
1. toy local labs
2. 1M-token TinyStories smoke cache
3. full TinyStories cache
4. TinyStories 3-seed AdamW vs Muon benchmark
5. FineWeb-Edu 500M-token cache
6. FineWeb-Edu AdamW vs Muon benchmark
7. optional longer FineWeb-Edu runs
8. later: larger model, e.g. ~125M parameters
```

Why this ladder is correct:

```text
small runs catch bugs cheaply
full TinyStories validates pipeline
FineWeb-Edu gives more realistic web/education data
multiple seeds reduce luck
larger model comes after the training system is stable
```

---

## 17. Current confirmed results to remember

### Seed-matched TinyStories 57M, 4000-step runs

Setup:

```text
model: MiniGPT-Dense-60M-v1
parameters: ~57.47M
context_length: 512
micro_batch_size: 16
grad_accum_steps: 8
tokens_per_step: 65,536
steps: 4000
tokens_per_run: 262,144,000
```

Results observed so far:

```text
TinyStories seed 1337:
  AdamW best val loss: 1.4629
  Muon  best val loss: 1.4129

TinyStories seed 2024:
  AdamW best val loss: 1.4685
  Muon  best val loss: 1.4122

TinyStories seed 2025:
  AdamW best val loss: 1.4660
  Muon  best val loss: 1.4098
```

Interpretation:

```text
Muon beat AdamW on validation loss across these TinyStories seeds.
AdamW is faster in raw tokens/sec.
Muon appears more token-efficient in these runs.
```

### FineWeb-Edu 500M cache, 4000-step runs

Setup:

```text
source: HuggingFaceFW/fineweb-edu / sample-10BT
train cache: 500,000,000 tokens
validation cache: 5,000,000 tokens
steps: 4000
tokens_per_run: 262,144,000
```

Results observed so far:

```text
FineWeb-Edu seed 1337:
  AdamW best val loss: 3.9691
  Muon  best val loss: 3.8571

FineWeb-Edu seed 2024:
  AdamW best val loss: 3.9634
  Muon  best val loss: 3.8553

FineWeb-Edu seed 2025:
  AdamW best val loss: 3.9918
  Muon  best val loss: 3.8810
```

Interpretation:

```text
Muon beat AdamW on validation loss across these FineWeb-Edu 4000-step seeds.
AdamW remains faster in tokens/sec; Muon appears more token-efficient.
```

### FineWeb-Edu 500M cache, seed 1337, 8000-step longer run

```text
FineWeb-Edu seed 1337, 8000 steps:
  AdamW best val loss: 3.7377
  Muon  best val loss: 3.6614
  tokens_per_run: 524,288,000
```

Interpretation:

```text
The longer FineWeb-Edu seed-1337 run kept the same direction: Muon reached lower validation loss than AdamW.
This is still a small 57M single-GPU experiment, not a frontier-scale claim.
```

---

## 18. Main tunable knobs in pretraining

### Data knobs

```text
dataset source
data quality
filtering
deduplication
data mixture
token budget
train/validation split
```

Effect:

```text
controls what the model can learn
```

### Tokenizer knobs

```text
tokenizer algorithm
vocab size
special tokens
byte fallback
```

Effect:

```text
changes how text is represented and how many tokens data becomes
```

### Model architecture knobs

```text
parameter count
number of layers
hidden size
number of attention heads
MLP size
context length
dense vs MoE
positional embeddings
normalization type
```

Effect:

```text
changes capacity, speed, memory, and trainability
```

### Optimizer knobs

```text
optimizer type: AdamW, Muon, etc.
learning rate
weight decay
betas / momentum
gradient clipping
Muon Newton-Schulz steps
```

Effect:

```text
changes stability and how efficiently loss improves
```

### Schedule knobs

```text
warmup steps
cosine decay
total steps
minimum learning rate
```

Effect:

```text
controls how aggressively the model learns over time
```

### Batch knobs

```text
micro_batch_size
grad_accum_steps
effective_batch_tokens
```

Effect:

```text
changes GPU memory, speed, gradient noise, and optimizer behavior
```

### Precision/system knobs

```text
bf16/fp16/fp32
checkpoint frequency
evaluation frequency
data loader speed
number of GPUs
distributed strategy
```

Effect:

```text
changes runtime, stability, recovery, and cost
```

---

## 19. Industry pretraining pipeline, simplified

A real pretraining pipeline usually looks like:

```text
1. collect raw data
2. filter low-quality documents
3. deduplicate
4. classify / mix domains
5. tokenize
6. shard token files
7. choose model architecture
8. choose token budget
9. run small scaling experiments
10. tune learning rate, batch size, schedule
11. launch large distributed run
12. evaluate continuously
13. checkpoint frequently
14. recover from failures
15. post-train the base model
```

Most developers have more hands-on experience with post-training because:

```text
LoRA/QLoRA/RAG/SFT/DPO are cheaper
pretraining requires large data and compute
pretraining failures are expensive
pretraining infrastructure is complex
```

So yes: many people know post-training much better than pretraining.

---

## 20. Safe long-running training requirements

Before any serious multi-hour or multi-day run, we need:

```text
checkpoint saving
resume from checkpoint
STOP file safe exit
periodic validation
periodic sample generation
logs written to disk
OOM-safe batch settings
remote tmux execution
local rsync backup of lightweight artifacts
```

A checkpoint includes:

```text
model weights
optimizer state
step count
tokens seen
best validation loss
random number generator state
```

A STOP file lets us stop safely:

```bash
touch <run_dir>/STOP
```

The script should then:

```text
finish current safe point
save latest checkpoint
exit cleanly
```

---

## 21. Core formulas

Memorize these.

### Tokens per micro-batch

```text
tokens_per_micro_batch = context_length × micro_batch_size
```

Our run:

```text
512 × 16 = 8192
```

### Tokens per optimizer step

```text
tokens_per_step = context_length × micro_batch_size × grad_accum_steps
```

Our run:

```text
512 × 16 × 8 = 65,536
```

### Total tokens seen

```text
tokens_seen = tokens_per_step × steps
```

Our 4000-step run:

```text
65,536 × 4000 = 262,144,000
```

### Epochs

```text
epochs = tokens_seen / train_cache_tokens
```

FineWeb-Edu 500M cache example:

```text
262,144,000 / 500,000,000 ≈ 0.52 epochs
```

---

## 22. The final mental model

Use this as your compressed answer:

```text
Pretraining starts with random model weights and trains the model to predict the
next token on huge tokenized datasets. Text is converted to token IDs by a
tokenizer; token IDs index into an embedding matrix to become vectors. Training
is measured by optimizer steps and tokens seen. One step updates weights once
using an effective batch of tokens. Epochs mean full passes over a dataset, but
LLM pretraining usually talks more about token budgets because datasets are huge
and often streamed. Seeds control randomness so optimizer comparisons are fair.
Validation loss on held-out data tells us whether the model is generalizing.
Post-training then turns the base model into a helpful assistant.
```

---

## 23. Self-check quiz

You should be able to answer these quickly.

1. In GPT pretraining, what is the model predicting?
2. What is the difference between pretraining and post-training?
3. What is the difference between a token ID and an embedding?
4. What is a seed?
5. What is one optimizer step?
6. What is an epoch?
7. Why do LLM runs often talk about tokens instead of epochs?
8. Why did the 1M-token smoke cache overfit?
9. What is the difference between FineWeb-Edu sample-10BT and our 500M-token cache?
10. If Muon has lower validation loss at the same token count, what does that suggest?
11. If train loss goes down and validation loss goes up, what does that suggest?
12. Why do we run multiple seeds?
13. Compute tokens per step for context=512, micro_batch=16, grad_accum=8.
14. Compute total tokens for 4000 steps if each step has 65,536 tokens.
15. Compute epochs over a 500M-token cache if the model sees 262,144,000 tokens.

Answers:

```text
1. next token
2. pretraining learns broad patterns; post-training shapes assistant behavior
3. token ID is an integer; embedding is the vector row looked up by that ID
4. number controlling randomness
5. one weight update
6. one full pass through the train dataset/cache
7. datasets are huge/streamed; token budget is more practical
8. tiny data repeated too much; model memorized
9. sample-10BT is source pool; 500M cache is our extracted subset
10. Muon is more token-efficient in that run
11. overfitting
12. reduce chance the result is random luck
13. 512 × 16 × 8 = 65,536
14. 65,536 × 4000 = 262,144,000
15. 262,144,000 / 500,000,000 ≈ 0.52 epochs
```

---

## 24. Corrections from our latest quiz/discussion

This section exists because these are exactly the places where confusion tends
to come back later.

### Token vs embedding

Do not say:

```text
The token is divided into matrices.
```

Say:

```text
Text is split into tokens.
Each token becomes a token ID.
The token ID selects one row from the embedding matrix.
That selected row is the vector the model works with.
```

Concrete example:

```text
"dog" -> token ID 121
token ID 121 -> embedding_matrix[121] -> vector
```

The integer is not meaningful by itself. The learned embedding vector is what
lets the model represent the token inside neural-network math.

### Seed

Do not say:

```text
Seed is where training starts in the dataset.
```

Say:

```text
Seed is the number used to make randomness repeatable.
```

It affects:

```text
initial model weights
which chunks are sampled
dropout randomness, if dropout is used
generation randomness
```

Why we care:

```text
AdamW seed 1337 vs Muon seed 1337 is a fairer comparison than
AdamW seed 1337 vs Muon seed 2025.
```

Same seed does **not** mean same optimizer behavior. It means both optimizers
start from the same random setup, so the comparison is cleaner.

### FineWeb-Edu source vs our cache

Do not say:

```text
We trained on the whole FineWeb-Edu sample-10BT dataset.
```

Say:

```text
FineWeb-Edu sample-10BT is the larger source pool.
We created a 500M-token train cache and a 5M-token validation cache from it.
Our benchmark used that cache, not the entire source pool.
```

This matters because a benchmark claim must state the exact data budget:

```text
source: HuggingFaceFW/fineweb-edu / sample-10BT
train cache: 500M tokens
validation cache: 5M tokens
tokens per 4000-step run: 262.144M
```

### Lower validation loss is not automatically overfitting

If:

```text
AdamW validation loss = 3.96
Muon validation loss  = 3.85
same model
same data cache
same seed
same token budget
```

then the clean interpretation is:

```text
Muon got better validation performance per token in that run.
```

This is **not** overfitting.

Overfitting looks like:

```text
training loss goes down
validation loss goes up
```

Validation data is held out. If validation loss improves, the model is doing
better on data it was not directly trained on.

### Steps vs epochs

One optimizer step is:

```text
one weight update
```

One epoch is:

```text
one full pass through the training cache/dataset
```

For our current setup:

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

For a 500M-token train cache:

```text
epochs = 262,144,000 / 500,000,000 ≈ 0.52
```

So a 4000-step FineWeb-Edu run sees about half an epoch over our 500M-token
cache.

### The answer you should give if someone asks what we proved

Do not overclaim:

```text
We proved Muon is better than AdamW everywhere.
```

Say:

```text
On our single-GPU 57M-parameter MiniGPT benchmark, using the same architecture,
same token budget, same data cache, and seed-matched runs, Hybrid Muon reached
lower validation loss than AdamW on TinyStories and on our 500M-token
FineWeb-Edu cache. AdamW was faster in raw tokens/sec, while Muon looked more
token-efficient. This is a controlled small-scale benchmark, not a frontier-scale
proof.
```

That is the defensible claim.
