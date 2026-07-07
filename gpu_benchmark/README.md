# GPU Benchmark Code

This folder contains the from-scratch pretraining benchmark code.

Main script:

```text
gpu_benchmark/train_gpt.py
```

It implements:

- MiniGPT-Dense-60M-v1 / tiny_debug GPT models
- AdamW optimizer
- Hybrid Muon optimizer
- tokenized dataset cache
- checkpoint/resume
- STOP-file safe exit
- metrics JSONL logging
- plots
- sample generations

## First local/EC2 toy smoke

```bash
python gpu_benchmark/train_gpt.py \
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

Muon:

```bash
python gpu_benchmark/train_gpt.py \
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

## Resume

```bash
python gpu_benchmark/train_gpt.py \
  --dataset toy \
  --data-dir /tmp/minigpt_toy_data \
  --run-dir /tmp/minigpt_toy_muon \
  --optimizer muon \
  --model-config tiny_debug \
  --resume latest \
  --max-steps 3 \
  --max-minutes 0
```

## STOP file

```bash
touch <run-dir>/STOP
```

The script will save `checkpoints/latest.pt`, write summaries, and exit cleanly.

## EC2 TinyStories smoke shape

```bash
python gpu_benchmark/train_gpt.py \
  --run-id tinystories_60m_adamw_smoke_001 \
  --run-dir ~/pretrain_runs/tinystories_60m_adamw_smoke_001 \
  --dataset tinystories \
  --data-dir ~/pretrain_data/tinystories_gpt2_512 \
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
