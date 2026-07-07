#!/usr/bin/env python3
"""Batch-size sweep runner for MiniGPT pretraining.

Why this exists
---------------
When we say "increase batch size", there are two different knobs:

1. micro_batch_size
   The number of sequences that fit in one GPU forward/backward pass.
   This mostly affects GPU efficiency and VRAM use.

2. effective_batch_tokens
   The total number of tokens used before one optimizer update.
   This affects training dynamics and model quality.

For a fair efficiency sweep, this script changes micro_batch_size but keeps
`effective_batch_tokens` approximately constant by adjusting grad_accum_steps.
That means we are mostly testing "how full can we make the GPU?" instead of
also changing the optimizer's learning behavior.

Example on EC2:

    python -u gpu_benchmark/batch_sweep.py \
      --data-dir ~/pretrain_data/tinystories_gpt2_512_full \
      --run-root ~/pretrain_runs/batch_sweep_60m_adamw_fullts_001 \
      --optimizer adamw \
      --micro-batches 4,8,16,32,64,128 \
      --effective-batch-tokens 65536 \
      --steps 30
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_run(run_dir: Path, micro_batch: int, grad_accum: int, returncode: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    train = [r for r in metrics if r.get("event") == "train_step"]
    val = [r for r in metrics if r.get("event") == "validation"]
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    # First steps include tokenizer/model warmup and checkpoint overhead. For speed,
    # use later train log points when available.
    speed_points = [r.get("tokens_per_second") for r in train[2:] if r.get("tokens_per_second")]
    if not speed_points:
        speed_points = [r.get("tokens_per_second") for r in train if r.get("tokens_per_second")]

    max_mem_gb = 0.0
    if train:
        max_mem_gb = max(float(r.get("gpu_max_memory_allocated_bytes", 0)) for r in train) / (1024**3)

    return {
        "micro_batch_size": micro_batch,
        "grad_accum_steps": grad_accum,
        "effective_batch_tokens": micro_batch * grad_accum * 512,  # our locked block size for this sweep
        "returncode": returncode,
        "status": summary.get("status", "missing_summary"),
        "last_step": summary.get("last_step"),
        "tokens_seen": summary.get("tokens_seen"),
        "best_val_loss": summary.get("best_val_loss"),
        "last_val_loss": summary.get("last_val_loss"),
        "last_train_loss": train[-1].get("train_loss") if train else None,
        "median_tokens_per_second": median(speed_points) if speed_points else None,
        "max_gpu_allocated_gb": max_mem_gb,
        "validation_count": len(val),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a clean micro-batch-size sweep.")
    parser.add_argument("--data-dir", required=True, help="Token cache directory containing train.bin/validation.bin")
    parser.add_argument("--dataset", choices=["tinystories", "finewebedu"], default="tinystories", help="Dataset label passed to train_gpt.py. Use finewebedu for FineWeb-Edu caches.")
    parser.add_argument("--fineweb-config", default="sample-10BT", help="FineWeb-Edu config label when --dataset finewebedu.")
    parser.add_argument("--run-root", required=True, help="Directory that will contain one run per micro-batch size")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--model-config", default="minigpt_dense_60m_v1")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seed", type=int, default=1337, help="Seed passed to train_gpt.py for reproducible sweeps")
    parser.add_argument("--micro-batches", default="4,8,16,32,64,128")
    parser.add_argument("--effective-batch-tokens", type=int, default=65_536)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--validation-tokens", type=int, default=16_384)
    parser.add_argument("--compile", action="store_true", help="Pass --compile to train_gpt.py")
    args = parser.parse_args()

    script = Path(__file__).with_name("train_gpt.py")
    run_root = Path(args.run_root).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)

    micro_batches = [int(x) for x in args.micro_batches.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []

    for micro_batch in micro_batches:
        tokens_per_micro = micro_batch * 512
        if args.effective_batch_tokens % tokens_per_micro != 0:
            print(
                f"Skipping micro_batch={micro_batch}: effective_batch_tokens "
                f"{args.effective_batch_tokens} is not divisible by {tokens_per_micro}",
                flush=True,
            )
            continue
        grad_accum = args.effective_batch_tokens // tokens_per_micro
        run_dir = run_root / f"{args.optimizer}_mb{micro_batch}_ga{grad_accum}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--run-id",
            run_dir.name,
            "--run-dir",
            str(run_dir),
            "--dataset",
            args.dataset,
            "--fineweb-config",
            args.fineweb_config,
            "--data-dir",
            str(Path(args.data_dir).expanduser()),
            "--model-config",
            args.model_config,
            "--optimizer",
            args.optimizer,
            "--precision",
            args.precision,
            "--seed",
            str(args.seed),
            "--micro-batch-size",
            str(micro_batch),
            "--grad-accum-steps",
            str(grad_accum),
            "--max-steps",
            str(args.steps),
            "--max-minutes",
            "0",
            "--log-every-steps",
            "5",
            "--eval-every-steps",
            str(args.steps),
            "--save-every-steps",
            str(args.steps),
            "--validation-tokens",
            str(args.validation_tokens),
            "--sample-new-tokens",
            "0",
        ]
        if args.compile:
            cmd.append("--compile")

        print("\n=== running", run_dir.name, "===", flush=True)
        print(" ".join(cmd), flush=True)
        completed = subprocess.run(cmd, text=True)
        row = summarize_run(run_dir, micro_batch, grad_accum, completed.returncode)
        rows.append(row)
        print("summary:", json.dumps(row, indent=2), flush=True)

        # If we hit OOM at a large batch, keep going would usually fail for even
        # larger batches. Stop early and keep the successful results.
        if completed.returncode != 0:
            print(f"Stopping sweep after failure at micro_batch={micro_batch}.", flush=True)
            break

    summary_json = run_root / "batch_sweep_summary.json"
    summary_csv = run_root / "batch_sweep_summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\nWrote", summary_json)
    print("Wrote", summary_csv)


if __name__ == "__main__":
    main()
