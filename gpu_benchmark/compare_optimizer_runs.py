"""Compare two optimizer runs from ``train_gpt.py``.

This script is intentionally small and boring:

1. Read each run directory.
2. Parse ``config.json``, ``run_summary.json``, and ``metrics.jsonl``.
3. Write a Markdown report, a CSV table, and simple loss plots.

Example:

python gpu_benchmark/compare_optimizer_runs.py \
  --adamw-run gpu_benchmark/downloaded_runs/finewebedu_125m_adamw_sample10bt_seed1337_4000s_001 \
  --muon-run gpu_benchmark/downloaded_runs/finewebedu_125m_muon_sample10bt_seed1337_4000s_001 \
  --out-dir gpu_benchmark/downloaded_runs/compare_125m_seed1337
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def optimizer_label(run_dir: Path, config: dict[str, Any]) -> str:
    optimizer_config = config.get("optimizer")
    if isinstance(optimizer_config, dict):
        opt = optimizer_config.get("name")
    else:
        opt = optimizer_config
    opt = opt or run_dir.name.lower()
    opt = str(opt).lower()
    if "muon" in opt:
        return "Hybrid Muon"
    if "adamw" in opt:
        return "AdamW"
    return str(opt)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "run_summary.json")
    metrics = read_metrics(run_dir / "metrics.jsonl")
    model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
    data_config = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    train_config = config.get("training") if isinstance(config.get("training"), dict) else {}

    train_rows = [r for r in metrics if r.get("event") == "train_step"]
    val_rows = [r for r in metrics if r.get("event") == "validation"]

    # Median over the second half avoids early warmup/validation-save stalls.
    speeds = [float(r["tokens_per_second"]) for r in train_rows if "tokens_per_second" in r]
    warm_speeds = speeds[len(speeds) // 2 :] if speeds else []

    max_gpu_bytes = 0
    for r in train_rows:
        max_gpu_bytes = max(max_gpu_bytes, int(r.get("gpu_max_memory_allocated_bytes") or 0))

    return {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "optimizer": optimizer_label(run_dir, config),
        "status": summary.get("status"),
        "model_config": model_config.get("name") or config.get("model_config"),
        "parameter_count": model_config.get("parameter_count"),
        "dataset": data_config.get("dataset") or config.get("dataset"),
        "fineweb_config": data_config.get("config") or config.get("fineweb_config"),
        "train_tokens": data_config.get("train_tokens"),
        "validation_tokens": data_config.get("validation_tokens"),
        "seed": train_config.get("seed") or config.get("seed"),
        "max_steps": train_config.get("max_steps") or config.get("max_steps"),
        "micro_batch_size": train_config.get("micro_batch_size") or config.get("micro_batch_size"),
        "grad_accum_steps": train_config.get("grad_accum_steps") or config.get("grad_accum_steps"),
        "effective_batch_tokens": train_config.get("effective_batch_tokens"),
        "block_size": model_config.get("block_size") or data_config.get("block_size") or config.get("block_size"),
        "tokens_seen": summary.get("tokens_seen"),
        "best_val_loss": summary.get("best_val_loss"),
        "last_val_loss": summary.get("last_val_loss"),
        "last_step": summary.get("last_step"),
        "median_tokens_per_second": median(warm_speeds) if warm_speeds else None,
        "max_gpu_gb": max_gpu_bytes / (1024**3) if max_gpu_bytes else None,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fields = [
        "run_id",
        "optimizer",
        "status",
        "model_config",
        "parameter_count",
        "dataset",
        "fineweb_config",
        "train_tokens",
        "validation_tokens",
        "seed",
        "last_step",
        "tokens_seen",
        "best_val_loss",
        "last_val_loss",
        "median_tokens_per_second",
        "max_gpu_gb",
        "micro_batch_size",
        "grad_accum_steps",
        "effective_batch_tokens",
        "block_size",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    for row in rows:
        xs = [r["step"] for r in row["train_rows"] if "train_loss" in r]
        ys = [r["train_loss"] for r in row["train_rows"] if "train_loss" in r]
        plt.plot(xs, ys, label=row["optimizer"], alpha=0.9)
    plt.title("Training loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "train_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    for row in rows:
        xs = [r["step"] for r in row["val_rows"] if "val_loss" in r]
        ys = [r["val_loss"] for r in row["val_rows"] if "val_loss" in r]
        plt.plot(xs, ys, marker="o", label=row["optimizer"], alpha=0.9)
    plt.title("Validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss.png", dpi=180)
    plt.close()


def write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    adamw = next((r for r in rows if r["optimizer"] == "AdamW"), None)
    muon = next((r for r in rows if r["optimizer"] == "Hybrid Muon"), None)

    delta_text = "n/a"
    rel_text = "n/a"
    if adamw and muon and adamw["best_val_loss"] is not None and muon["best_val_loss"] is not None:
        delta = float(adamw["best_val_loss"]) - float(muon["best_val_loss"])
        rel = delta / float(adamw["best_val_loss"]) * 100.0
        delta_text = fmt_float(delta)
        rel_text = f"{rel:.2f}%"

    first = rows[0]
    lines = [
        "# 125M MiniGPT AdamW vs Hybrid Muon comparison",
        "",
        "Positive delta means `AdamW best validation loss - Muon best validation loss`.",
        "Positive is good for Muon; negative is bad for Muon.",
        "",
        "## Setup",
        "",
        "```text",
        f"model_config: {first.get('model_config')}",
        f"parameter_count: {first.get('parameter_count')}",
        f"dataset: {first.get('dataset')}",
        f"fineweb_config: {first.get('fineweb_config')}",
        f"train_tokens_in_cache: {first.get('train_tokens')}",
        f"validation_tokens_in_cache: {first.get('validation_tokens')}",
        f"seed: {first.get('seed')}",
        f"max_steps: {first.get('max_steps')}",
        f"micro_batch_size: {first.get('micro_batch_size')}",
        f"grad_accum_steps: {first.get('grad_accum_steps')}",
        f"effective_batch_tokens: {first.get('effective_batch_tokens')}",
        f"block_size: {first.get('block_size')}",
        "```",
        "",
        "## Run table",
        "",
        "| Optimizer | Status | Step | Tokens | Best val loss | Last val loss | Median tok/s | Max GPU GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["optimizer"]),
                    str(row["status"]),
                    fmt_int(row["last_step"]),
                    fmt_int(row["tokens_seen"]),
                    fmt_float(row["best_val_loss"]),
                    fmt_float(row["last_val_loss"]),
                    fmt_int(row["median_tokens_per_second"]),
                    fmt_float(row["max_gpu_gb"], 2),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Paired result",
        "",
        f"- Best validation-loss delta: `{delta_text}`",
        f"- Relative improvement: `{rel_text}`",
        "",
        "## Plots",
        "",
        "- `train_loss.png`",
        "- `val_loss.png`",
        "",
        "## Interpretation guardrail",
        "",
        "This is a one-seed 125M check. It tells us whether the 57M result still appears at GPT-2-small scale, but it is not yet a statistically stable claim. For a stronger claim, repeat this exact pair for seeds `2024` and `2025`.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adamw-run", type=Path, required=True)
    parser.add_argument("--muon-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(args.adamw_run), summarize_run(args.muon_run)]
    write_csv(rows, args.out_dir / "summary.csv")
    write_plots(rows, args.out_dir)
    write_markdown(rows, args.out_dir / "summary.md")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
