#!/usr/bin/env python3
"""Generate publication-quality figures for the MiniGPT AdamW vs Muon benchmark.

Outputs both PDF (vector, for the final paper/report) and PNG (preview/sharing).
The script reads only completed local artifacts under gpu_benchmark/downloaded_runs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RUNS_125M = ROOT / "gpu_benchmark" / "downloaded_runs" / "compare_125m_finewebedu_3seed_4000s" / "runs.csv"
PAIRS_125M = ROOT / "gpu_benchmark" / "downloaded_runs" / "compare_125m_finewebedu_3seed_4000s" / "pairs.csv"
AGG_57M = ROOT / "gpu_benchmark" / "downloaded_runs" / "multi_seed_aggregate.csv"

# Publication defaults: clean, colorblind-safe, PDF-friendly.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "lines.linewidth": 1.9,
    "lines.markersize": 5,
})

ADAMW = "#0072B2"      # Okabe-Ito blue
MUON = "#D55E00"       # Okabe-Ito vermillion
GRAY = "#7A7A7A"
LIGHT_BLUE = "#E8F1FA"
LIGHT_ORANGE = "#FFF1E8"
LIGHT_GRAY = "#F4F4F2"
GREEN = "#009E73"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"fig_{stem}.pdf")
    fig.savefig(OUT / f"fig_{stem}.png", dpi=300)
    plt.close(fig)


def fig_125m_best_val_loss_by_seed(runs: list[dict[str, str]]) -> None:
    seeds = sorted({int(r["seed"]) for r in runs})
    adamw = [float(next(r for r in runs if int(r["seed"]) == s and r["optimizer"] == "AdamW")["best_val_loss"]) for s in seeds]
    muon = [float(next(r for r in runs if int(r["seed"]) == s and r["optimizer"] == "Hybrid Muon")["best_val_loss"]) for s in seeds]

    x = np.arange(len(seeds))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.7, 3.2))
    ax.bar(x - width / 2, adamw, width, label="AdamW", color=ADAMW, edgecolor="white", linewidth=0.7)
    ax.bar(x + width / 2, muon, width, label="Hybrid Muon", color=MUON, edgecolor="white", linewidth=0.7)
    for xi, a, m in zip(x, adamw, muon):
        ax.text(xi - width / 2, a + 0.006, f"{a:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
        ax.text(xi + width / 2, m + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Best validation loss ↓")
    ax.set_title("125M FineWeb-Edu: Muon reaches lower loss for every seed")
    ax.set_ylim(3.62, 3.86)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, "125m_best_val_loss_by_seed")


def fig_125m_delta_by_seed(pairs: list[dict[str, str]]) -> None:
    seeds = [int(p["seed"]) for p in pairs]
    deltas = [float(p["delta"]) for p in pairs]
    rels = [float(p["relative_improvement_pct"]) for p in pairs]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bars = ax.bar([str(s) for s in seeds], deltas, color=MUON, edgecolor="white", linewidth=0.7, width=0.55)
    ax.axhline(0, color="#222", linewidth=0.8)
    for bar, d, r in zip(bars, deltas, rels):
        ax.text(bar.get_x() + bar.get_width() / 2, d + 0.003, f"{d:.3f}\n({r:.2f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("AdamW loss − Muon loss ↑")
    ax.set_title("125M FineWeb-Edu: validation-loss advantage of Muon")
    ax.set_ylim(0, max(deltas) * 1.35)
    save(fig, "125m_muon_delta_by_seed")


def fig_validation_curves_125m(runs: list[dict[str, str]]) -> None:
    seeds = sorted({int(r["seed"]) for r in runs})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharey=True)
    handles, labels = None, None
    for ax, seed in zip(axes, seeds):
        for optimizer, color, marker in [("AdamW", ADAMW, "o"), ("Hybrid Muon", MUON, "s")]:
            row = next(r for r in runs if int(r["seed"]) == seed and r["optimizer"] == optimizer)
            metrics = read_jsonl(ROOT / row["run_dir"] / "metrics.jsonl")
            vals = [m for m in metrics if m.get("event") == "validation"]
            steps = [v["step"] for v in vals if v["step"] >= 200]
            losses = [v["val_loss"] for v in vals if v["step"] >= 200]
            ax.plot(steps, losses, label=optimizer, color=color, marker=marker, markevery=2)
        ax.set_title(f"seed {seed}", pad=5)
        ax.set_xlabel("Step")
        ax.set_xlim(200, 4000)
        handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Validation loss ↓")
    fig.suptitle("125M FineWeb-Edu validation curves", y=0.98, fontsize=11, fontweight="bold")
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    fig.subplots_adjust(top=0.76, bottom=0.18, wspace=0.18)
    save(fig, "125m_validation_curves_by_seed")


def fig_throughput_tradeoff(runs: list[dict[str, str]]) -> None:
    adamw_speed = mean(float(r["median_tokens_per_second"]) for r in runs if r["optimizer"] == "AdamW")
    muon_speed = mean(float(r["median_tokens_per_second"]) for r in runs if r["optimizer"] == "Hybrid Muon")
    adamw_loss = mean(float(r["best_val_loss"]) for r in runs if r["optimizer"] == "AdamW")
    muon_loss = mean(float(r["best_val_loss"]) for r in runs if r["optimizer"] == "Hybrid Muon")

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.05))
    axes[0].bar(["AdamW", "Hybrid\nMuon"], [adamw_loss, muon_loss], color=[ADAMW, MUON], width=0.58, edgecolor="white")
    axes[0].set_ylabel("Mean best val loss ↓")
    axes[0].set_title("Quality")
    axes[0].set_ylim(3.65, 3.86)
    for i, v in enumerate([adamw_loss, muon_loss]):
        axes[0].text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)

    axes[1].bar(["AdamW", "Hybrid\nMuon"], [adamw_speed / 1000, muon_speed / 1000], color=[ADAMW, MUON], width=0.58, edgecolor="white")
    axes[1].set_ylabel("Median throughput (k tokens/s) ↑")
    axes[1].set_title("Speed")
    axes[1].set_ylim(0, 115)
    for i, v in enumerate([adamw_speed / 1000, muon_speed / 1000]):
        axes[1].text(i, v + 2, f"{v:.1f}k", ha="center", fontsize=8)
    fig.suptitle("Tradeoff: Muon improves loss, AdamW is faster", y=0.98, fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.78, wspace=0.45)
    save(fig, "125m_quality_speed_tradeoff")


def fig_scaling_summary(pairs125: list[dict[str, str]], agg57: list[dict[str, str]]) -> None:
    labels = []
    rel = []
    err = []
    for row in agg57:
        if row["dataset"] == "FineWeb-Edu-500M" and row["steps"] == "4000":
            labels.append("57M\nFineWeb-Edu")
            rel.append(float(row["relative_mean_pct"]))
            err.append(0.0)
        if row["dataset"] == "TinyStories" and row["steps"] == "4000":
            labels.append("57M\nTinyStories")
            rel.append(float(row["relative_mean_pct"]))
            err.append(0.0)
    labels.append("125M\nFineWeb-Edu")
    rel.append(mean(float(p["relative_improvement_pct"]) for p in pairs125))
    err.append(float(np.std([float(p["relative_improvement_pct"]) for p in pairs125], ddof=1)))

    order = [1, 0, 2] if len(labels) == 3 else list(range(len(labels)))
    labels = [labels[i] for i in order]
    rel = [rel[i] for i in order]
    err = [err[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars = ax.bar(labels, rel, yerr=err, capsize=3, color=["#56B4E9", "#009E73", MUON], edgecolor="white", linewidth=0.7, width=0.58)
    for bar, v in zip(bars, rel):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean relative loss improvement (%) ↑")
    ax.set_title("Muon advantage persists from 57M to 125M")
    ax.set_ylim(0, max(rel) * 1.35)
    save(fig, "scaling_summary_57m_to_125m")


def rounded_box(ax, xy, w, h, text, fc, ec="#333", fontsize=9):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.035", linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#222")


def arrow(ax, start, end, color="#555"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=color, shrinkA=5, shrinkB=5))


def fig_experiment_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rounded_box(ax, (0.04, 0.58), 0.19, 0.22, "FineWeb-Edu\n500M-token cache", LIGHT_GRAY)
    rounded_box(ax, (0.29, 0.58), 0.19, 0.22, "MiniGPT-Dense\n125M params", LIGHT_BLUE)
    rounded_box(ax, (0.54, 0.68), 0.18, 0.17, "AdamW\nseed-matched", "#E7F0FA", ADAMW)
    rounded_box(ax, (0.54, 0.45), 0.18, 0.17, "Hybrid Muon\nseed-matched", LIGHT_ORANGE, MUON)
    rounded_box(ax, (0.78, 0.58), 0.18, 0.22, "Compare\nvalidation loss\n+ throughput", "#EEF8F3", GREEN)

    arrow(ax, (0.23, 0.69), (0.29, 0.69))
    arrow(ax, (0.48, 0.69), (0.54, 0.765), ADAMW)
    arrow(ax, (0.48, 0.66), (0.54, 0.535), MUON)
    arrow(ax, (0.72, 0.765), (0.78, 0.70), ADAMW)
    arrow(ax, (0.72, 0.535), (0.78, 0.64), MUON)

    ax.text(0.63, 0.31, "Seeds: 1337, 2024, 2025\nSame data, model, batch size, and token budget", ha="center", va="center", fontsize=8.5, color="#333")
    ax.text(0.5, 0.94, "Controlled optimizer benchmark protocol", ha="center", va="center", fontsize=12, fontweight="bold")
    save(fig, "experiment_pipeline")


def fig_optimizer_split() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.93, "Hybrid Muon optimizer split", ha="center", va="center", fontsize=12, fontweight="bold")
    rounded_box(ax, (0.05, 0.40), 0.20, 0.28, "Model\nparameters", LIGHT_GRAY)
    rounded_box(ax, (0.36, 0.60), 0.24, 0.22, "2D hidden matrices\nattention + MLP", LIGHT_ORANGE, MUON)
    rounded_box(ax, (0.36, 0.24), 0.24, 0.22, "Embeddings, norms,\nbiases, vectors", "#E7F0FA", ADAMW)
    rounded_box(ax, (0.72, 0.60), 0.20, 0.22, "Muon update\nmatrix-aware", LIGHT_ORANGE, MUON)
    rounded_box(ax, (0.72, 0.24), 0.20, 0.22, "AdamW update\nelement-wise", "#E7F0FA", ADAMW)
    arrow(ax, (0.25, 0.56), (0.36, 0.71), MUON)
    arrow(ax, (0.25, 0.48), (0.36, 0.35), ADAMW)
    arrow(ax, (0.60, 0.71), (0.72, 0.71), MUON)
    arrow(ax, (0.60, 0.35), (0.72, 0.35), ADAMW)
    ax.text(0.50, 0.08, "Key idea: use Muon only where matrix geometry matters; keep AdamW for non-matrix parameters.", ha="center", fontsize=8.5, color="#333")
    save(fig, "hybrid_muon_optimizer_split")


def write_manifest() -> None:
    files = sorted(OUT.glob("fig_*.pdf"))
    lines = ["# Benchmark figures", "", "Generated by `figures/gen_fig_muon_benchmark.py`.", "", "## Figure set", ""]
    desc = {
        "fig_125m_best_val_loss_by_seed.pdf": "Grouped bar chart showing AdamW vs Hybrid Muon best validation loss for each 125M seed.",
        "fig_125m_muon_delta_by_seed.pdf": "Per-seed validation-loss delta; positive means Muon wins.",
        "fig_125m_validation_curves_by_seed.pdf": "Validation curves over training steps for all three 125M seeds.",
        "fig_125m_quality_speed_tradeoff.pdf": "Quality-speed tradeoff: Muon lower loss, AdamW higher throughput.",
        "fig_scaling_summary_57m_to_125m.pdf": "Summary showing Muon improvement across 57M and 125M experiments.",
        "fig_experiment_pipeline.pdf": "Visual protocol diagram for the controlled benchmark.",
        "fig_hybrid_muon_optimizer_split.pdf": "Conceptual diagram explaining which parameters use Muon vs AdamW.",
    }
    for p in files:
        lines.append(f"- `{p.name}` — {desc.get(p.name, '')}")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = read_csv(RUNS_125M)
    pairs = read_csv(PAIRS_125M)
    agg57 = read_csv(AGG_57M)
    fig_125m_best_val_loss_by_seed(runs)
    fig_125m_delta_by_seed(pairs)
    fig_validation_curves_125m(runs)
    fig_throughput_tradeoff(runs)
    fig_scaling_summary(pairs, agg57)
    fig_experiment_pipeline()
    fig_optimizer_split()
    write_manifest()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
