#!/usr/bin/env python3
"""Generate visually appealing 3D-style schematic figures for the MiniGPT/Muon report.

These are code-generated figures inspired by classic neural-network architecture
slides: stacked feature maps, repeated blocks, arrows, and callout labels.

Outputs:
- fig_3d_minigpt_architecture.png/pdf/svg
- fig_3d_transformer_block_exploded.png/pdf/svg
- fig_3d_muon_optimizer_split.png/pdf/svg
- fig_3d_schematics_contact_sheet.png
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 220,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COL = {
    "blue": "#2F80ED",
    "blue_dark": "#174A7C",
    "blue_light": "#DCEBFF",
    "orange": "#E67E22",
    "orange_dark": "#A64E00",
    "orange_light": "#FFE8CC",
    "green": "#27AE60",
    "green_light": "#DDF5E7",
    "purple": "#9B59B6",
    "purple_light": "#EBDCF4",
    "gray": "#6B7280",
    "gray_light": "#F3F4F6",
    "ink": "#1F2937",
}


def save(fig, stem: str):
    fig.savefig(OUT / f"fig_{stem}.png", dpi=300)
    fig.savefig(OUT / f"fig_{stem}.pdf")
    fig.savefig(OUT / f"fig_{stem}.svg")
    plt.close(fig)


def add_panel_title(ax, title, subtitle=None):
    ax.text(0.5, 0.965, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=18, fontweight="bold", color=COL["ink"])
    if subtitle:
        ax.text(0.5, 0.925, subtitle, transform=ax.transAxes, ha="center", va="top",
                fontsize=10.5, color=COL["gray"])


def arrow(ax, xy1, xy2, color="#4B5563", lw=2, style="-|>", rad=0.0):
    a = FancyArrowPatch(
        xy1, xy2,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(a)
    return a


def box(ax, x, y, w, h, text, fc, ec=None, fontsize=10, weight="bold", radius=0.035):
    ec = ec or fc
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.5,
    )
    patch.set_path_effects([pe.SimplePatchShadow(offset=(1.6, -1.6), alpha=0.14), pe.Normal()])
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=COL["ink"])
    return patch


def stacked_sheets(ax, x, y, w, h, n=5, dx=0.018, dy=0.018, fc="#DCEBFF", ec="#2F80ED", label="", side_label=None, label_color=None):
    """Draw a pseudo-3D stack of 2D feature maps."""
    for i in range(n - 1, -1, -1):
        xi = x + i * dx
        yi = y + i * dy
        rect = Rectangle((xi, yi), w, h, facecolor=fc, edgecolor=ec, linewidth=1.2)
        rect.set_path_effects([pe.SimplePatchShadow(offset=(1.1, -1.1), alpha=0.10), pe.Normal()])
        ax.add_patch(rect)
    if label:
        ax.text(x + w / 2 + dx * (n - 1) / 2, y - 0.038, label, ha="center", va="top",
                fontsize=9.2, fontweight="bold", color=label_color or ec)
    if side_label:
        ax.text(x + w + dx * n + 0.010, y + h / 2, side_label, ha="left", va="center",
                fontsize=8.5, color=COL["gray"])


def vertical_slab(ax, x, y, w, h, text, fc, ec, fontsize=10):
    # front face
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.5)
    rect.set_path_effects([pe.SimplePatchShadow(offset=(1.4, -1.4), alpha=0.15), pe.Normal()])
    ax.add_patch(rect)
    # pseudo-side face
    poly = plt.Polygon([[x+w, y], [x+w+0.025, y+0.025], [x+w+0.025, y+h+0.025], [x+w, y+h]],
                       facecolor=ec, edgecolor=ec, alpha=0.32)
    ax.add_patch(poly)
    ax.text(x + w/2, y+h/2, text, ha="center", va="center", rotation=90,
            fontsize=fontsize, fontweight="bold", color="white")


def fig_3d_minigpt_architecture():
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_title(
        ax,
        "MiniGPT-Dense-125M-v1 architecture",
        "Decoder-only GPT: token stream → embeddings → 12 Transformer blocks → tied vocabulary head",
    )

    # Input token column.
    x0, y0 = 0.055, 0.34
    for i in range(10):
        c = COL["blue_light"] if i not in (4, 5) else "#7DB7FF"
        ax.add_patch(Rectangle((x0, y0 + i * 0.035), 0.05, 0.03, facecolor=c, edgecolor="#8EA7C2", linewidth=0.8))
        ax.text(x0 - 0.016, y0 + i * 0.035 + 0.015, f"t{i}", ha="right", va="center", fontsize=7.5, color=COL["gray"])
    ax.text(x0 + 0.025, y0 + 0.39, "Input\ntokens", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.annotate("current position", xy=(x0 + 0.052, y0 + 0.035 * 5 + 0.015), xytext=(0.155, 0.54),
                arrowprops=dict(arrowstyle="->", color=COL["blue"], lw=1.8), fontsize=8.5, color=COL["blue_dark"])

    # Main architecture path.
    stacked_sheets(ax, 0.18, 0.34, 0.065, 0.30, n=5, fc=COL["blue_light"], ec=COL["blue"],
                   label="Embeddings", side_label="B × T × 768")
    arrow(ax, (0.115, 0.49), (0.18, 0.49), COL["gray"])

    vertical_slab(ax, 0.31, 0.31, 0.045, 0.36, "LayerNorm", "#9CC7FF", COL["blue"])
    arrow(ax, (0.26, 0.49), (0.31, 0.49), COL["gray"])

    # Repeated transformer blocks as orange stacks.
    xs = [0.395, 0.455, 0.515, 0.575, 0.635]
    labels = ["Block 1", "Block 2", "...", "Block 11", "Block 12"]
    for x, lab in zip(xs, labels):
        stacked_sheets(ax, x, 0.31, 0.038, 0.36, n=4 if lab != "..." else 2,
                       dx=0.008, dy=0.010, fc=COL["orange_light"] if lab != "..." else "#E5E7EB",
                       ec=COL["orange"] if lab != "..." else "#9CA3AF", label=lab,
                       label_color=COL["orange_dark"] if lab != "..." else COL["gray"])
    arrow(ax, (0.358, 0.49), (0.395, 0.49), COL["gray"])
    arrow(ax, (0.44, 0.49), (0.455, 0.49), COL["gray"])
    arrow(ax, (0.50, 0.49), (0.515, 0.49), COL["gray"])
    arrow(ax, (0.56, 0.49), (0.575, 0.49), COL["gray"])
    arrow(ax, (0.62, 0.49), (0.635, 0.49), COL["gray"])

    vertical_slab(ax, 0.745, 0.33, 0.035, 0.32, "Final Norm", "#9CC7FF", COL["blue"], fontsize=9)
    arrow(ax, (0.70, 0.49), (0.745, 0.49), COL["gray"])

    stacked_sheets(ax, 0.83, 0.29, 0.050, 0.40, n=4, dx=0.010, dy=0.012,
                   fc=COL["green_light"], ec=COL["green"], label="Tied LM head", side_label="logits over 50,257 tokens",
                   label_color=COL["green"])
    arrow(ax, (0.785, 0.49), (0.83, 0.49), COL["gray"])

    # Callouts.
    box(ax, 0.31, 0.12, 0.18, 0.09, "12 layers\n768 hidden", COL["gray_light"], "#D1D5DB", fontsize=9)
    box(ax, 0.51, 0.12, 0.18, 0.09, "12 heads\n64 dim/head", COL["gray_light"], "#D1D5DB", fontsize=9)
    box(ax, 0.71, 0.12, 0.18, 0.09, "MLP width\n3072", COL["gray_light"], "#D1D5DB", fontsize=9)
    ax.text(0.5, 0.055, "Parameters: 123,963,648  •  context length: 512  •  GPT-2 tokenizer vocabulary: 50,257",
            ha="center", va="center", fontsize=10, color=COL["ink"])
    save(fig, "3d_minigpt_architecture")


def fig_3d_transformer_block_exploded():
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_title(
        ax,
        "One Transformer block, exploded",
        "The repeated unit inside MiniGPT: attention path + MLP path wrapped by residual connections",
    )

    # Residual stream baseline.
    arrow(ax, (0.06, 0.50), (0.94, 0.50), COL["gray"], lw=2.1)
    ax.text(0.07, 0.535, "residual stream\nB × T × 768", fontsize=9, color=COL["gray"], ha="left")

    box(ax, 0.12, 0.43, 0.10, 0.12, "LN₁", COL["blue_light"], COL["blue"], fontsize=12)
    arrow(ax, (0.22, 0.50), (0.28, 0.50), COL["gray"])

    # QKV fanout.
    box(ax, 0.28, 0.66, 0.10, 0.10, "Q proj\n768×768", COL["orange_light"], COL["orange"], fontsize=8.5)
    box(ax, 0.28, 0.48, 0.10, 0.10, "K proj\n768×768", COL["orange_light"], COL["orange"], fontsize=8.5)
    box(ax, 0.28, 0.30, 0.10, 0.10, "V proj\n768×768", COL["orange_light"], COL["orange"], fontsize=8.5)
    arrow(ax, (0.22, 0.50), (0.28, 0.71), COL["orange"], rad=0.08)
    arrow(ax, (0.22, 0.50), (0.28, 0.53), COL["orange"])
    arrow(ax, (0.22, 0.50), (0.28, 0.35), COL["orange"], rad=-0.08)

    stacked_sheets(ax, 0.45, 0.34, 0.075, 0.32, n=8, dx=0.007, dy=0.010,
                   fc="#FFE0B8", ec=COL["orange"], label="12 attention heads", side_label="causal attention")
    arrow(ax, (0.38, 0.71), (0.45, 0.60), COL["orange"])
    arrow(ax, (0.38, 0.53), (0.45, 0.50), COL["orange"])
    arrow(ax, (0.38, 0.35), (0.45, 0.40), COL["orange"])

    box(ax, 0.60, 0.43, 0.11, 0.12, "O proj\n768×768", COL["orange_light"], COL["orange"], fontsize=8.5)
    arrow(ax, (0.535, 0.50), (0.60, 0.50), COL["orange"])
    arrow(ax, (0.71, 0.50), (0.76, 0.50), COL["gray"])
    box(ax, 0.76, 0.43, 0.08, 0.12, "+", "#FFFFFF", "#6B7280", fontsize=18)
    ax.text(0.80, 0.37, "residual add", ha="center", fontsize=8.5, color=COL["gray"])

    # MLP lower path.
    box(ax, 0.25, 0.12, 0.09, 0.10, "LN₂", COL["blue_light"], COL["blue"], fontsize=11)
    box(ax, 0.40, 0.12, 0.13, 0.10, "MLP up\n768→3072", COL["orange_light"], COL["orange"], fontsize=8.5)
    box(ax, 0.57, 0.12, 0.09, 0.10, "GELU", "#FFF7CC", "#D4A000", fontsize=10)
    box(ax, 0.70, 0.12, 0.13, 0.10, "MLP down\n3072→768", COL["orange_light"], COL["orange"], fontsize=8.5)
    arrow(ax, (0.80, 0.43), (0.30, 0.22), COL["blue"], rad=-0.18)
    arrow(ax, (0.34, 0.17), (0.40, 0.17), COL["gray"])
    arrow(ax, (0.53, 0.17), (0.57, 0.17), COL["gray"])
    arrow(ax, (0.66, 0.17), (0.70, 0.17), COL["gray"])
    arrow(ax, (0.83, 0.17), (0.89, 0.48), COL["blue"], rad=-0.12)
    box(ax, 0.88, 0.43, 0.08, 0.12, "+", "#FFFFFF", "#6B7280", fontsize=18)
    ax.text(0.92, 0.37, "residual add", ha="center", fontsize=8.5, color=COL["gray"])

    ax.text(0.50, 0.06, "Orange matrices are Muon candidates in our Hybrid Muon run; blue/norm/vector parameters stay on AdamW.",
            ha="center", fontsize=10, color=COL["ink"])
    save(fig, "3d_transformer_block_exploded")


def fig_3d_muon_optimizer_split():
    fig, ax = plt.subplots(figsize=(13.2, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_title(
        ax,
        "Hybrid Muon optimizer split",
        "Same model parameters, two update rules: matrix-aware Muon for hidden 2D weights, AdamW for fallback parameters",
    )

    # Left: parameter stack.
    stacked_sheets(ax, 0.08, 0.36, 0.08, 0.30, n=7, fc=COL["gray_light"], ec="#9CA3AF", label="Model parameters")

    # Split groups.
    box(ax, 0.31, 0.60, 0.20, 0.13, "Hidden 2D matrices\nattention + MLP", COL["orange_light"], COL["orange"], fontsize=10)
    box(ax, 0.31, 0.29, 0.20, 0.13, "Fallback params\nembeddings, norms, vectors", COL["blue_light"], COL["blue"], fontsize=10)
    arrow(ax, (0.18, 0.54), (0.31, 0.665), COL["orange"], rad=0.08)
    arrow(ax, (0.18, 0.45), (0.31, 0.355), COL["blue"], rad=-0.08)

    # Updates.
    stacked_sheets(ax, 0.62, 0.58, 0.08, 0.18, n=5, fc="#FFD9B0", ec=COL["orange"], label="Muon update")
    stacked_sheets(ax, 0.62, 0.27, 0.08, 0.18, n=5, fc="#DCEBFF", ec=COL["blue"], label="AdamW update")
    arrow(ax, (0.51, 0.665), (0.62, 0.68), COL["orange"])
    arrow(ax, (0.51, 0.355), (0.62, 0.37), COL["blue"])

    # Final model.
    box(ax, 0.82, 0.43, 0.11, 0.15, "Updated\nmodel", COL["green_light"], COL["green"], fontsize=11)
    arrow(ax, (0.72, 0.67), (0.82, 0.54), COL["orange"], rad=-0.06)
    arrow(ax, (0.72, 0.36), (0.82, 0.48), COL["blue"], rad=0.06)

    # Practical note row.
    ax.text(0.40, 0.16, "Muon is not replacing AdamW everywhere. The point of the paper implementation is the split.",
            ha="center", fontsize=10, color=COL["ink"])
    ax.text(0.40, 0.11, "This makes the benchmark fair: same model, same data, same seed, same token budget — only optimizer logic changes.",
            ha="center", fontsize=9, color=COL["gray"])
    save(fig, "3d_muon_optimizer_split")


def contact_sheet():
    items = [
        ("fig_3d_minigpt_architecture.png", "Full MiniGPT architecture"),
        ("fig_3d_transformer_block_exploded.png", "One Transformer block exploded"),
        ("fig_3d_muon_optimizer_split.png", "Hybrid Muon optimizer split"),
    ]
    thumbs = []
    for fname, label in items:
        img = Image.open(OUT / fname).convert("RGB")
        img.thumbnail((850, 410))
        canvas = Image.new("RGB", (900, 480), "white")
        canvas.paste(img, ((900 - img.width) // 2, 20))
        d = ImageDraw.Draw(canvas)
        d.text((25, 435), label, fill=(20, 20, 20))
        d.text((25, 455), str(OUT / fname), fill=(90, 90, 90))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (900, 480 * len(thumbs)), (245, 245, 245))
    for i, t in enumerate(thumbs):
        sheet.paste(t, (0, i * 480))
    sheet.save(OUT / "fig_3d_schematics_contact_sheet.png")


def main():
    fig_3d_minigpt_architecture()
    fig_3d_transformer_block_exploded()
    fig_3d_muon_optimizer_split()
    contact_sheet()
    print("Generated 3D schematic figures:")
    for p in sorted(OUT.glob("fig_3d_*.png")):
        print("-", p)


if __name__ == "__main__":
    main()
