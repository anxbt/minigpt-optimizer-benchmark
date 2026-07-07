#!/usr/bin/env python3
"""Generate architecture visualization artifacts for MiniGPT-Dense-125M-v1.

Artifacts generated:
- TikZ standalone architecture diagram.
- PGFPlots standalone result plot.
- PlotNeuralNet 3D TikZ source/PDF.
- SVG architecture diagram.
- Netron-readable ONNX schematic.
- TensorBoard graph event from the real PyTorch model.
- VisualKeras layered view using KERAS_BACKEND=torch.
- Matplotlib 3D architecture diagram.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TIKZ = FIG / "tikz"
PGF = FIG / "pgfplots"
SVG = FIG / "svg"
NETRON = FIG / "netron"
TB = FIG / "tensorboard"
VK = FIG / "visualkeras"
PNN = FIG / "plotneuralnet"
for d in [TIKZ, PGF, SVG, NETRON, TB, VK, PNN]:
    d.mkdir(parents=True, exist_ok=True)

MODEL = {
    "name": "MiniGPT-Dense-125M-v1",
    "params": 123_963_648,
    "layers": 12,
    "hidden": 768,
    "heads": 12,
    "head_dim": 64,
    "context": 512,
    "vocab": 50_257,
    "mlp": 3072,
}

RUNS = FIG.parent / "gpu_benchmark" / "downloaded_runs" / "compare_125m_finewebedu_3seed_4000s" / "runs.csv"
PAIRS = FIG.parent / "gpu_benchmark" / "downloaded_runs" / "compare_125m_finewebedu_3seed_4000s" / "pairs.csv"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def compile_latex(tex: Path) -> None:
    try:
        out = run(["pdflatex", "-interaction=nonstopmode", tex.name], cwd=tex.parent, check=False)
        (tex.with_suffix(".compile.log")).write_text(out.stdout, encoding="utf-8")
        if out.returncode != 0:
            print(f"WARN: pdflatex failed for {tex}; see {tex.with_suffix('.compile.log')}")
    except FileNotFoundError:
        print("WARN: pdflatex not found; leaving .tex source only")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_tikz_architecture() -> None:
    tex = TIKZ / "fig_minigpt_architecture_tikz.tex"
    tex.write_text(rf"""
\documentclass[tikz,border=8pt]{{standalone}}
\usetikzlibrary{{arrows.meta,positioning,fit,backgrounds,calc}}
\definecolor{{adamw}}{{HTML}}{{0072B2}}
\definecolor{{muon}}{{HTML}}{{D55E00}}
\definecolor{{softblue}}{{HTML}}{{E8F1FA}}
\definecolor{{softorange}}{{HTML}}{{FFF1E8}}
\definecolor{{softgray}}{{HTML}}{{F4F4F2}}
\begin{{document}}
\begin{{tikzpicture}}[
  box/.style={{rounded corners=4pt, draw=black!55, very thick, align=center, minimum height=0.85cm, font=\small}},
  arrow/.style={{-{{Latex[length=2mm]}}, thick, draw=black!65}},
  muonbox/.style={{box, fill=softorange, draw=muon!80}},
  adamwbox/.style={{box, fill=softblue, draw=adamw!80}},
  neutral/.style={{box, fill=softgray}}
]
\node[neutral, minimum width=2.4cm] (tok) {{Token IDs\\$B \times T$}};
\node[adamwbox, right=0.75cm of tok, minimum width=2.8cm] (emb) {{Token + position\\embeddings\\$V={MODEL['vocab']:,}$, $T={MODEL['context']}$}};
\node[muonbox, right=0.75cm of emb, minimum width=3.5cm] (blocks) {{$\times {MODEL['layers']}$ Transformer blocks\\hidden $d={MODEL['hidden']}$\\heads $h={MODEL['heads']}$}};
\node[adamwbox, right=0.75cm of blocks, minimum width=2.5cm] (ln) {{Final LayerNorm}};
\node[adamwbox, right=0.75cm of ln, minimum width=2.6cm] (head) {{Tied LM head\\logits $B \times T \times V$}};

\draw[arrow] (tok) -- (emb);
\draw[arrow] (emb) -- (blocks);
\draw[arrow] (blocks) -- (ln);
\draw[arrow] (ln) -- (head);

\node[muonbox, below=1.05cm of blocks, minimum width=3.1cm] (attn) {{Attention matrices\\Q,K,V,O: $768\times768$}};
\node[muonbox, right=0.45cm of attn, minimum width=3.0cm] (mlp) {{MLP matrices\\$768\to3072\to768$}};
\node[adamwbox, left=0.45cm of attn, minimum width=3.0cm] (fallback) {{AdamW fallback\\embeddings, norms, vectors}};
\draw[arrow, draw=muon] (blocks.south) -- (attn.north);
\draw[arrow, draw=muon] (blocks.south east) -- (mlp.north);
\draw[arrow, draw=adamw] (emb.south) -- (fallback.north);

\node[font=\bfseries\large, above=0.55cm of blocks] {{MiniGPT-Dense-125M-v1 architecture and optimizer split}};
\node[align=center, font=\small, below=0.35cm of attn] {{Hybrid Muon applies matrix-aware updates to hidden 2D matrices; AdamW handles embeddings, norms, biases/vectors.}};
\end{{tikzpicture}}
\end{{document}}
""", encoding="utf-8")
    compile_latex(tex)


def write_pgfplots_result() -> None:
    pairs = read_csv(PAIRS)
    coords_adamw, coords_muon = [], []
    for p in pairs:
        seed = p["seed"]
        coords_adamw.append(f"({seed},{float(p['adamw_best']):.6f})")
        coords_muon.append(f"({seed},{float(p['muon_best']):.6f})")
    symbolic = ",".join(p["seed"] for p in pairs)
    tex = PGF / "fig_125m_best_val_loss_pgfplots.tex"
    tex.write_text(rf"""
\documentclass[border=6pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\definecolor{{adamw}}{{HTML}}{{0072B2}}
\definecolor{{muon}}{{HTML}}{{D55E00}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=12cm,height=6.2cm,
    ybar=0pt,
    bar width=16pt,
    ymin=3.62,ymax=3.86,
    ylabel={{Best validation loss $\downarrow$}},
    xlabel={{Random seed}},
    title={{125M FineWeb-Edu: AdamW vs Hybrid Muon}},
    symbolic x coords={{{symbolic}}},
    xtick=data,
    xticklabels={{{symbolic}}},
    grid=major,
    grid style={{black!10}},
    legend style={{draw=none, at={{(0.5,1.12)}}, anchor=south, legend columns=2}},
    nodes near coords,
    every node near coord/.append style={{font=\scriptsize, rotate=0, anchor=south}},
]
\addplot+[fill=adamw, draw=adamw] coordinates {{{' '.join(coords_adamw)}}};
\addlegendentry{{AdamW}}
\addplot+[fill=muon, draw=muon] coordinates {{{' '.join(coords_muon)}}};
\addlegendentry{{Hybrid Muon}}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
""", encoding="utf-8")
    compile_latex(tex)


def write_svg_architecture() -> None:
    out = SVG / "fig_minigpt_architecture.svg"
    boxes = [
        (30, 80, 150, 70, "Token IDs", "$B × T", "#F4F4F2", "#555"),
        (220, 80, 180, 70, "Embeddings", "token + position", "#E8F1FA", "#0072B2"),
        (455, 65, 220, 100, "12× Transformer Blocks", "d=768, h=12, MLP=3072", "#FFF1E8", "#D55E00"),
        (730, 80, 155, 70, "Final Norm", "LayerNorm", "#E8F1FA", "#0072B2"),
        (930, 80, 175, 70, "Tied LM Head", "logits over 50,257", "#E8F1FA", "#0072B2"),
        (420, 230, 170, 65, "Attention", "Q/K/V/O matrices", "#FFF1E8", "#D55E00"),
        (625, 230, 170, 65, "MLP", "up/down matrices", "#FFF1E8", "#D55E00"),
        (220, 230, 170, 65, "AdamW fallback", "embeddings/norms", "#E8F1FA", "#0072B2"),
    ]
    def rect(x,y,w,h,title,sub,fill,stroke):
        return f'''<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>\n<text x="{x+w/2}" y="{y+30}" text-anchor="middle" font-weight="700">{title}</text>\n<text x="{x+w/2}" y="{y+52}" text-anchor="middle" font-size="13" fill="#444">{sub}</text>'''
    arrows = [
        (180,115,220,115,'#555'), (400,115,455,115,'#555'), (675,115,730,115,'#555'), (885,115,930,115,'#555'),
        (565,165,505,230,'#D55E00'), (590,165,710,230,'#D55E00'), (310,150,305,230,'#0072B2'),
    ]
    arrow_svg = []
    for x1,y1,x2,y2,c in arrows:
        arrow_svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2.5" marker-end="url(#arrow)"/>')
    body = "\n".join(rect(*b) for b in boxes) + "\n" + "\n".join(arrow_svg)
    out.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="360" viewBox="0 0 1140 360">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#555"/></marker></defs>
<rect width="100%" height="100%" fill="white"/>
<text x="570" y="36" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">MiniGPT-Dense-125M-v1 architecture</text>
<g font-family="Arial, sans-serif" font-size="15">{body}</g>
<text x="570" y="335" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#333">Muon: matrix-aware updates for attention/MLP matrices. AdamW: embeddings, norms, vectors, fallback parameters.</text>
</svg>''', encoding="utf-8")


def write_netron_onnx_schematic() -> None:
    import onnx
    from onnx import TensorProto, helper
    nodes = []
    nodes.append(helper.make_node("TokenEmbedding", ["token_ids"], ["x_tok"], domain="leanml", vocab=MODEL["vocab"], hidden=MODEL["hidden"]))
    nodes.append(helper.make_node("PositionEmbedding", ["positions"], ["x_pos"], domain="leanml", context=MODEL["context"], hidden=MODEL["hidden"]))
    nodes.append(helper.make_node("Add", ["x_tok", "x_pos"], ["x0"]))
    last = "x0"
    for i in range(1, MODEL["layers"] + 1):
        out = f"x_block_{i}"
        nodes.append(helper.make_node(
            "TransformerBlock", [last], [out], domain="leanml",
            layer=i, hidden=MODEL["hidden"], heads=MODEL["heads"], head_dim=MODEL["head_dim"], mlp_hidden=MODEL["mlp"],
            attention="q,k,v,o", residuals=2,
        ))
        last = out
    nodes.append(helper.make_node("LayerNorm", [last], ["x_norm"], domain="leanml", hidden=MODEL["hidden"]))
    nodes.append(helper.make_node("TiedLMHead", ["x_norm"], ["logits"], domain="leanml", vocab=MODEL["vocab"], hidden=MODEL["hidden"]))
    graph = helper.make_graph(
        nodes,
        "MiniGPT-Dense-125M-v1 schematic",
        [
            helper.make_tensor_value_info("token_ids", TensorProto.INT64, ["B", "T"]),
            helper.make_tensor_value_info("positions", TensorProto.INT64, ["T"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["B", "T", MODEL["vocab"]])],
    )
    model = helper.make_model(graph, producer_name="lean-ml", opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("leanml", 1)])
    onnx.save(model, NETRON / "minigpt_125m_schematic.onnx")


def write_tensorboard_graph() -> None:
    sys.path.insert(0, str(ROOT / "gpu_benchmark"))
    import torch
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    import train_gpt

    class LogitsOnly(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model
        def forward(self, idx):
            logits, _ = self.model(idx)
            return logits

    cfg = train_gpt.build_model_config("minigpt_dense_125m_v1", 50257, block_size_override=32)
    model = train_gpt.MiniGPT(cfg).eval()
    writer = SummaryWriter(log_dir=str(TB / "minigpt_125m_graph"))
    example = torch.zeros((1, 16), dtype=torch.long)
    writer.add_graph(LogitsOnly(model), example)
    writer.flush()
    writer.close()
    (TB / "README.md").write_text("Open with: tensorboard --logdir figures/tensorboard\n", encoding="utf-8")


def write_visualkeras() -> None:
    """Render a VisualKeras layered view from a lightweight conceptual model.

    VisualKeras expects Keras-like layer objects with `.output_shape`. Our
    benchmark model is PyTorch, so this intentionally builds a conceptual layer
    list with the same high-level GPT structure rather than a trainable Keras
    clone.
    """
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import visualkeras
    from PIL import ImageFont

    class FakeModel:
        def __init__(self, layers):
            self.layers = layers

    def layer_class(name: str):
        return type(name, (), {})

    def make_layer(class_name: str, name: str, output_shape: tuple):
        cls = layer_class(class_name)
        obj = cls()
        obj.name = name
        obj.output_shape = output_shape
        return obj

    layers = [
        make_layer("TokenEmbedding", "token_embedding", (None, MODEL["context"], MODEL["hidden"])),
        make_layer("PositionEmbedding", "position_embedding", (None, MODEL["context"], MODEL["hidden"])),
        make_layer("TransformerBlockStack", "12x_transformer_blocks", (None, MODEL["context"], MODEL["hidden"])),
        make_layer("FinalLayerNorm", "final_layernorm", (None, MODEL["context"], MODEL["hidden"])),
        make_layer("TiedLMHead", "tied_lm_head", (None, MODEL["context"], MODEL["vocab"])),
    ]
    model = FakeModel(layers)
    font = None
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        pass
    img = visualkeras.layered_view(
        model,
        to_file=str(VK / "fig_visualkeras_minigpt_125m.png"),
        legend=True,
        draw_volume=True,
        spacing=30,
        scale_xy=0.35,
        scale_z=0.006,
        max_xy=260,
        max_z=160,
        min_z=28,
        font=font,
    )
    img.save(VK / "fig_visualkeras_minigpt_125m.png")
    (VK / "README.md").write_text(
        "VisualKeras is Keras-oriented. This diagram is a conceptual Keras-like layered view of the PyTorch GPT architecture, not a trainable Keras clone used in the benchmark.\n",
        encoding="utf-8",
    )


def write_matplotlib_3d() -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def cuboid_data(origin, size):
        x, y, z = origin
        dx, dy, dz = size
        X = [x, x + dx]
        Y = [y, y + dy]
        Z = [z, z + dz]
        return [
            [(X[0], Y[0], Z[0]), (X[1], Y[0], Z[0]), (X[1], Y[1], Z[0]), (X[0], Y[1], Z[0])],
            [(X[0], Y[0], Z[1]), (X[1], Y[0], Z[1]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
            [(X[0], Y[0], Z[0]), (X[1], Y[0], Z[0]), (X[1], Y[0], Z[1]), (X[0], Y[0], Z[1])],
            [(X[0], Y[1], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
            [(X[0], Y[0], Z[0]), (X[0], Y[1], Z[0]), (X[0], Y[1], Z[1]), (X[0], Y[0], Z[1])],
            [(X[1], Y[0], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[1], Y[0], Z[1])],
        ]

    fig = plt.figure(figsize=(9.2, 4.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    layers = [
        ("Embeddings", 0.0, 0.75, "#0072B2"),
        ("Block 1", 1.25, 0.75, "#D55E00"),
        ("Block 2", 2.25, 0.75, "#D55E00"),
        ("...", 3.25, 0.45, "#B0B0B0"),
        ("Block 12", 4.15, 0.75, "#D55E00"),
        ("Final Norm", 5.45, 0.58, "#0072B2"),
        ("LM Head", 6.55, 0.85, "#009E73"),
    ]
    for label, x, dx, color in layers:
        verts = cuboid_data((x, 0, 0), (dx, 1.35, 2.1))
        poly = Poly3DCollection(verts, facecolors=color, edgecolors="white", linewidths=0.8, alpha=0.9)
        ax.add_collection3d(poly)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#0072B2", label="Embeddings / norms"),
        Patch(facecolor="#D55E00", label="12 Transformer blocks"),
        Patch(facecolor="#009E73", label="Tied LM head"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.05))
    fig.text(0.5, 0.13, "hidden d=768   heads=12   MLP=3072   context=512   vocab=50,257", ha="center", fontsize=9)
    ax.set_xlim(-0.2, 7.8)
    ax.set_ylim(-0.55, 2.0)
    ax.set_zlim(0, 2.9)
    ax.view_init(elev=21, azim=-56)
    ax.set_axis_off()
    ax.set_title("MiniGPT-Dense-125M-v1: 3D layer stack", pad=12, fontweight="bold")
    fig.savefig(FIG / "fig_minigpt_125m_3d_stack.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_minigpt_125m_3d_stack.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_plotneuralnet() -> None:
    vendor = FIG / "vendor" / "PlotNeuralNet"
    sys.path.insert(0, str(vendor))
    from pycore.tikzeng import to_Conv, to_ConvConvRelu, to_ConvSoftMax, to_connection, to_cor, to_begin, to_end, to_generate, to_head

    arch = [
        to_head(str(vendor)),
        to_cor(),
        to_begin(),
        to_Conv("embed", s_filer=512, n_filer=768, offset="(0,0,0)", to="(0,0,0)", width=2, height=36, depth=36, caption="Embeddings"),
        to_ConvConvRelu("blocks", s_filer=512, n_filer=(768, 3072), offset="(1.6,0,0)", to="(embed-east)", width=(4,4), height=44, depth=44, caption="12x Transformer Blocks"),
        to_Conv("norm", s_filer=512, n_filer=768, offset="(1.6,0,0)", to="(blocks-east)", width=1.2, height=34, depth=34, caption="Final Norm"),
        to_ConvSoftMax("lmhead", s_filer=50257, offset="(1.6,0,0)", to="(norm-east)", width=1.5, height=42, depth=42, caption="Tied LM Head"),
        to_connection("embed", "blocks"),
        to_connection("blocks", "norm"),
        to_connection("norm", "lmhead"),
        to_end(),
    ]
    tex = PNN / "fig_plotneuralnet_minigpt_125m.tex"
    to_generate(arch, str(tex))
    compile_latex(tex)


def write_readme() -> None:
    (FIG / "ARCHITECTURE_FIGURES.md").write_text("""# Architecture and figure artifacts

Generated artifacts:

## Matplotlib numeric figures
- `fig_125m_best_val_loss_by_seed.pdf/png`
- `fig_125m_muon_delta_by_seed.pdf/png`
- `fig_125m_validation_curves_by_seed.pdf/png`
- `fig_125m_quality_speed_tradeoff.pdf/png`
- `fig_scaling_summary_57m_to_125m.pdf/png`
- `fig_minigpt_125m_3d_stack.pdf/png`

## TikZ / PGFPlots
- `tikz/fig_minigpt_architecture_tikz.tex/pdf`
- `pgfplots/fig_125m_best_val_loss_pgfplots.tex/pdf`

## PlotNeuralNet
- `plotneuralnet/fig_plotneuralnet_minigpt_125m.tex/pdf`
- vendor source: `vendor/PlotNeuralNet/`

## SVG
- `svg/fig_minigpt_architecture.svg`

## Netron
- `netron/minigpt_125m_schematic.onnx`
- Open with: `python -m netron figures/netron/minigpt_125m_schematic.onnx`

## TensorBoard
- `tensorboard/minigpt_125m_graph/`
- Open with: `tensorboard --logdir figures/tensorboard`

## VisualKeras
- `visualkeras/fig_visualkeras_minigpt_125m.png`
- This is a conceptual Keras-backed view of the PyTorch architecture, not the benchmark training model.
""", encoding="utf-8")


def main() -> None:
    write_tikz_architecture()
    write_pgfplots_result()
    write_svg_architecture()
    write_netron_onnx_schematic()
    write_tensorboard_graph()
    write_visualkeras()
    write_matplotlib_3d()
    write_plotneuralnet()
    write_readme()
    print("Generated architecture artifacts under figures/")


if __name__ == "__main__":
    main()
