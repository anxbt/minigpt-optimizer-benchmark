#!/usr/bin/env python3
"""From-scratch MiniGPT training script: AdamW vs Hybrid Muon.

This file is intentionally verbose and heavily commented because it is also a
learning artifact. The code follows `spec.md` and implements the first locked
benchmark:

    MiniGPT-Dense-60M-v1 on TinyStories

Key features:
- decoder-only GPT model written directly in PyTorch
- AdamW optimizer OR Hybrid Muon optimizer
- TinyStories tokenized-cache preparation
- local toy dataset fallback for code smoke tests
- JSONL metrics logging
- checkpoint save/resume
- STOP-file safe exit
- basic plots and sample generations

Run tiny local smoke, no Hugging Face dataset required:

    python gpu_benchmark/train_gpt.py --dataset toy --run-dir /tmp/minigpt_toy \
      --optimizer adamw --model-config tiny_debug --max-steps 5

Run EC2 TinyStories smoke:

    python gpu_benchmark/train_gpt.py --dataset tinystories --run-dir ~/pretrain_runs/... \
      --optimizer adamw --model-config minigpt_dense_60m_v1 --precision bf16 \
      --micro-batch-size 4 --grad-accum-steps 8 --max-minutes 10
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import shutil
import signal
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Matplotlib is only used for saved PNG plots. The Agg backend works on headless EC2.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Small utility functions
# -----------------------------------------------------------------------------


def utc_now() -> str:
    """Return an ISO timestamp in UTC.

    We use UTC in logs so EC2/local timezone differences do not confuse us.
    """
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """Write a dict as pretty JSON, atomically enough for our use case."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    """Append one JSON object to a .jsonl file.

    JSONL means "JSON lines": each line is a separate JSON record. It is a good
    format for training logs because we can keep appending without rewriting a
    large file.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def file_size(path: Path) -> int:
    """Return file size in bytes, or 0 if the file does not exist."""
    return path.stat().st_size if path.exists() else 0


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Return total, embedding, and non-embedding parameter counts.

    The paper's FLOPs estimate usually uses non-embedding parameters. Embedding
    parameters are the token lookup table and are often counted separately.
    """
    total = sum(p.numel() for p in model.parameters())
    emb = 0
    for name, p in model.named_parameters():
        if "embedding" in name or "token_embedding" in name or "position_embedding" in name:
            emb += p.numel()
    return total, emb, total - emb


def get_device() -> torch.device:
    """Pick the best available training device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducibility_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for comparable runs.

    RNG means random-number generator. We need this for optimizer comparisons:
    if AdamW and Muon start from different random model weights, then part of
    the loss difference may come from luck instead of the optimizer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_memory_stats(device: torch.device) -> dict[str, int]:
    """Return GPU memory stats, or zeros on CPU/MPS.

    PyTorch exposes detailed allocation stats for CUDA. For MPS/CPU we keep the
    keys but return zeros to keep metrics schema consistent.
    """
    if device.type != "cuda":
        return {
            "gpu_memory_allocated_bytes": 0,
            "gpu_memory_reserved_bytes": 0,
            "gpu_max_memory_allocated_bytes": 0,
        }
    return {
        "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """All architectural knobs for our GPT model.

    These names intentionally mirror common GPT/nanoGPT terminology:
    - n_layer: number of Transformer blocks
    - n_head: number of attention heads
    - n_embd: hidden size, also called model dimension
    - block_size: context length in tokens
    - vocab_size: tokenizer vocabulary size
    """

    name: str
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False
    tie_lm_head: bool = True

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden_size(self) -> int:
        return 4 * self.n_embd


def build_model_config(name: str, vocab_size: int, block_size_override: int | None = None) -> ModelConfig:
    """Return a locked model config by name.

    `tiny_debug` exists only for local smoke tests. The benchmark config is
    `minigpt_dense_60m_v1`.
    """
    if name == "tiny_debug":
        cfg = ModelConfig(
            name=name,
            vocab_size=vocab_size,
            block_size=64,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            bias=False,
            tie_lm_head=True,
        )
    elif name == "minigpt_dense_60m_v1":
        cfg = ModelConfig(
            name="MiniGPT-Dense-60M-v1",
            vocab_size=vocab_size,
            block_size=512,
            n_layer=10,
            n_head=8,
            n_embd=512,
            dropout=0.0,
            bias=False,
            tie_lm_head=True,
        )
    elif name == "minigpt_dense_125m_v1":
        # GPT-2-small / nanoGPT-like scale.
        # This is the next benchmark size after the 57M model.
        # We keep context length at 512 first to reduce OOM risk on one L40S.
        cfg = ModelConfig(
            name="MiniGPT-Dense-125M-v1",
            vocab_size=vocab_size,
            block_size=512,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=False,
            tie_lm_head=True,
        )
    else:
        raise ValueError(f"Unknown model config: {name}")

    if block_size_override is not None:
        cfg.block_size = block_size_override
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")
    return cfg


# -----------------------------------------------------------------------------
# GPT model implementation
# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    "Causal" means token position t can only look at positions <= t. This is
    what makes next-token prediction honest: the model cannot cheat by looking
    at future tokens.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        # These are the attention projection matrices. In our Hybrid Muon run,
        # these hidden matrix weights are Muon candidates.
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, time, hidden]
        B, T, C = x.shape

        # Project hidden states into Q, K, V.
        # After projection, split hidden dim into attention heads:
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # PyTorch's scaled_dot_product_attention is memory-efficient and supports
        # causal masking internally when is_causal=True.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads back: [B, n_head, T, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection mixes the heads back into hidden space.
        return self.o_proj(y)


class MLP(nn.Module):
    """The Transformer feed-forward network.

    This is the block's private per-token computation. It expands hidden size by
    4x, applies GELU, then projects back down.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(cfg.n_embd, cfg.mlp_hidden_size, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.mlp_hidden_size, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = F.gelu(x)
        x = self.down_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """One GPT Transformer block using pre-LayerNorm.

    Pre-LayerNorm means LayerNorm is applied before attention/MLP. This tends to
    train more stably than the original GPT-2 post-LayerNorm style.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection: x + sublayer(x). The residual path lets gradients
        # flow through many layers without vanishing too badly.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """Small decoder-only GPT model.

    Input: token IDs of shape [batch, time]
    Output: logits of shape [batch, time, vocab_size]

    "Logits" are raw scores before softmax. The training loss compares logits
    to the correct next-token IDs.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding: converts token ID -> vector.
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # Learned absolute position embedding: tells the model where each token is.
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # If weights are tied, the LM head uses token_embedding.weight directly.
        # This saves parameters and is common in language models.
        if not cfg.tie_lm_head:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with a GPT-like normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.cfg.block_size}")

        # Create position IDs: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, device=idx.device)

        # Add token and position embeddings. Shape becomes [B, T, hidden].
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        if self.cfg.tie_lm_head:
            # Matrix multiply hidden states by embedding table transposed:
            # [B,T,H] @ [H,V] -> [B,T,V]
            logits = x @ self.token_embedding.weight.T
        else:
            logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy expects [N, classes] logits and [N] target IDs, so we
            # flatten batch/time into one dimension.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        """Autoregressively sample tokens from the model."""
        self.eval()
        for _ in range(max_new_tokens):
            # Keep only the last block_size tokens because model context is finite.
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            # Use only last position logits to predict the next token.
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# -----------------------------------------------------------------------------
# Muon optimizer implementation
# -----------------------------------------------------------------------------


def zeropower_newton_schulz_5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate matrix orthogonalization used by Muon.

    Input `g` is a 2D matrix, usually a momentum-like gradient update.
    Output is another 2D matrix of the same shape whose singular values are more
    balanced. We use fp32 internally for stability, even during bf16 training.
    """
    if g.ndim != 2:
        raise ValueError("Newton-Schulz Muon update expects a 2D matrix")

    original_dtype = g.dtype
    x = g.float()
    norm = x.norm()
    if not torch.isfinite(norm) or norm < eps:
        return torch.zeros_like(g)
    x = x / (norm + eps)

    # For numerical efficiency, run iterations on a matrix with rows <= cols.
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    # Coefficients from the Muon paper / original Muon implementation.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        b_x = xx_t @ x
        x = a * x + b * b_x + c * (xx_t @ b_x)

    if transposed:
        x = x.T
    return x.to(original_dtype)


def is_muon_parameter(name: str, p: nn.Parameter) -> bool:
    """Decide whether a parameter should be updated by Muon.

    This implements the locked MiniGPT-Dense-60M-v1 split.
    Muon gets hidden matrix weights only.
    """
    if p.ndim != 2:
        return False
    if "embedding" in name or "lm_head" in name:
        return False
    return any(
        key in name
        for key in [
            "attn.q_proj.weight",
            "attn.k_proj.weight",
            "attn.v_proj.weight",
            "attn.o_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
    )


class HybridMuon:
    """Hybrid Muon optimizer wrapper.

    It contains:
    - a real torch.optim.AdamW for fallback parameters
    - custom Muon logic for selected hidden matrix weights

    We implement state_dict/load_state_dict so checkpoints can resume.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, nn.Parameter]],
        lr: float,
        weight_decay: float,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        muon_momentum: float = 0.95,
        ns_steps: int = 5,
        rms_scale_enabled: bool = True,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.muon_momentum = muon_momentum
        self.ns_steps = ns_steps
        self.rms_scale_enabled = rms_scale_enabled

        self.muon_params: list[tuple[str, nn.Parameter]] = []
        adamw_params: list[nn.Parameter] = []
        self.adamw_param_names: list[str] = []

        for name, p in named_parameters:
            if not p.requires_grad:
                continue
            if is_muon_parameter(name, p):
                self.muon_params.append((name, p))
            else:
                adamw_params.append(p)
                self.adamw_param_names.append(name)

        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
        )

        # Momentum buffer per Muon matrix, keyed by parameter name for stable checkpointing.
        self.momentum_buffers: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p) for name, p in self.muon_params
        }
        self.last_update_rms: dict[str, float] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adamw.zero_grad(set_to_none=set_to_none)
        for _, p in self.muon_params:
            if set_to_none:
                p.grad = None
            else:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        # AdamW updates fallback parameters first.
        self.adamw.step()

        self.last_update_rms = {}
        for name, p in self.muon_params:
            if p.grad is None:
                continue
            if p.ndim != 2:
                raise RuntimeError(f"Muon selected non-2D parameter {name}: {tuple(p.shape)}")

            grad = p.grad
            buf = self.momentum_buffers[name]
            buf.mul_(self.muon_momentum).add_(grad)

            # Nesterov-style momentum update: use current grad plus momentum lookahead.
            momentum_update = grad + self.muon_momentum * buf
            ortho = zeropower_newton_schulz_5(momentum_update, steps=self.ns_steps)

            if self.rms_scale_enabled:
                rows, cols = p.shape
                scale = 0.2 * math.sqrt(max(rows, cols))
            else:
                scale = 1.0
            update = ortho * scale

            self.last_update_rms[name] = update.float().pow(2).mean().sqrt().item()

            # Decoupled weight decay like AdamW: shrink parameter directly.
            if self.weight_decay != 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            p.add_(update, alpha=-self.lr)

    def set_lr(self, lr: float) -> None:
        """Set learning rate for both AdamW fallback and Muon matrices."""
        self.lr = lr
        for group in self.adamw.param_groups:
            group["lr"] = lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "muon_momentum": self.muon_momentum,
            "ns_steps": self.ns_steps,
            "rms_scale_enabled": self.rms_scale_enabled,
            "adamw": self.adamw.state_dict(),
            "momentum_buffers": {k: v.detach().cpu() for k, v in self.momentum_buffers.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]
        self.muon_momentum = state["muon_momentum"]
        self.ns_steps = state["ns_steps"]
        self.rms_scale_enabled = state.get("rms_scale_enabled", True)
        self.adamw.load_state_dict(state["adamw"])

        # Move loaded CPU buffers back to the parameter device/dtype.
        loaded = state["momentum_buffers"]
        for name, p in self.muon_params:
            if name in loaded:
                self.momentum_buffers[name] = loaded[name].to(device=p.device, dtype=p.dtype)

    def param_group_manifest(self) -> dict[str, Any]:
        muon = [
            {"name": name, "shape": list(p.shape), "numel": p.numel()}
            for name, p in self.muon_params
        ]
        adamw = [
            {"name": name, "numel": None}
            for name in self.adamw_param_names
        ]
        return {
            "muon_params": muon,
            "adamw_param_names": adamw,
            "totals": {
                "muon_matrix_count": len(muon),
                "muon_numel": sum(x["numel"] for x in muon),
                "adamw_param_count": len(adamw),
            },
        }

    def update_rms_summary(self) -> dict[str, Any] | None:
        if not self.last_update_rms:
            return None
        vals = list(self.last_update_rms.values())
        return {
            "avg_update_rms": float(sum(vals) / len(vals)),
            "min_update_rms": float(min(vals)),
            "max_update_rms": float(max(vals)),
            "num_muon_matrices": len(vals),
        }


# -----------------------------------------------------------------------------
# Dataset/token cache
# -----------------------------------------------------------------------------


TOY_TEXT = """\
Once upon a time there was a small cat who liked stories.
The little girl found a bright blue stone near the river.
In a small village, children learned to read and write.
The dog ran through the garden and chased a red ball.
A kind teacher told a story about the moon and stars.
""" * 200


class TokenBinDataset:
    """Random contiguous-token batches from a .bin token file.

    Language model training usually samples chunks like:
    x = tokens[i : i + block_size]
    y = tokens[i + 1 : i + block_size + 1]
    """

    def __init__(self, bin_path: Path, block_size: int, dtype: np.dtype = np.uint16):
        self.bin_path = bin_path
        self.block_size = block_size
        self.tokens = np.memmap(bin_path, dtype=dtype, mode="r")
        if len(self.tokens) <= block_size + 1:
            raise ValueError(f"Not enough tokens in {bin_path}")

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Random start indices. We leave room for block_size+1 because y is shifted.
        ix = np.random.randint(0, len(self.tokens) - self.block_size - 1, size=(batch_size,))
        x_np = np.stack([self.tokens[i : i + self.block_size].astype(np.int64) for i in ix])
        y_np = np.stack([self.tokens[i + 1 : i + self.block_size + 1].astype(np.int64) for i in ix])
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
        y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
        return x, y

    @property
    def token_count(self) -> int:
        return int(len(self.tokens))


def lazy_import_tokenizer():
    """Import the Hugging Face tokenizer dependency only when needed."""
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Tokenization requires `transformers`. "
            "Install it in the active environment, e.g. `pip install transformers`."
        ) from e
    return AutoTokenizer


def lazy_import_transformers_and_datasets():
    """Import optional Hugging Face dependencies only when TinyStories is needed."""
    try:
        from transformers import AutoTokenizer
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Hugging Face dataset preparation requires `transformers` and `datasets`. "
            "Install them in the active environment, e.g. `pip install transformers datasets`."
        ) from e
    return AutoTokenizer, load_dataset


def prepare_toy_cache(cache_dir: Path, tokenizer_name: str, block_size: int) -> dict[str, Any]:
    """Prepare a tiny local dataset cache for smoke tests.

    This avoids depending on network/Hugging Face for local syntax testing.
    """
    AutoTokenizer = lazy_import_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ensure_dir(cache_dir)
    ids = tokenizer.encode(TOY_TEXT)
    split = int(0.9 * len(ids))
    train_ids = np.array(ids[:split], dtype=np.uint16)
    val_ids = np.array(ids[split:], dtype=np.uint16)
    train_ids.tofile(cache_dir / "train.bin")
    val_ids.tofile(cache_dir / "validation.bin")
    meta = {
        "dataset": "toy",
        "tokenizer": tokenizer_name,
        "vocab_size": int(tokenizer.vocab_size),
        "block_size": block_size,
        "train_tokens": int(len(train_ids)),
        "validation_tokens": int(len(val_ids)),
        "dtype": "uint16",
        "created_at": utc_now(),
    }
    write_json(cache_dir / "metadata.json", meta)
    return meta


def prepare_tinystories_cache(
    cache_dir: Path,
    tokenizer_name: str,
    block_size: int,
    max_train_tokens: int | None = None,
    max_val_tokens: int | None = None,
) -> dict[str, Any]:
    """Download/tokenize TinyStories into local .bin files.

    For first versions we create one train.bin and one validation.bin. Later, if
    needed, this can become multiple shards.
    """
    AutoTokenizer, load_dataset = lazy_import_transformers_and_datasets()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ensure_dir(cache_dir)

    def tokenize_split(split: str, out_path: Path, max_tokens: int | None) -> int:
        """Tokenize one HF split into a flat uint16 .bin file.

        Important implementation detail:
        we stream encoded token IDs to disk in chunks instead of keeping the
        entire tokenized dataset in Python memory. The 1M-token smoke cache can
        fit in RAM either way, but full TinyStories is large enough that a big
        list of tiny NumPy arrays is wasteful and easier to crash.
        """
        ds = load_dataset("roneneldan/TinyStories", split=split)
        total = 0
        rows_seen = 0
        pending: list[int] = []

        # GPT-2 has an end-of-text token. We append it after each story so the
        # model sees a boundary between separate stories.
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = tokenizer.encode("<|endoftext|>")[0]

        # Flush every N tokens. Larger flushes reduce disk overhead while keeping
        # memory bounded. 1M uint16 tokens is only about 2 MB.
        flush_tokens = 1_000_000

        def flush_pending(f) -> None:
            nonlocal pending
            if not pending:
                return
            arr = np.asarray(pending, dtype=np.uint16)
            arr.tofile(f)
            pending = []

        with out_path.open("wb") as f:
            for row in ds:
                rows_seen += 1
                text = row.get("text", "")
                ids = tokenizer.encode(text)
                ids.append(eos)

                if max_tokens is not None and total + len(ids) > max_tokens:
                    ids = ids[: max(0, max_tokens - total)]

                if ids:
                    pending.extend(ids)
                    total += len(ids)

                if len(pending) >= flush_tokens:
                    flush_pending(f)

                # Progress matters for long EC2 data jobs. Print occasionally so
                # stdout.log shows that the process is alive.
                if rows_seen % 50_000 == 0:
                    print(f"tokenized {split}: rows={rows_seen:,} tokens={total:,}", flush=True)

                if max_tokens is not None and total >= max_tokens:
                    break

            flush_pending(f)

        if total == 0:
            raise RuntimeError(f"No tokens produced for split {split}")
        return int(total)

    train_tokens = tokenize_split("train", cache_dir / "train.bin", max_train_tokens)
    # TinyStories has validation split in HF dataset. If this ever changes, load_dataset will fail loudly.
    val_tokens = tokenize_split("validation", cache_dir / "validation.bin", max_val_tokens)

    meta = {
        "dataset": "roneneldan/TinyStories",
        "tokenizer": tokenizer_name,
        "vocab_size": int(tokenizer.vocab_size),
        "block_size": block_size,
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "dtype": "uint16",
        "num_train_shards": 1,
        "num_validation_shards": 1,
        "cache_dir": str(cache_dir),
        "created_at": utc_now(),
    }
    write_json(cache_dir / "metadata.json", meta)
    return meta



def prepare_finewebedu_cache(
    cache_dir: Path,
    tokenizer_name: str,
    block_size: int,
    max_train_tokens: int | None = None,
    max_val_tokens: int | None = None,
    config_name: str = "sample-10BT",
) -> dict[str, Any]:
    """Stream a capped FineWeb-Edu token cache into train.bin/validation.bin.

    FineWeb-Edu is much larger than TinyStories. We must not accidentally prepare
    the entire dataset on a single EC2 disk. Therefore this function requires an
    explicit train-token cap and uses Hugging Face streaming.

    The dataset has a train split. We create validation.bin from the first
    max_val_tokens tokens, then train.bin from the next max_train_tokens tokens.
    This gives us a stable held-out validation slice without needing a separate
    validation split.
    """
    if max_train_tokens is None:
        raise ValueError(
            "FineWeb-Edu cache preparation requires --max-train-tokens. "
            "Use a cap such as 500000000 for a first single-GPU experiment."
        )
    if max_val_tokens is None:
        max_val_tokens = min(5_000_000, max(262_144, max_train_tokens // 100))

    AutoTokenizer, load_dataset = lazy_import_transformers_and_datasets()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ensure_dir(cache_dir)

    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.encode("<|endoftext|>")[0]

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=config_name, split="train", streaming=True)

    train_path = cache_dir / "train.bin"
    val_path = cache_dir / "validation.bin"
    flush_tokens = 1_000_000
    train_total = 0
    val_total = 0
    rows_seen = 0
    pending_train: list[int] = []
    pending_val: list[int] = []

    def flush(f, pending: list[int]) -> list[int]:
        if pending:
            np.asarray(pending, dtype=np.uint16).tofile(f)
        return []

    with val_path.open("wb") as vf, train_path.open("wb") as tf:
        for row in ds:
            rows_seen += 1
            text = row.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text)
            ids.append(eos)

            cursor = 0
            if val_total < max_val_tokens:
                take = min(len(ids), max_val_tokens - val_total)
                if take > 0:
                    pending_val.extend(ids[:take])
                    val_total += take
                    cursor += take

            if cursor < len(ids) and train_total < max_train_tokens:
                take = min(len(ids) - cursor, max_train_tokens - train_total)
                if take > 0:
                    pending_train.extend(ids[cursor:cursor + take])
                    train_total += take

            if len(pending_val) >= flush_tokens:
                pending_val = flush(vf, pending_val)
            if len(pending_train) >= flush_tokens:
                pending_train = flush(tf, pending_train)

            if rows_seen % 10_000 == 0:
                print(
                    f"tokenized finewebedu/{config_name}: rows={rows_seen:,} "
                    f"train_tokens={train_total:,} val_tokens={val_total:,}",
                    flush=True,
                )

            if val_total >= max_val_tokens and train_total >= max_train_tokens:
                break

        pending_val = flush(vf, pending_val)
        pending_train = flush(tf, pending_train)

    if train_total == 0 or val_total == 0:
        raise RuntimeError("FineWeb-Edu tokenization produced empty train or validation cache")

    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": config_name,
        "tokenizer": tokenizer_name,
        "vocab_size": int(tokenizer.vocab_size),
        "block_size": block_size,
        "train_tokens": int(train_total),
        "validation_tokens": int(val_total),
        "dtype": "uint16",
        "num_train_shards": 1,
        "num_validation_shards": 1,
        "cache_dir": str(cache_dir),
        "max_train_tokens": int(max_train_tokens),
        "max_val_tokens": int(max_val_tokens),
        "created_at": utc_now(),
    }
    write_json(cache_dir / "metadata.json", meta)
    return meta

def ensure_token_cache(args: argparse.Namespace, cfg: ModelConfig) -> dict[str, Any]:
    """Make sure tokenized train/validation .bin files exist."""
    cache_dir = Path(args.data_dir).expanduser()
    meta_path = cache_dir / "metadata.json"
    if args.prepare_data or not (cache_dir / "train.bin").exists() or not (cache_dir / "validation.bin").exists():
        print(f"Preparing token cache in {cache_dir} ...")
        if args.dataset == "toy":
            return prepare_toy_cache(cache_dir, args.tokenizer, cfg.block_size)
        if args.dataset == "tinystories":
            return prepare_tinystories_cache(
                cache_dir,
                args.tokenizer,
                cfg.block_size,
                max_train_tokens=args.max_train_tokens,
                max_val_tokens=args.max_val_tokens,
            )
        if args.dataset == "finewebedu":
            return prepare_finewebedu_cache(
                cache_dir,
                args.tokenizer,
                cfg.block_size,
                max_train_tokens=args.max_train_tokens,
                max_val_tokens=args.max_val_tokens,
                config_name=args.fineweb_config,
            )
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return json.loads(meta_path.read_text())


# -----------------------------------------------------------------------------
# Learning-rate schedule
# -----------------------------------------------------------------------------


def cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    """Linear warmup then cosine decay.

    Warmup prevents unstable early training by slowly increasing LR from 0 to
    base_lr. Cosine decay gradually lowers LR toward min_lr.
    """
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    if isinstance(optimizer, HybridMuon):
        optimizer.set_lr(lr)
    else:
        for group in optimizer.param_groups:
            group["lr"] = lr


# -----------------------------------------------------------------------------
# Checkpointing and run artifacts
# -----------------------------------------------------------------------------


class RunState:
    """Mutable run state shared with signal handlers."""

    def __init__(self):
        self.should_stop = False
        self.stop_reason = ""


RUN_STATE = RunState()


def install_signal_handlers() -> None:
    """Turn Ctrl-C/SIGTERM into safe-stop requests."""
    def handler(signum, frame):  # noqa: ARG001
        RUN_STATE.should_stop = True
        RUN_STATE.stop_reason = f"signal {signum} received"
        print(f"\nSafe-stop requested: {RUN_STATE.stop_reason}", flush=True)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def artifact_manifest(run_dir: Path) -> dict[str, Any]:
    """Create a lightweight manifest of files in run_dir."""
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(run_dir)
            files.append({"path": str(rel), "bytes": file_size(path)})
    return {"updated_at": utc_now(), "run_dir": str(run_dir), "files": files}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Any,
    step: int,
    tokens_seen: int,
    cfg: ModelConfig,
    args: argparse.Namespace,
    best_val_loss: float | None,
) -> None:
    """Save enough state to resume training."""
    ensure_dir(path.parent)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "model_config": asdict(cfg),
        "args": vars(args),
        "best_val_loss": best_val_loss,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_python": random.getstate(),
    }
    if torch.cuda.is_available():
        ckpt["rng_state_cuda"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: Any, device: torch.device) -> tuple[int, int, float | None]:
    """Load checkpoint and restore model/optimizer/RNG state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    torch.set_rng_state(ckpt["rng_state_torch"].cpu())
    try:
        np.random.set_state(ckpt["rng_state_numpy"])
        random.setstate(ckpt["rng_state_python"])
    except Exception:
        pass
    if torch.cuda.is_available() and "rng_state_cuda" in ckpt:
        torch.cuda.set_rng_state_all([x.cpu() for x in ckpt["rng_state_cuda"]])
    return int(ckpt["step"]), int(ckpt["tokens_seen"]), ckpt.get("best_val_loss")


def write_run_summary(
    run_dir: Path,
    status: str,
    step: int,
    tokens_seen: int,
    started_at: str,
    best_val_loss: float | None,
    last_val_loss: float | None,
    latest_checkpoint: str = "checkpoints/latest.pt",
) -> None:
    summary = {
        "status": status,
        "last_step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "last_val_loss": last_val_loss,
        "started_at": started_at,
        "updated_at": utc_now(),
        "latest_checkpoint": latest_checkpoint,
    }
    write_json(run_dir / "run_summary.json", summary)
    write_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir))


# -----------------------------------------------------------------------------
# Evaluation, generation, and plotting
# -----------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    dataset: TokenBinDataset,
    batch_size: int,
    device: torch.device,
    precision: str,
    max_tokens: int,
) -> float:
    """Estimate validation loss over a fixed number of tokens."""
    model.eval()
    losses = []
    tokens_done = 0
    while tokens_done < max_tokens:
        x, y = dataset.get_batch(batch_size, device)
        with autocast_context(device, precision):
            _, loss = model(x, y)
        losses.append(float(loss.detach().cpu()))
        tokens_done += x.numel()
    model.train()
    return float(sum(losses) / len(losses))


@torch.no_grad()
def generate_samples(
    model: MiniGPT,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 80,
) -> str:
    """Generate text samples for human sanity-checking."""
    model.eval()
    chunks = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8)
        text = tokenizer.decode(out[0].detach().cpu().tolist())
        chunks.append(f"PROMPT: {prompt}\n{text}\n")
    model.train()
    return "\n".join(chunks)


def plot_metrics(run_dir: Path) -> None:
    """Create basic plots from metrics.jsonl."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return
    train_steps, train_losses = [], []
    val_tokens, val_losses = [], []
    val_flops = []
    muon_steps, muon_rms = [], []

    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "train_step":
            train_steps.append(row["step"])
            train_losses.append(row["train_loss"])
        elif row.get("event") == "validation":
            val_tokens.append(row["tokens_seen"])
            val_losses.append(row["val_loss"])
            val_flops.append(row.get("estimated_flops", 0.0))
        elif row.get("event") == "muon_update_rms":
            muon_steps.append(row["step"])
            muon_rms.append(row["avg_update_rms"])

    plots_dir = ensure_dir(run_dir / "plots")
    if train_steps:
        plt.figure(figsize=(8, 4))
        plt.plot(train_steps, train_losses, label="train loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training loss vs step")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "train_loss_vs_step.png", dpi=160)
        plt.close()

    if val_tokens:
        plt.figure(figsize=(8, 4))
        plt.plot(val_tokens, val_losses, marker="o", label="val loss")
        plt.xlabel("tokens seen")
        plt.ylabel("validation loss")
        plt.title("Validation loss vs tokens")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_vs_tokens.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(val_flops, val_losses, marker="o", label="val loss")
        plt.xlabel("approximate training FLOPs")
        plt.ylabel("validation loss")
        plt.title("Validation loss vs approximate FLOPs")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_vs_flops.png", dpi=160)
        plt.close()

    if muon_steps:
        plt.figure(figsize=(8, 4))
        plt.plot(muon_steps, muon_rms, label="avg Muon update RMS")
        plt.xlabel("step")
        plt.ylabel("average update RMS")
        plt.title("Muon update RMS")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "muon_update_rms.png", dpi=160)
        plt.close()


def autocast_context(device: torch.device, precision: str):
    """Return a PyTorch autocast context for mixed precision."""
    if precision == "bf16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    # nullcontext without importing contextlib for this one tiny use.
    class NullContext:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return NullContext()


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------


def build_optimizer(args: argparse.Namespace, model: MiniGPT) -> Any:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "muon":
        return HybridMuon(
            model.named_parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.epsilon,
            muon_momentum=args.muon_momentum,
            ns_steps=args.newton_schulz_steps,
            rms_scale_enabled=not args.disable_muon_rms_scale,
        )
    raise ValueError(args.optimizer)


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    """Optionally use torch.compile for speed.

    Disabled by default for easier debugging. Can be enabled later in stable runs.
    """
    if enabled and hasattr(torch, "compile"):
        return torch.compile(model)  # type: ignore[attr-defined]
    return model


def make_config_json(
    args: argparse.Namespace,
    cfg: ModelConfig,
    dataset_meta: dict[str, Any],
    model: nn.Module,
    optimizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    total, emb, non_emb = count_parameters(model)
    gpu_name = None
    gpu_mem = 0
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = int(torch.cuda.get_device_properties(device).total_memory)
    return {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "gpu_memory_bytes": gpu_mem,
            "cpu_count": os.cpu_count(),
        },
        "dataset": dataset_meta,
        "tokenizer": {"name": args.tokenizer, "vocab_size": cfg.vocab_size},
        "model": {
            **asdict(cfg),
            "parameter_count": total,
            "embedding_parameter_count": emb,
            "non_embedding_parameter_count": non_emb,
        },
        "optimizer": {
            "name": args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "epsilon": args.epsilon,
            "muon_momentum": args.muon_momentum if args.optimizer == "muon" else None,
            "newton_schulz_steps": args.newton_schulz_steps if args.optimizer == "muon" else None,
            "muon_rms_scale_enabled": (not args.disable_muon_rms_scale) if args.optimizer == "muon" else None,
        },
        "training": {
            "seed": args.seed,
            "precision": args.precision,
            "micro_batch_size": args.micro_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_tokens": args.micro_batch_size * args.grad_accum_steps * cfg.block_size,
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "max_steps": args.max_steps,
            "max_minutes": args.max_minutes,
        },
    }


def train(args: argparse.Namespace) -> None:
    install_signal_handlers()
    set_reproducibility_seed(args.seed)

    run_dir = ensure_dir(Path(args.run_dir).expanduser())
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    metrics_path = run_dir / "metrics.jsonl"
    started_at = utc_now()

    device = get_device()
    print(f"Using device: {device}")

    # Import tokenizer lazily here; training needs it for sample generation and vocab size.
    AutoTokenizer = lazy_import_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = int(tokenizer.vocab_size)

    cfg = build_model_config(args.model_config, vocab_size=vocab_size, block_size_override=args.block_size)
    dataset_meta = ensure_token_cache(args, cfg)
    if args.prepare_only:
        print("Data preparation complete; exiting because --prepare-only was set.")
        return

    train_data = TokenBinDataset(Path(args.data_dir).expanduser() / "train.bin", cfg.block_size)
    val_data = TokenBinDataset(Path(args.data_dir).expanduser() / "validation.bin", cfg.block_size)

    model = MiniGPT(cfg).to(device)
    optimizer = build_optimizer(args, model)
    if args.optimizer == "muon" and isinstance(optimizer, HybridMuon):
        write_json(run_dir / "muon_param_groups.json", optimizer.param_group_manifest())

    model = maybe_compile(model, args.compile)

    total, emb, non_emb = count_parameters(model)
    print(f"Parameters: total={total:,} embedding={emb:,} non_embedding={non_emb:,}")

    write_json(run_dir / "config.json", make_config_json(args, cfg, dataset_meta, model, optimizer, device))

    step = 0
    tokens_seen = 0
    best_val_loss: float | None = None
    last_val_loss: float | None = None

    if args.resume:
        ckpt_name = "latest.pt" if args.resume == "latest" else args.resume
        ckpt_path = checkpoints_dir / ckpt_name
        print(f"Resuming from {ckpt_path}")
        step, tokens_seen, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, device)
        append_jsonl(metrics_path, {"event": "resume", "step": step, "tokens_seen": tokens_seen, "checkpoint": str(ckpt_path), "time": utc_now()})

    # max_steps is required for cosine schedule. If user only sets max_minutes,
    # choose a large placeholder; run will stop by time.
    total_steps_for_schedule = args.max_steps if args.max_steps > 0 else 100_000
    start_time = time.time()
    last_log_time = time.time()
    last_log_tokens = tokens_seen

    write_run_summary(run_dir, "running", step, tokens_seen, started_at, best_val_loss, last_val_loss)

    # Ensure manifest is updated at process exit if Python exits normally.
    atexit.register(lambda: write_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir)))

    model.train()
    while True:
        if args.max_steps > 0 and step >= args.max_steps:
            status = "completed"
            break
        if args.max_minutes > 0 and (time.time() - start_time) / 60.0 >= args.max_minutes:
            status = "stopped_max_minutes"
            break
        if RUN_STATE.should_stop:
            status = "stopped_signal"
            break
        if (run_dir / "STOP").exists():
            RUN_STATE.should_stop = True
            RUN_STATE.stop_reason = "STOP file detected"
            status = "stopped_stop_file"
            break

        lr = cosine_lr(step, args.learning_rate, args.warmup_steps, total_steps_for_schedule, args.min_lr_ratio)
        set_optimizer_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        # Gradient accumulation: we do several forward/backward passes and only
        # update once. This simulates a larger batch without needing all examples
        # in memory at once.
        for micro in range(args.grad_accum_steps):
            x, y = train_data.get_batch(args.micro_batch_size, device)
            with autocast_context(device, args.precision):
                _, loss = model(x, y)
                # Divide by grad_accum_steps so accumulated gradient equals the
                # average gradient over the effective batch.
                scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            accum_loss += float(loss.detach().cpu())

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        step += 1
        step_tokens = args.micro_batch_size * args.grad_accum_steps * cfg.block_size
        tokens_seen += step_tokens
        train_loss = accum_loss / args.grad_accum_steps

        if step % args.log_every_steps == 0 or step == 1:
            now = time.time()
            elapsed = max(now - last_log_time, 1e-9)
            recent_tokens = tokens_seen - last_log_tokens
            tokens_per_second = recent_tokens / elapsed
            last_log_time = now
            last_log_tokens = tokens_seen
            estimated_flops = 6 * non_emb * tokens_seen

            event = {
                "event": "train_step",
                "time": utc_now(),
                "step": step,
                "tokens_seen": tokens_seen,
                "estimated_flops": estimated_flops,
                "train_loss": train_loss,
                "learning_rate": lr,
                "tokens_per_second": tokens_per_second,
                "wall_clock_seconds": now - start_time,
                **get_gpu_memory_stats(device),
            }
            append_jsonl(metrics_path, event)

            if isinstance(optimizer, HybridMuon):
                rms = optimizer.update_rms_summary()
                if rms is not None:
                    append_jsonl(metrics_path, {
                        "event": "muon_update_rms",
                        "time": utc_now(),
                        "step": step,
                        "tokens_seen": tokens_seen,
                        **rms,
                    })

            print(f"step {step:6d} loss {train_loss:.4f} lr {lr:.2e} tok/s {tokens_per_second:.0f}")

        if step % args.eval_every_steps == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, args.validation_batch_size, device, args.precision, args.validation_tokens)
            last_val_loss = val_loss
            best_val_loss = val_loss if best_val_loss is None else min(best_val_loss, val_loss)
            append_jsonl(metrics_path, {
                "event": "validation",
                "time": utc_now(),
                "step": step,
                "tokens_seen": tokens_seen,
                "estimated_flops": 6 * non_emb * tokens_seen,
                "val_loss": val_loss,
                "validation_tokens": args.validation_tokens,
            })
            print(f"validation step {step}: val_loss {val_loss:.4f}")

            # Generate short samples after validation for qualitative inspection.
            samples = generate_samples(
                model,
                tokenizer,
                prompts=["Once upon a time", "The little girl", "In a small village"],
                device=device,
                max_new_tokens=args.sample_new_tokens,
            )
            with (run_dir / "sample_generations.txt").open("a", encoding="utf-8") as f:
                f.write(f"\n===== step {step} =====\n{samples}\n")

        if step % args.save_every_steps == 0 or step == 1:
            latest = checkpoints_dir / "latest.pt"
            save_checkpoint(latest, model, optimizer, step, tokens_seen, cfg, args, best_val_loss)
            append_jsonl(metrics_path, {
                "event": "checkpoint",
                "time": utc_now(),
                "step": step,
                "tokens_seen": tokens_seen,
                "path": str(latest.relative_to(run_dir)),
                "bytes": file_size(latest),
            })
            write_run_summary(run_dir, "running", step, tokens_seen, started_at, best_val_loss, last_val_loss)
            plot_metrics(run_dir)

    # Final safe save on completion/stop.
    latest = checkpoints_dir / "latest.pt"
    save_checkpoint(latest, model, optimizer, step, tokens_seen, cfg, args, best_val_loss)
    append_jsonl(metrics_path, {
        "event": "safe_stop" if status.startswith("stopped") else "completed",
        "time": utc_now(),
        "step": step,
        "tokens_seen": tokens_seen,
        "reason": RUN_STATE.stop_reason or status,
        "checkpoint": str(latest.relative_to(run_dir)),
    })
    write_run_summary(run_dir, status, step, tokens_seen, started_at, best_val_loss, last_val_loss)
    plot_metrics(run_dir)
    print(f"Run ended with status={status}; latest checkpoint saved to {latest}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MiniGPT from scratch with AdamW or Hybrid Muon.")

    # Run identity / filesystem.
    p.add_argument("--run-id", default=None, help="Human-readable run id. Defaults to run directory name.")
    p.add_argument("--run-dir", required=True, help="Directory for config, metrics, checkpoints, plots.")
    p.add_argument("--data-dir", default="~/pretrain_data/tinystories_gpt2_512", help="Tokenized dataset cache directory.")
    p.add_argument("--resume", default=None, help="Resume checkpoint name, e.g. 'latest' or 'step_00010000.pt'.")

    # Dataset/tokenizer.
    p.add_argument("--dataset", choices=["toy", "tinystories", "finewebedu"], default="tinystories")
    p.add_argument("--tokenizer", default="openai-community/gpt2")
    p.add_argument("--fineweb-config", default="sample-10BT", help="FineWeb-Edu config used when --dataset finewebedu.")
    p.add_argument("--prepare-data", action="store_true", help="Force rebuild tokenized cache.")
    p.add_argument("--prepare-only", action="store_true", help="Prepare token cache and exit.")
    p.add_argument("--max-train-tokens", type=int, default=None, help="Optional cap while preparing data, useful for smoke tests.")
    p.add_argument("--max-val-tokens", type=int, default=None)

    # Model.
    p.add_argument("--model-config", choices=["tiny_debug", "minigpt_dense_60m_v1", "minigpt_dense_125m_v1"], default="minigpt_dense_60m_v1")
    p.add_argument("--block-size", type=int, default=None, help="Override model context length.")

    # Optimizer.
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--newton-schulz-steps", type=int, default=5)
    p.add_argument("--disable-muon-rms-scale", action="store_true")

    # Training schedule.
    p.add_argument("--seed", type=int, default=1337, help="Random seed for comparable/reproducible runs.")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--validation-batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0, help="0 means no step limit; use max-minutes instead.")
    p.add_argument("--max-minutes", type=float, default=10)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--compile", action="store_true", help="Use torch.compile; off by default for debuggability.")

    # Logging/eval/checkpoint cadence.
    p.add_argument("--log-every-steps", type=int, default=10)
    p.add_argument("--eval-every-steps", type=int, default=100)
    p.add_argument("--save-every-steps", type=int, default=200)
    p.add_argument("--validation-tokens", type=int, default=262_144)
    p.add_argument("--sample-new-tokens", type=int, default=80)

    args = p.parse_args(argv)
    if args.run_id is None:
        args.run_id = Path(args.run_dir).expanduser().name
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        train(args)
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM. Try reducing --micro-batch-size, then increase --grad-accum-steps if needed.", file=sys.stderr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    main()
