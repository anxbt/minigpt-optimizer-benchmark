# %% [markdown]
# # Lab 05: AdamW vs Muon on Tiny GPT and Tiny MoE
#
# This lab connects our toy models to the Muon paper.
#
# We compare:
#
# 1. Tiny GPT + AdamW
# 2. Tiny GPT + Hybrid Muon
# 3. Tiny MoE GPT + AdamW
# 4. Tiny MoE GPT + Hybrid Muon
#
# Hybrid Muon means:
#
# - Muon for hidden matrix weights:
#   - attention q/k/v/o matrices
#   - MLP matrices
#   - MoE router matrix
#   - MoE expert matrices
# - AdamW for everything else:
#   - embeddings
#   - LM head
#   - norm weights
#   - biases
#   - scalars/vectors
#
# This is not a claim that tiny toy models prove the Muon paper. It is a controlled
# implementation and visualization lab.

# %%
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(42)
random.seed(42)

FIG_DIR = Path("visual_labs/reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("device:", device)
print("torch:", torch.__version__)

# %% [markdown]
# ## 1. Muon implementation
#
# Core recipe from the paper:
#
# ```text
# momentum = 0.95 * momentum + grad
# update = NewtonSchulz(momentum-ish update)
# update *= 0.2 * sqrt(max(matrix_shape))
# param *= 1 - lr * weight_decay
# param -= lr * update
# ```
#
# The important distinction:
#
# ```text
# AdamW = element-wise
# Muon  = matrix-aware
# ```

# %%
def zeropower_newton_schulz_5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate matrix orthogonalization used by Muon.

    Returns an update with the same shape as g.
    This small educational implementation uses fp32 for stability.
    """
    assert g.ndim == 2
    orig_dtype = g.dtype
    x = g.float()
    norm = x.norm()
    if not torch.isfinite(norm) or norm < eps:
        return torch.zeros_like(g)
    x = x / (norm + eps)

    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        b_x = xx_t @ x
        x = a * x + b * b_x + c * (xx_t @ b_x)

    if transposed:
        x = x.T
    return x.to(orig_dtype)


class HybridMuon:
    """Small educational hybrid optimizer.

    - Muon updates matrix params selected by `is_muon_param`.
    - AdamW updates all remaining params.
    """

    def __init__(
        self,
        named_params,
        is_muon_param: Callable[[str, torch.nn.Parameter], bool],
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.muon_params: list[tuple[str, torch.nn.Parameter]] = []
        adamw_params: list[torch.nn.Parameter] = []

        for name, p in named_params:
            if not p.requires_grad:
                continue
            if is_muon_param(name, p):
                self.muon_params.append((name, p))
            else:
                adamw_params.append(p)

        self.adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
        self.momentum_buffers = {id(p): torch.zeros_like(p) for _, p in self.muon_params}
        self.last_update_rms: dict[str, float] = {}

    def zero_grad(self, set_to_none: bool = True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        for _, p in self.muon_params:
            p.grad = None if set_to_none else torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        self.last_update_rms = {}
        self.adamw.step()

        for name, p in self.muon_params:
            if p.grad is None:
                continue
            if p.ndim != 2:
                raise ValueError(f"Muon param must be 2D, got {name} shape {tuple(p.shape)}")

            grad = p.grad
            buf = self.momentum_buffers[id(p)]
            buf.mul_(self.momentum).add_(grad)

            # Nesterov-style update used in common Muon implementations.
            momentum_update = grad + self.momentum * buf
            ortho = zeropower_newton_schulz_5(momentum_update, steps=self.ns_steps)

            rows, cols = p.shape
            scale = 0.2 * math.sqrt(max(rows, cols))
            update = ortho * scale
            self.last_update_rms[name] = update.float().pow(2).mean().sqrt().item()

            # Decoupled weight decay, AdamW-style.
            if self.weight_decay != 0:
                p.mul_(1 - self.lr * self.weight_decay)
            p.add_(update, alpha=-self.lr)

    def param_summary(self) -> tuple[int, int]:
        return len(self.muon_params), sum(p.numel() for _, p in self.muon_params)


def avg_update_rms(opt) -> float | None:
    if not isinstance(opt, HybridMuon) or not opt.last_update_rms:
        return None
    vals = list(opt.last_update_rms.values())
    return sum(vals) / len(vals)

# %% [markdown]
# ## 2. Tiny GPT setup

# %%
char_sentences = [
    "the cat sat on the mat.",
    "the dog ran in the park.",
    "the bird flew over the tree.",
    "a cat chased a mouse.",
    "a dog chased a ball.",
    "the child read a story.",
    "the teacher wrote code.",
    "python code sorts a list.",
    "quick sort splits the list.",
    "three plus five equals eight.",
    "two times three equals six.",
    "bonjour means hello in french.",
    "hola means hello in spanish.",
]
char_text = ("\n".join(char_sentences) + "\n") * 300
chars = sorted(set(char_text))
char_stoi = {ch: i for i, ch in enumerate(chars)}
char_itos = {i: ch for ch, i in char_stoi.items()}
char_data = torch.tensor([char_stoi[c] for c in char_text], dtype=torch.long)


def char_decode(ids):
    return "".join(char_itos[i] for i in ids)


@dataclass
class TinyGPTConfig:
    vocab_size: int
    block_size: int = 48
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class GPTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_proj = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.down_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class GPTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = GPTMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=80, temperature=0.8):
        self.eval()
        idx = torch.tensor([[char_stoi[c] for c in prompt]], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return char_decode(idx[0].detach().cpu().tolist())


def get_char_batch(batch_size=64, block_size=48):
    n = int(0.9 * len(char_data))
    source = char_data[:n]
    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def is_gpt_muon_param(name, p):
    if p.ndim != 2:
        return False
    if "embedding" in name or "lm_head" in name:
        return False
    return any(k in name for k in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "up_proj.weight", "down_proj.weight"])

# %% [markdown]
# ## 3. Tiny MoE setup

# %%
code_sentences = [
    "code def quicksort sorts list eos",
    "code python function returns value eos",
    "code class object stores data eos",
    "code loop iterates over array eos",
    "code import torch creates tensor eos",
]
math_sentences = [
    "math three plus five equals eight eos",
    "math two times three equals six eos",
    "math ten minus four equals six eos",
    "math square root of nine equals three eos",
    "math matrix times vector gives vector eos",
]
story_sentences = [
    "story the cat sat on mat eos",
    "story the dog ran in park eos",
    "story the bird flew over tree eos",
    "story the child read book eos",
    "story the mouse ate cheese eos",
]
language_sentences = [
    "language bonjour means hello in french eos",
    "language hola means hello in spanish eos",
    "language gracias means thanks in spanish eos",
    "language chat means cat in french eos",
    "language guten tag means hello in german eos",
]
word_sentences = code_sentences + math_sentences + story_sentences + language_sentences
word_tokenized = [["bos"] + s.split() for s in (word_sentences * 600)]
word_vocab = sorted({tok for sent in word_tokenized for tok in sent})
word_stoi = {tok: i for i, tok in enumerate(word_vocab)}
word_itos = {i: tok for tok, i in word_stoi.items()}
word_encoded = [torch.tensor([word_stoi[t] for t in sent], dtype=torch.long) for sent in word_tokenized]
word_block_size = max(max(len(s) for s in word_encoded) - 1, 12)
word_pad_id = word_stoi["eos"]


@dataclass
class MoEGPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    n_experts: int = 4
    balance_loss_weight: float = 0.01


class Expert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up_proj = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.down_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class Top1MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        router_probs = F.softmax(self.router(x), dim=-1)
        top_expert = router_probs.argmax(dim=-1)
        y = torch.zeros_like(x)
        flat_x = x.reshape(B * T, C)
        flat_y = y.reshape(B * T, C)
        flat_expert = top_expert.reshape(B * T)
        for expert_id, expert in enumerate(self.experts):
            mask = flat_expert == expert_id
            if mask.any():
                flat_y[mask] = expert(flat_x[mask])
        y = flat_y.view(B, T, C)
        expert_fraction = F.one_hot(top_expert, num_classes=self.n_experts).float().mean(dim=(0, 1))
        prob_fraction = router_probs.mean(dim=(0, 1))
        balance_loss = self.n_experts * torch.sum(expert_fraction * prob_fraction)
        return y, balance_loss, top_expert


class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.moe = Top1MoE(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        moe_out, balance_loss, top_expert = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, balance_loss, top_expert


class TinyMoEGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        balance = torch.tensor(0.0, device=idx.device)
        top_experts = []
        for block in self.blocks:
            x, b, top = block(x)
            balance = balance + b
            top_experts.append(top)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        ce_loss = None
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + self.cfg.balance_loss_weight * balance
        return logits, loss, {"ce_loss": ce_loss, "balance_loss": balance, "top_experts": top_experts}

    @torch.no_grad()
    def generate(self, prompt_tokens: list[str], max_new_tokens=8, temperature=0.8):
        self.eval()
        idx = torch.tensor([[word_stoi[t] for t in prompt_tokens]], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx[:, -self.cfg.block_size :])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == word_stoi["eos"]:
                break
        return " ".join(word_itos[i] for i in idx[0].detach().cpu().tolist())


def get_word_batch(batch_size=64):
    samples = random.choices(word_encoded, k=batch_size)
    xs, ys = [], []
    for s in samples:
        x, y = s[:-1], s[1:]
        if len(x) < word_block_size:
            pad = torch.full((word_block_size - len(x),), word_pad_id, dtype=torch.long)
            x = torch.cat([x, pad])
            y = torch.cat([y, pad])
        xs.append(x[:word_block_size])
        ys.append(y[:word_block_size])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def is_moe_muon_param(name, p):
    if p.ndim != 2:
        return False
    if "embedding" in name or "lm_head" in name:
        return False
    if ".router.weight" in name:
        return True
    if ".experts." in name and name.endswith("weight"):
        return True
    return any(k in name for k in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"])

# %% [markdown]
# ## 4. Shared training/evaluation helpers

# %%
def make_optimizer(kind: str, model: nn.Module, is_muon_param: Callable, lr=3e-3, weight_decay=0.01):
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if kind == "muon":
        opt = HybridMuon(model.named_parameters(), is_muon_param, lr=lr, weight_decay=weight_decay)
        n_params, n_values = opt.param_summary()
        print(f"Hybrid Muon selected {n_params} matrix params / {n_values:,} values")
        return opt
    raise ValueError(kind)


def train_run(label, model, optimizer, get_batch_fn, steps=300, print_every=100):
    model.train()
    losses = []
    ce_losses = []
    update_rms = []
    for step in range(1, steps + 1):
        xb, yb = get_batch_fn()
        _, loss, *rest = model(xb, yb)
        # TinyGPT returns (logits, loss). TinyMoE returns (logits, loss, info).
        info = rest[0] if rest else {}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        ce_losses.append(info.get("ce_loss", loss).item() if isinstance(info, dict) and info.get("ce_loss") is not None else loss.item())
        update_rms.append(avg_update_rms(optimizer))
        if step == 1 or step % print_every == 0:
            msg = f"{label:16s} step {step:4d} loss {loss.item():.3f}"
            if update_rms[-1] is not None:
                msg += f" avg_muon_update_rms {update_rms[-1]:.3f}"
            print(msg)
    return {"loss": losses, "ce_loss": ce_losses, "update_rms": update_rms}


def clone_with_same_init(make_model_fn):
    torch.manual_seed(123)
    base = make_model_fn().to(device)
    model_a = copy.deepcopy(base).to(device)
    model_b = copy.deepcopy(base).to(device)
    return model_a, model_b

# %% [markdown]
# ## 5. Tiny GPT: AdamW vs Hybrid Muon

# %%
gpt_cfg = TinyGPTConfig(vocab_size=len(chars))

def make_gpt():
    return TinyGPT(gpt_cfg)

gpt_adamw, gpt_muon = clone_with_same_init(make_gpt)
print("TinyGPT parameters:", sum(p.numel() for p in gpt_adamw.parameters()))

gpt_adamw_opt = make_optimizer("adamw", gpt_adamw, is_gpt_muon_param, lr=3e-3, weight_decay=0.01)
gpt_muon_opt = make_optimizer("muon", gpt_muon, is_gpt_muon_param, lr=3e-3, weight_decay=0.01)

gpt_adamw_hist = train_run("GPT AdamW", gpt_adamw, gpt_adamw_opt, get_char_batch, steps=350)
gpt_muon_hist = train_run("GPT Muon", gpt_muon, gpt_muon_opt, get_char_batch, steps=350)

plt.figure(figsize=(8, 3))
plt.plot(gpt_adamw_hist["loss"], label="Tiny GPT AdamW")
plt.plot(gpt_muon_hist["loss"], label="Tiny GPT Hybrid Muon")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Tiny GPT: AdamW vs Hybrid Muon")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "05_tiny_gpt_adamw_vs_muon_loss.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "05_tiny_gpt_adamw_vs_muon_loss.png")

print("\nGPT AdamW sample:")
print(gpt_adamw.generate("the ", temperature=0.8))
print("\nGPT Muon sample:")
print(gpt_muon.generate("the ", temperature=0.8))

# %% [markdown]
# ## 6. Tiny MoE: AdamW vs Hybrid Muon

# %%
moe_cfg = MoEGPTConfig(vocab_size=len(word_vocab), block_size=word_block_size)

def make_moe():
    return TinyMoEGPT(moe_cfg)

moe_adamw, moe_muon = clone_with_same_init(make_moe)
print("TinyMoE parameters:", sum(p.numel() for p in moe_adamw.parameters()))

moe_adamw_opt = make_optimizer("adamw", moe_adamw, is_moe_muon_param, lr=3e-3, weight_decay=0.01)
moe_muon_opt = make_optimizer("muon", moe_muon, is_moe_muon_param, lr=3e-3, weight_decay=0.01)

moe_adamw_hist = train_run("MoE AdamW", moe_adamw, moe_adamw_opt, get_word_batch, steps=500)
moe_muon_hist = train_run("MoE Muon", moe_muon, moe_muon_opt, get_word_batch, steps=500)

plt.figure(figsize=(8, 3))
plt.plot(moe_adamw_hist["ce_loss"], label="Tiny MoE AdamW CE")
plt.plot(moe_muon_hist["ce_loss"], label="Tiny MoE Hybrid Muon CE")
plt.xlabel("step")
plt.ylabel("next-token CE loss")
plt.title("Tiny MoE: AdamW vs Hybrid Muon")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "05_tiny_moe_adamw_vs_muon_loss.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "05_tiny_moe_adamw_vs_muon_loss.png")

for prompt in [["bos", "code"], ["bos", "math"], ["bos", "story"], ["bos", "language"]]:
    print("\nPROMPT", prompt)
    print("AdamW:", moe_adamw.generate(prompt, temperature=0.7))
    print("Muon :", moe_muon.generate(prompt, temperature=0.7))

# %% [markdown]
# ## 7. Muon update RMS
#
# The paper's core practical fix is to scale Muon updates so matrix update RMS
# is in a useful range. We log the average Muon update RMS across selected matrices.

# %%
plt.figure(figsize=(8, 3))
plt.plot([x if x is not None else float("nan") for x in gpt_muon_hist["update_rms"]], label="Tiny GPT Muon avg update RMS")
plt.plot([x if x is not None else float("nan") for x in moe_muon_hist["update_rms"]], label="Tiny MoE Muon avg update RMS")
plt.xlabel("step")
plt.ylabel("average selected-matrix update RMS")
plt.title("Hybrid Muon update RMS")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "05_muon_update_rms.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "05_muon_update_rms.png")

# %% [markdown]
# ## Takeaway
#
# This lab gives us the paper bridge:
#
# ```text
# AdamW baseline
# vs
# Hybrid Muon = Muon for hidden matrices + AdamW for the rest
# ```
#
# What to inspect:
#
# - Did both optimizers train?
# - Which one converged faster on the toy task?
# - Are Muon update RMS values stable?
# - Does MoE still route/generate sensible examples?
#
# Next natural lab:
#
# ```text
# 06_svd_entropy_visualization.py
# ```
#
# There we compare singular values and SVD entropy of AdamW-trained vs Muon-trained matrices.

# %%
