# %% [markdown]
# # Lab 03: Tiny Transformer From Scratch
#
# This is the first non-repetitive step: train a real tiny next-character model.
#
# It connects earlier concepts in one project:
#
# - token/embedding
# - attention q_proj/k_proj/v_proj/o_proj
# - MLP
# - LM head
# - loss curve
# - learned attention heatmap
# - next-token probability chart
# - parameter taxonomy: Muon candidates vs AdamW candidates
#
# Run cell-by-cell in VS Code with the `lean-ml visual` kernel.

# %%
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Reproducibility.
torch.manual_seed(7)
random.seed(7)

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
# ## 1. A tiny real dataset
#
# Character-level language modeling: given previous characters, predict the next character.
#
# This is tiny on purpose so you can understand every part.

# %%
sentences = [
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

# Repeat the corpus so batches have enough variation.
text = "\n".join(sentences) + "\n"
text = text * 300

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

print("dataset chars:", len(data))
print("vocab size:", vocab_size)
print("vocab:", "".join(chars))
print("sample text:\n", text[:250])

# %% [markdown]
# ## 2. Batch creation
#
# Input `x` is a sequence of characters.
# Target `y` is the same sequence shifted one step to the left.
#
# Example:
#
# ```text
# x = "the ca"
# y = "he cat"
# ```

# %%
block_size = 48
batch_size = 64

def get_batch(split: str = "train"):
    # Simple train/val split over the repeated text.
    n = int(0.9 * len(data))
    source = data[:n] if split == "train" else data[n:]
    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch()
print("x batch:", xb.shape)
print("y batch:", yb.shape)
print("first x example:", repr(decode(xb[0].cpu().tolist())))
print("first y example:", repr(decode(yb[0].cpu().tolist())))

# %% [markdown]
# ## 3. Model definition
#
# This is a tiny GPT-like Transformer.
#
# Important named matrices:
#
# - `q_proj.weight`
# - `k_proj.weight`
# - `v_proj.weight`
# - `o_proj.weight`
# - MLP weights
# - `lm_head.weight`

# %%
@dataclass
class TinyGPTConfig:
    vocab_size: int
    block_size: int = 48
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        if return_attn:
            return y, attn
        return y, None


class MLP(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.up_proj = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.down_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, return_attn: bool = False):
        attn_out, attn = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None, return_attn: bool = False):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        attentions = []
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn:
                attentions.append(attn)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, attentions

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

cfg = TinyGPTConfig(vocab_size=vocab_size, block_size=block_size)
model = TinyGPT(cfg).to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 4. Parameter taxonomy: what would Muon touch?
#
# This does not train with Muon yet. It only labels parameters.

# %%
def classify_param(name: str, p: torch.nn.Parameter) -> str:
    if p.ndim != 2:
        return "AdamW: scalar/vector/non-matrix"
    if "token_embedding" in name or "position_embedding" in name:
        return "AdamW: embedding"
    if "lm_head" in name:
        return "AdamW: LM head"
    if any(key in name for key in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "up_proj.weight", "down_proj.weight"]):
        return "Muon candidate: hidden matrix"
    return "AdamW: other"

for name, p in model.named_parameters():
    print(f"{name:45s} {tuple(p.shape)!s:16s} {classify_param(name, p)}")

# %% [markdown]
# ## 5. Train the tiny Transformer

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
max_steps = 99990
losses = []
val_losses = []

model.train()
for step in range(1, max_steps + 1):
    xb, yb = get_batch("train")
    logits, loss, _ = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

    if step % 50 == 0 or step == 1:
        model.eval()
        with torch.no_grad():
            vx, vy = get_batch("val")
            _, vloss, _ = model(vx, vy)
        val_losses.append((step, vloss.item()))
        print(f"step {step:4d} train loss {loss.item():.3f} val loss {vloss.item():.3f}")
        model.train()

plt.figure(figsize=(7, 3))
plt.plot(losses, label="train loss")
if val_losses:
    xs, ys = zip(*val_losses)
    plt.plot(xs, ys, marker="o", label="val loss")
plt.xlabel("training step")
plt.ylabel("cross-entropy loss")
plt.title("Tiny Transformer training curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_training_curve.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "03_training_curve.png")

# %% [markdown]
# ## 6. Generate text
#
# This shows the whole model working as a next-character predictor.

# %%
prompt = "the cat sat on the"
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=120, temperature=0.9)
print("PROMPT:", repr(prompt))
print("GENERATED:")
print(decode(out[0].detach().cpu().tolist()))

# %% [markdown]
# ## 7. LM head next-character probabilities
#
# This is the real learned version of the previous fake bar chart.

# %%
model.eval()
with torch.no_grad():
    idx = torch.tensor([encode(prompt[-block_size:])], dtype=torch.long, device=device)
    logits, _, _ = model(idx)
    probs = F.softmax(logits[0, -1], dim=-1).detach().cpu()

topk = torch.topk(probs, k=min(12, vocab_size))
labels = [repr(itos[i.item()]) for i in topk.indices]
values = topk.values.numpy()

plt.figure(figsize=(8, 3))
plt.bar(labels, values)
plt.ylabel("probability")
plt.title(f"LM head top next-character probabilities after {prompt!r}")
for i, p in enumerate(values):
    plt.text(i, p + 0.005, f"{p:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_lm_head_probs.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "03_lm_head_probs.png")

# %% [markdown]
# ## 8. Learned attention heatmap
#
# This is no longer hand-crafted. It comes from the trained model.
#
# Rows = query/listener character positions.
# Columns = key/value character positions being attended to.

# %%
attn_prompt = "the cat sat on the mat."
idx = torch.tensor([encode(attn_prompt[-block_size:])], dtype=torch.long, device=device)
with torch.no_grad():
    _, _, attentions = model(idx, return_attn=True)

# Last layer, first head, first batch item.
attn = attentions[-1][0, 0].detach().cpu()
labels = list(attn_prompt[-block_size:])

plt.figure(figsize=(9, 7))
plt.imshow(attn, cmap="magma")
plt.colorbar(label="attention strength")
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.xlabel("character being looked at / Key-Value position")
plt.ylabel("character doing the looking / Query position")
plt.title("Learned causal attention heatmap, last layer head 0")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_attention_heatmap.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "03_attention_heatmap.png")

# %% [markdown]
# ## 9. Weight matrix heatmaps
#
# These are actual learned hidden matrices.

# %%
q_weight = model.blocks[0].attn.q_proj.weight.detach().cpu()
mlp_weight = model.blocks[0].mlp.up_proj.weight.detach().cpu()

fig, axs = plt.subplots(1, 2, figsize=(9, 3))
axs[0].imshow(q_weight, aspect="auto", cmap="coolwarm")
axs[0].set_title("Layer 0 q_proj.weight")
axs[0].set_xlabel("input hidden dim")
axs[0].set_ylabel("output hidden dim")
axs[1].imshow(mlp_weight, aspect="auto", cmap="coolwarm")
axs[1].set_title("Layer 0 MLP up_proj.weight")
axs[1].set_xlabel("input hidden dim")
axs[1].set_ylabel("expanded dim")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_weight_heatmaps.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "03_weight_heatmaps.png")

# %% [markdown]
# ## Takeaway
#
# You now have one tiny trained model where the concepts are connected:
#
# - embeddings create hidden vectors
# - attention projections create Q/K/V/O
# - MLP processes each token internally
# - LM head predicts the next character
# - attention heatmap shows learned token/character dependencies
# - parameter taxonomy prepares us for Muon vs AdamW
#
# Next lab: either KV-cache timing demo or tiny MoE training demo.
