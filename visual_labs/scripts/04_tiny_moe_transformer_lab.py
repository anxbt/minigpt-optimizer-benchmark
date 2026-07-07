# %% [markdown]
# # Lab 04: Tiny MoE Transformer
#
# This lab takes the tiny GPT idea and replaces the normal MLP with a
# **Mixture of Experts (MoE)** MLP.
#
# Goal:
#
# - train a real tiny MoE next-word model
# - see router probabilities
# - see expert load histograms
# - understand top-1 routing
# - connect MoE to the Muon paper's "expert matrices" and "router matrix"
#
# Run cell-by-cell in VS Code with the `lean-ml visual` kernel.

# %%
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(11)
random.seed(11)

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
# ## 1. Synthetic word-level dataset
#
# We use a tiny mixed-domain dataset so MoE routing has something to specialize on:
#
# - code-like sentences
# - math-like sentences
# - story-like sentences
# - translation/language-like sentences
#
# This is still a toy dataset, but less random than the previous hand-crafted router heatmap.

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

base_sentences = code_sentences + math_sentences + story_sentences + language_sentences
random.shuffle(base_sentences)

# Repeat many times for training.
all_sentences = base_sentences * 600
random.shuffle(all_sentences)

# Add BOS at sentence start.
tokenized = [["bos"] + s.split() for s in all_sentences]

vocab = sorted({tok for sent in tokenized for tok in sent})
stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}
vocab_size = len(vocab)

encoded_sentences = [torch.tensor([stoi[t] for t in sent], dtype=torch.long) for sent in tokenized]

print("num sentences:", len(encoded_sentences))
print("vocab size:", vocab_size)
print("vocab:", vocab)
print("example:", tokenized[0])

# %% [markdown]
# ## 2. Batch creation
#
# This is a next-token task:
#
# ```text
# x = bos code def quicksort sorts list
# y = code def quicksort sorts list eos
# ```

# %%
# Keep context long enough for both training examples and router probe examples.
block_size = max(max(len(s) for s in encoded_sentences) - 1, 12)
batch_size = 64
pad_id = stoi["eos"]  # harmless padding token for this toy setup


def get_batch():
    samples = random.choices(encoded_sentences, k=batch_size)
    xs, ys = [], []
    for s in samples:
        x = s[:-1]
        y = s[1:]
        # right-pad to block_size
        if len(x) < block_size:
            pad = torch.full((block_size - len(x),), pad_id, dtype=torch.long)
            x = torch.cat([x, pad])
            y = torch.cat([y, pad])
        xs.append(x[:block_size])
        ys.append(y[:block_size])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)

xb, yb = get_batch()
print("block_size:", block_size)
print("x batch:", xb.shape)
print("y batch:", yb.shape)
print("x example:", [itos[i] for i in xb[0].cpu().tolist()])
print("y example:", [itos[i] for i in yb[0].cpu().tolist()])

# %% [markdown]
# ## 3. Tiny MoE GPT definition
#
# The attention part is normal.
#
# The MLP part becomes:
#
# ```text
# token hidden state → router → choose expert → expert MLP
# ```

# %%
@dataclass
class MoEGPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    n_experts: int = 4
    dropout: float = 0.0
    balance_loss_weight: float = 0.01


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MoEGPTConfig):
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

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return (y, attn) if return_attn else (y, None)


class Expert(nn.Module):
    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.up_proj = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.down_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class Top1MoE(nn.Module):
    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])

    def forward(self, x, return_router=False):
        B, T, C = x.shape
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_expert = router_probs.argmax(dim=-1)  # [B, T]

        # Top-1 routing: each token goes to exactly one expert.
        y = torch.zeros_like(x)
        flat_x = x.reshape(B * T, C)
        flat_y = y.reshape(B * T, C)
        flat_expert = top_expert.reshape(B * T)

        for expert_id, expert in enumerate(self.experts):
            mask = flat_expert == expert_id
            if mask.any():
                flat_y[mask] = expert(flat_x[mask])

        y = flat_y.view(B, T, C)

        # Simple load-balancing auxiliary loss.
        # It discourages every token from collapsing to one expert.
        expert_fraction = F.one_hot(top_expert, num_classes=self.n_experts).float().mean(dim=(0, 1))
        prob_fraction = router_probs.mean(dim=(0, 1))
        balance_loss = self.n_experts * torch.sum(expert_fraction * prob_fraction)

        info = {
            "router_probs": router_probs,
            "top_expert": top_expert,
            "expert_fraction": expert_fraction,
            "prob_fraction": prob_fraction,
            "balance_loss": balance_loss,
        }
        return (y, info) if return_router else (y, {"balance_loss": balance_loss})


class Block(nn.Module):
    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.moe = Top1MoE(cfg)

    def forward(self, x, return_attn=False, return_router=False):
        attn_out, attn = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        moe_out, router_info = self.moe(self.ln_2(x), return_router=return_router)
        x = x + moe_out
        return x, attn, router_info


class TinyMoEGPT(nn.Module):
    def __init__(self, cfg: MoEGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None, return_attn=False, return_router=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        attentions = []
        router_infos = []
        balance_loss = torch.tensor(0.0, device=idx.device)
        for block in self.blocks:
            x, attn, router_info = block(x, return_attn=return_attn, return_router=return_router)
            balance_loss = balance_loss + router_info["balance_loss"]
            if return_attn:
                attentions.append(attn)
            if return_router:
                router_infos.append(router_info)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        ce_loss = None
        total_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss = ce_loss + self.cfg.balance_loss_weight * balance_loss
        return logits, total_loss, {"ce_loss": ce_loss, "balance_loss": balance_loss, "attentions": attentions, "router_infos": router_infos}

    @torch.no_grad()
    def generate(self, prompt_tokens: list[str], max_new_tokens=8, temperature=0.8):
        self.eval()
        idx = torch.tensor([[stoi[t] for t in prompt_tokens]], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == stoi["eos"]:
                break
        return [itos[i] for i in idx[0].detach().cpu().tolist()]

cfg = MoEGPTConfig(vocab_size=vocab_size, block_size=block_size, n_experts=4)
model = TinyMoEGPT(cfg).to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 4. Parameter taxonomy: MoE + Muon candidates

# %%
def classify_param(name: str, p: torch.nn.Parameter) -> str:
    if p.ndim != 2:
        return "AdamW: scalar/vector/non-matrix"
    if "embedding" in name:
        return "AdamW: embedding"
    if "lm_head" in name:
        return "AdamW: LM head"
    if ".router.weight" in name:
        return "Muon candidate: router matrix"
    if ".experts." in name and name.endswith("weight"):
        return "Muon candidate: expert matrix"
    if any(key in name for key in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]):
        return "Muon candidate: attention matrix"
    return "AdamW: other"

for name, p in model.named_parameters():
    print(f"{name:65s} {tuple(p.shape)!s:16s} {classify_param(name, p)}")

# %% [markdown]
# ## 5. Train

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
max_steps = 800
losses = []
ce_losses = []
balance_losses = []

model.train()
for step in range(1, max_steps + 1):
    xb, yb = get_batch()
    _, loss, info = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())
    ce_losses.append(info["ce_loss"].item())
    balance_losses.append(info["balance_loss"].item())

    if step == 1 or step % 100 == 0:
        print(f"step {step:4d} total {loss.item():.3f} ce {info['ce_loss'].item():.3f} balance {info['balance_loss'].item():.3f}")

plt.figure(figsize=(7, 3))
plt.plot(ce_losses, label="next-token CE loss")
plt.plot(losses, label="total loss", alpha=0.7)
plt.xlabel("training step")
plt.ylabel("loss")
plt.title("Tiny MoE Transformer training curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "04_moe_training_curve.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "04_moe_training_curve.png")

# %% [markdown]
# ## 6. Generate examples

# %%
for prompt in [["bos", "code"], ["bos", "math"], ["bos", "story"], ["bos", "language"]]:
    out = model.generate(prompt, max_new_tokens=8, temperature=0.7)
    print("PROMPT", prompt, "=>", " ".join(out))

# %% [markdown]
# ## 7. Router heatmap on interpretable prompts
#
# Now this router heatmap comes from a trained tiny MoE model.
# It may not perfectly match human categories, but it is a real learned router.

# %%
probe_tokens = ["bos", "code", "def", "quicksort", "math", "three", "plus", "story", "cat", "language", "bonjour", "french"]
probe_ids = torch.tensor([[stoi[t] for t in probe_tokens]], dtype=torch.long, device=device)

model.eval()
with torch.no_grad():
    _, _, info = model(probe_ids, return_router=True)

# Use last layer router for visualization.
router_info = info["router_infos"][-1]
router_probs = router_info["router_probs"][0].detach().cpu()
top_expert = router_info["top_expert"][0].detach().cpu()

plt.figure(figsize=(8, 5))
plt.imshow(router_probs, cmap="viridis", vmin=0, vmax=1, aspect="auto")
plt.colorbar(label="router probability")
plt.xticks(range(cfg.n_experts), [f"Expert {i}" for i in range(cfg.n_experts)])
plt.yticks(range(len(probe_tokens)), probe_tokens)
plt.xlabel("expert")
plt.ylabel("token")
plt.title("Learned MoE router probabilities, last layer")
for i in range(len(probe_tokens)):
    for j in range(cfg.n_experts):
        val = router_probs[i, j].item()
        color = "white" if val > 0.5 else "black"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_router_heatmap.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "04_router_heatmap.png")

for tok, eid in zip(probe_tokens, top_expert.tolist()):
    print(f"{tok:10s} -> Expert {eid}")

# %% [markdown]
# ## 8. Expert load histogram
#
# Count which expert gets selected for the probe tokens.

# %%
loads = torch.bincount(top_expert, minlength=cfg.n_experts).numpy()
plt.figure(figsize=(6, 3))
plt.bar([f"Expert {i}" for i in range(cfg.n_experts)], loads, color="tab:orange")
plt.ylabel("tokens routed")
plt.title("Expert load on probe tokens")
for i, v in enumerate(loads):
    plt.text(i, v + 0.05, str(int(v)), ha="center")
plt.ylim(0, max(loads) + 1)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_expert_load_probe.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "04_expert_load_probe.png")

# %% [markdown]
# ## 9. Expert load across a full random batch

# %%
xb, _ = get_batch()
with torch.no_grad():
    _, _, info = model(xb, return_router=True)
full_top = info["router_infos"][-1]["top_expert"].detach().cpu().reshape(-1)
full_loads = torch.bincount(full_top, minlength=cfg.n_experts).numpy()

plt.figure(figsize=(6, 3))
plt.bar([f"Expert {i}" for i in range(cfg.n_experts)], full_loads, color="tab:green")
plt.ylabel("tokens routed")
plt.title("Expert load across a random training batch")
for i, v in enumerate(full_loads):
    plt.text(i, v + 2, str(int(v)), ha="center")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_expert_load_batch.png", dpi=160)
plt.show()
print("saved", FIG_DIR / "04_expert_load_batch.png")

# %% [markdown]
# ## Takeaway
#
# A MoE layer adds two important learned components:
#
# ```text
# router.weight          = dispatcher matrix
# experts.*.up/down      = specialist MLP matrices
# ```
#
# In the Muon paper's language, these are matrix-shaped hidden weights and are
# plausible Muon candidates.
#
# The visual outputs to inspect are:
#
# - training curve
# - generated examples
# - router probability heatmap
# - expert load histograms
