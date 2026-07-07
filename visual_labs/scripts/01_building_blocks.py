# %% [markdown]
# # Visual Transformer Building Blocks
#
# Run this file cell-by-cell in VS Code using the Python/Jupyter extension.
# It teaches:
# - tensors: scalar/vector/matrix
# - MLP
# - LM head
# - attention projections Q/K/V/O
# - MoE router
# - biases and scalars
# - RNN recurrence

# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(0)
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Scalars, vectors, matrices, tensors

# %%
scalar = torch.tensor(3.0)
vector = torch.randn(5)
matrix = torch.randn(4, 5)
tensor3 = torch.randn(2, 3, 4)

for name, x in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("tensor3", tensor3)]:
    print(name, "shape =", tuple(x.shape), "ndim =", x.ndim)

plt.figure(figsize=(5, 3))
plt.imshow(matrix, cmap="coolwarm")
plt.colorbar()
plt.title("A matrix is a 2D table of numbers")
plt.show()

# %% [markdown]
# ## 2. Linear layer and bias
# A linear layer does: y = x @ W.T + b

# %%
linear = nn.Linear(3, 5)  # input dim 3, output dim 5
x = torch.randn(4, 3)     # batch of 4 examples

y = linear(x)
y_manual = x @ linear.weight.T + linear.bias

print("x:", x.shape)
print("weight:", linear.weight.shape, "<- matrix")
print("bias:", linear.bias.shape, "<- vector")
print("y:", y.shape)
print("manual matches nn.Linear:", torch.allclose(y, y_manual))

plt.figure(figsize=(5, 3))
plt.imshow(linear.weight.detach(), cmap="viridis")
plt.colorbar()
plt.title("Linear layer weight matrix")
plt.xlabel("input dimension")
plt.ylabel("output dimension")
plt.show()

# %% [markdown]
# ## 3. MLP
# An MLP expands hidden dimension, applies nonlinearity, then projects back.

# %%
hidden = 8
intermediate = 32
mlp = nn.Sequential(
    nn.Linear(hidden, intermediate),
    nn.GELU(),
    nn.Linear(intermediate, hidden),
)

x = torch.randn(2, 4, hidden)  # batch=2, tokens=4, hidden=8
y = mlp(x)
print("input:", x.shape)
print("output:", y.shape)
print("first MLP matrix:", mlp[0].weight.shape)
print("second MLP matrix:", mlp[2].weight.shape)

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
axs[0].imshow(mlp[0].weight.detach(), aspect="auto", cmap="coolwarm")
axs[0].set_title("MLP up matrix")
axs[1].imshow(mlp[2].weight.detach(), aspect="auto", cmap="coolwarm")
axs[1].set_title("MLP down matrix")
plt.show()

# %% [markdown]
# ## 4. LM head
# The LM head converts hidden states into vocabulary logits.

# %%
batch, tokens, hidden, vocab = 2, 4, 8, 20
hidden_states = torch.randn(batch, tokens, hidden)
lm_head = nn.Linear(hidden, vocab, bias=False)
logits = lm_head(hidden_states)
probs = F.softmax(logits, dim=-1)

print("hidden states:", hidden_states.shape)
print("lm_head.weight:", lm_head.weight.shape)
print("logits:", logits.shape)
print("probabilities sum to 1:", probs[0, 0].sum().item())
print("top token id at position [0,0]:", probs[0, 0].argmax().item())

plt.figure(figsize=(8, 3))
plt.bar(range(vocab), probs[0, 0].detach())
plt.title("LM head probability distribution for one token position")
plt.xlabel("vocab token id")
plt.ylabel("probability")
plt.show()

# %% [markdown]
# ## 5. Attention projections: Q, K, V, O
# Q = what am I looking for?
# K = what do I contain?
# V = what information do I pass?

# %%
batch, tokens, hidden = 1, 6, 8
x = torch.randn(batch, tokens, hidden)

q_proj = nn.Linear(hidden, hidden, bias=False)
k_proj = nn.Linear(hidden, hidden, bias=False)
v_proj = nn.Linear(hidden, hidden, bias=False)
o_proj = nn.Linear(hidden, hidden, bias=False)

Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)

scores = Q @ K.transpose(-2, -1) / math.sqrt(hidden)
attn = F.softmax(scores, dim=-1)
context = attn @ V
out = o_proj(context)

print("x:", x.shape)
print("Q/K/V:", Q.shape, K.shape, V.shape)
print("attention scores:", scores.shape)
print("attention weights:", attn.shape)
print("output:", out.shape)

plt.figure(figsize=(5, 4))
plt.imshow(attn[0].detach(), cmap="magma")
plt.colorbar()
plt.title("Attention heatmap: rows attend to columns")
plt.xlabel("key/value token position")
plt.ylabel("query token position")
plt.show()

# %% [markdown]
# ## 6. MoE router and experts
# Router chooses which expert processes each token.

# %%
class Expert(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden),
        )
    def forward(self, x):
        return self.net(x)

hidden = 8
num_experts = 4
experts = nn.ModuleList([Expert(hidden, 16) for _ in range(num_experts)])
router = nn.Linear(hidden, num_experts, bias=False)

x = torch.randn(1, 12, hidden)  # 12 tokens
router_logits = router(x)
router_probs = F.softmax(router_logits, dim=-1)
top_expert = router_probs.argmax(dim=-1)

print("router.weight:", router.weight.shape)
print("router_probs:", router_probs.shape)
print("top expert per token:", top_expert[0].tolist())

counts = torch.bincount(top_expert.flatten(), minlength=num_experts)
plt.figure(figsize=(5, 3))
plt.bar(range(num_experts), counts.detach())
plt.title("MoE expert load: tokens routed to each expert")
plt.xlabel("expert id")
plt.ylabel("token count")
plt.show()

plt.figure(figsize=(6, 4))
plt.imshow(router_probs[0].detach(), aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("Router probabilities: token x expert")
plt.xlabel("expert id")
plt.ylabel("token position")
plt.show()

# Simple top-1 MoE forward pass for visualization.
outputs = torch.zeros_like(x)
for expert_id, expert in enumerate(experts):
    mask = top_expert == expert_id
    if mask.any():
        outputs[mask] = expert(x[mask])
print("MoE output:", outputs.shape)

# %% [markdown]
# ## 7. RNN
# RNNs update hidden state sequentially: h_t = tanh(x_t W_x + h_{t-1} W_h + b)

# %%
seq_len, input_dim, hidden_dim = 10, 3, 5
x_seq = torch.randn(seq_len, input_dim)
W_x = torch.randn(input_dim, hidden_dim) * 0.5
W_h = torch.randn(hidden_dim, hidden_dim) * 0.5
b = torch.zeros(hidden_dim)

h = torch.zeros(hidden_dim)
history = []
for t in range(seq_len):
    h = torch.tanh(x_seq[t] @ W_x + h @ W_h + b)
    history.append(h.clone())

history = torch.stack(history)
print("RNN hidden history:", history.shape)

plt.figure(figsize=(7, 3))
plt.imshow(history.detach().T, aspect="auto", cmap="coolwarm")
plt.colorbar()
plt.title("RNN hidden state over time")
plt.xlabel("time step")
plt.ylabel("hidden unit")
plt.show()

# %% [markdown]
# ## Takeaway
# - MLP weights, attention projections, router weights, and expert weights are matrices.
# - Biases and norm weights are usually vectors.
# - Scalars are single learned numbers.
# - LM head is a matrix, but many Muon-style recipes keep it under AdamW.
# - RNNs are sequence models that reuse the same matrices over time.
