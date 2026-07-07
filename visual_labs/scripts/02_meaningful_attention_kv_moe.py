# %% [markdown]
# # Meaningful Visual Examples: Attention, KV Cache, LM Head, and MoE Router
#
# The first lab used random weights, so the pictures looked random.
# This lab uses hand-crafted examples so the images map to real intuitions:
#
# - attention: "it" should look at "cat"
# - KV cache: old keys/values are stored during generation
# - LM head: "The cat sat on the" should predict "mat"
# - MoE router: code-like tokens go to Code expert, math-like tokens to Math expert

# %%
import math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Attention heatmap with real token labels
# Sentence: "The cat sat because it was tired"
#
# We manually create an attention pattern where the token "it" strongly attends to "cat".
# This is what pronoun resolution looks like in attention-map intuition.

# %%
tokens = ["The", "cat", "sat", "because", "it", "was", "tired"]
n = len(tokens)

# Start with low attention everywhere.
attn = np.ones((n, n)) * 0.03

# Each token mostly attends to itself by default.
for i in range(n):
    attn[i, i] += 0.35

# Causal/local-ish dependencies.
attn[2, 1] += 0.35  # "sat" looks at "cat"
attn[3, 2] += 0.25  # "because" looks at "sat"
attn[4, 1] += 0.75  # "it" strongly looks at "cat"
attn[5, 4] += 0.35  # "was" looks at "it"
attn[6, 4] += 0.25  # "tired" looks at "it"
attn[6, 1] += 0.25  # "tired" also relates to "cat"

# Normalize rows so each row is a probability distribution.
attn = attn / attn.sum(axis=1, keepdims=True)

plt.figure(figsize=(8, 6))
plt.imshow(attn, cmap="magma", vmin=0, vmax=attn.max())
plt.colorbar(label="attention strength")
plt.xticks(range(n), tokens, rotation=45, ha="right")
plt.yticks(range(n), tokens)
plt.xlabel("token being looked at / Key-Value token")
plt.ylabel("token doing the looking / Query token")
plt.title("Attention heatmap: 'it' attends strongly to 'cat'")

# Annotate cells.
for i in range(n):
    for j in range(n):
        if attn[i, j] > 0.18:
            plt.text(j, i, f"{attn[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
plt.tight_layout()
plt.show()

print("Read this as: row token asks a question; column token may answer.")
print("Strong example: row='it', column='cat' means 'it' looks back at 'cat'.")

# %% [markdown]
# ## 2. Query, Key, Value as a library lookup
#
# Query asks: what am I looking for?
# Key says: what do I contain?
# Value gives: information to retrieve if selected.

# %%
qkv_cards = {
    "cat": {
        "Query": "What words around me describe what I do?",
        "Key": "I am an animal / noun / entity.",
        "Value": "cat = animal, singular, subject candidate",
    },
    "it": {
        "Query": "Which previous noun/entity do I refer to?",
        "Key": "I am a pronoun.",
        "Value": "it = pronoun needing referent",
    },
}

for tok, cards in qkv_cards.items():
    print(f"\nToken: {tok}")
    for name, text in cards.items():
        print(f"  {name:5s}: {text}")

# %% [markdown]
# ## 3. KV cache growth during generation
# Prompt/generation: "The cat sat on the mat"
#
# At each new token, we store K and V for that token.
# Old K/V do not need to be recomputed.

# %%
gen_tokens = ["The", "cat", "sat", "on", "the", "mat"]
cache_sizes = []
cache = []

for t, tok in enumerate(gen_tokens, start=1):
    cache.append((f"K_{tok}", f"V_{tok}"))
    cache_sizes.append(len(cache))
    print(f"Step {t}: generated/processed '{tok}'")
    print("  new compute: Q, K, V for only this token")
    print("  cache now:", cache)

plt.figure(figsize=(7, 3))
plt.plot(range(1, len(gen_tokens) + 1), cache_sizes, marker="o")
plt.xticks(range(1, len(gen_tokens) + 1), gen_tokens)
plt.xlabel("generation step / new token")
plt.ylabel("cached KV entries")
plt.title("KV cache grows one K/V pair per token per layer")
plt.grid(True, alpha=0.3)
plt.show()

print("KV cache saves compute but uses memory. Long context = large KV cache.")

# %% [markdown]
# ## 4. LM head probability example
# Prompt: "The cat sat on the"
#
# The LM head converts the final hidden state into probabilities over vocabulary tokens.

# %%
vocab = ["mat", "floor", "chair", "dog", "banana", ".", "roof", "keyboard"]
probs = np.array([0.52, 0.18, 0.08, 0.03, 0.005, 0.09, 0.06, 0.035])
probs = probs / probs.sum()

plt.figure(figsize=(8, 3))
colors = ["tab:green" if tok == "mat" else "tab:blue" for tok in vocab]
plt.bar(vocab, probs, color=colors)
plt.ylabel("probability")
plt.title("LM head: next-token probabilities after 'The cat sat on the'")
plt.ylim(0, max(probs) * 1.25)
for i, p in enumerate(probs):
    plt.text(i, p + 0.01, f"{p:.2f}", ha="center")
plt.show()

print("The tallest bar is the model's chosen next-token candidate.")

# %% [markdown]
# ## 5. MoE router heatmap with meaningful expert labels
#
# Router = dispatcher.
# It sends each token to one or more expert departments.

# %%
moe_tokens = ["def", "quicksort", "3x", "+", "cat", "story", "translate", "bonjour"]
experts = ["Code", "Math", "Language", "General"]

# Rows = tokens, columns = experts.
router_probs = np.array([
    [0.86, 0.03, 0.05, 0.06],  # def -> Code
    [0.78, 0.08, 0.06, 0.08],  # quicksort -> Code
    [0.04, 0.84, 0.04, 0.08],  # 3x -> Math
    [0.05, 0.76, 0.04, 0.15],  # + -> Math
    [0.04, 0.03, 0.65, 0.28],  # cat -> Language
    [0.05, 0.03, 0.74, 0.18],  # story -> Language
    [0.05, 0.03, 0.82, 0.10],  # translate -> Language
    [0.03, 0.02, 0.85, 0.10],  # bonjour -> Language
])

top_expert = router_probs.argmax(axis=1)

plt.figure(figsize=(8, 5))
plt.imshow(router_probs, cmap="viridis", vmin=0, vmax=1)
plt.colorbar(label="router probability")
plt.xticks(range(len(experts)), experts)
plt.yticks(range(len(moe_tokens)), moe_tokens)
plt.xlabel("expert department")
plt.ylabel("token")
plt.title("MoE router heatmap: dispatcher chooses experts")

for i in range(len(moe_tokens)):
    for j in range(len(experts)):
        color = "white" if router_probs[i, j] > 0.5 else "black"
        plt.text(j, i, f"{router_probs[i,j]:.2f}", ha="center", va="center", color=color, fontsize=8)
plt.tight_layout()
plt.show()

for tok, eid in zip(moe_tokens, top_expert):
    print(f"{tok:10s} -> {experts[eid]}")

# %% [markdown]
# ## 6. Expert load histogram
#
# This shows whether experts are balanced or one expert is overloaded.

# %%
loads = np.bincount(top_expert, minlength=len(experts))

plt.figure(figsize=(6, 3))
plt.bar(experts, loads, color="tab:orange")
plt.ylabel("number of tokens routed")
plt.title("Expert load histogram")
for i, v in enumerate(loads):
    plt.text(i, v + 0.05, str(v), ha="center")
plt.ylim(0, max(loads) + 1)
plt.show()

print("Balanced experts = all departments used.")
print("Router collapse = one expert gets almost everything.")

# %% [markdown]
# ## Final takeaway
#
# - Attention heatmap: who listens to whom?
# - Q/K/V: query asks, key matches, value answers.
# - KV cache: saved K/V cards from previous tokens.
# - LM head: final next-token probability chart.
# - MoE router: dispatcher selecting expert departments.
# - Expert load: workload balance across departments.

# %%
