import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import sentencepiece as spm
import re
from model import ArabicLM, BLOCK_SIZE

# ── Load everything ──────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/arabic.model")

model = ArabicLM()
model.load_state_dict(torch.load("model/arabic_lm.pt",
                     map_location="cpu"))
model.eval()

# ── Clean prompt (same as training) ──────────────────────
def clean(text):
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'[^؀-ۿs]', ' ', text)
    return re.sub(r's+', ' ', text).strip()

# ── Run prompt through model, collect attention ───────────
def get_attention(prompt, max_new=20):
    ids  = sp.encode(clean(prompt))
    idx  = torch.tensor([ids], dtype=torch.long)

    # generate tokens one at a time
    with torch.no_grad():
        for _ in range(max_new):
            crop    = idx[:, -BLOCK_SIZE:]
            logits, _ = model(crop)
            logits  = logits[:, -1, :] / 0.9
            for tid in set(idx[0].tolist()[-20:]):
                logits[0, tid] /= 1.8
            probs   = torch.softmax(logits, dim=-1)
            nxt     = torch.multinomial(probs, 1)
            idx     = torch.cat([idx, nxt], dim=1)

    # final forward pass to get attention for the full sequence
    final_ids = idx[:, -BLOCK_SIZE:]
    with torch.no_grad():
        model(final_ids)

    weights = model.all_attn_weights  # list of 4 tensors
    tokens  = sp.encode(sp.decode(final_ids[0].tolist()), out_type=str)
    return weights, tokens

# ── Draw the heatmaps ────────────────────────────────────
def plot_attention(prompt, max_new=20, max_tokens=24):
    weights, tokens = get_attention(prompt, max_new)
    tokens = tokens[:max_tokens]
    n_layers = len(weights)

    fig, axes = plt.subplots(1, n_layers,
                             figsize=(5 * n_layers, 5))
    fig.suptitle(f'Attention maps — "{prompt}"',
                 fontsize=13, y=1.02)

    for layer_idx, w in enumerate(weights):
        # w shape: (1, n_heads, T, T)
        # average across heads → (T, T)
        avg = w[0].mean(dim=0).cpu().numpy()
        avg = avg[:max_tokens, :max_tokens]

        ax = axes[layer_idx]
        sns.heatmap(
            avg, ax=ax,
            cmap="YlOrRd",
            xticklabels=tokens,
            yticklabels=tokens,
            cbar=layer_idx == n_layers - 1,
            vmin=0, vmax=avg.max()
        )
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=11)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)
        ax.set_xlabel("Attended to", fontsize=9)
        ax.set_ylabel("Query token",  fontsize=9)

    plt.tight_layout()
    out_path = "model/attention_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")

# ── Run it ───────────────────────────────────────────────
matplotlib.use("TkAgg")
plot_attention("السلام علي", max_new=20)