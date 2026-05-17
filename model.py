import torch
import torch.nn as nn

# ── Hyperparameters ──────────────────────────────────────
VOCAB_SIZE  = 4000
N_EMBED     = 256
N_HEADS     = 4
N_LAYERS    = 4
BLOCK_SIZE  = 128
DROPOUT     = 0.1

# ── 1. One attention head ────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril',
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.drop  = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        wei = self.drop(wei)
        v = self.value(x)
        return wei @ v, wei          # ← returns output AND weights

# ── 2. Multi-head attention ──────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(N_EMBED, N_EMBED)
        self.drop  = nn.Dropout(DROPOUT)
        self.attn_weights = None     # ← stored here after every forward

    def forward(self, x):
        head_outputs     = [h(x) for h in self.heads]
        out              = torch.cat([o for o, _ in head_outputs], dim=-1)
        self.attn_weights = torch.stack([w for _, w in head_outputs], dim=1)
        return self.drop(self.proj(out))

# ── 3. Feed-forward block ────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

# ── 4. One transformer block ─────────────────────────────
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBED // N_HEADS
        self.attn = MultiHeadAttention(N_HEADS, head_size)
        self.ff   = FeedForward()
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.ln2  = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ── 5. The full language model ───────────────────────────
class ArabicLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb     = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.pos_emb       = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks        = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_final      = nn.LayerNorm(N_EMBED)
        self.head          = nn.Linear(N_EMBED, VOCAB_SIZE)
        self.all_attn_weights = []   # ← filled every forward pass

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T))
        x   = tok + pos

        # run blocks one by one and collect weights
        self.all_attn_weights = []
        for block in self.blocks:
            x = block(x)
            self.all_attn_weights.append(block.attn.attn_weights)

        x      = self.ln_final(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(
                logits.view(B*T, C),
                targets.view(B*T)
            )
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_crop    = idx[:, -BLOCK_SIZE:]
            logits, _   = self(idx_crop)
            logits      = logits[:, -1, :]
            probs       = torch.softmax(logits, dim=-1)
            next_id     = torch.multinomial(probs, num_samples=1)
            idx         = torch.cat([idx, next_id], dim=1)
        return idx