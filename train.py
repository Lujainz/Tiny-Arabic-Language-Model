import torch
from model import ArabicLM, BLOCK_SIZE

# ── Load data from Phase 1 ───────────────────────────────
train_data = torch.load("data/train.pt")
val_data   = torch.load("data/val.pt")
print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens:   {len(val_data):,}")

# ── Training settings ────────────────────────────────────
BATCH_SIZE  = 32    # how many samples per step
MAX_STEPS   = 5000  # total training steps
EVAL_EVERY  = 500   # print loss every N steps
LR          = 1e-3  # learning rate

device = "cpu"      # Windows laptop, no GPU

# ── Helper: grab a random batch of training samples ──────
def get_batch(split):
    data = train_data if split == "train" else val_data
    # pick BATCH_SIZE random starting positions
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([data[i   : i+BLOCK_SIZE  ] for i in ix])
    y  = torch.stack([data[i+1 : i+BLOCK_SIZE+1] for i in ix])
    return x, y

# ── Build model ──────────────────────────────────────────
model = ArabicLM()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ── Training loop ────────────────────────────────────────
train_losses = []
val_losses   = []

for step in range(MAX_STEPS):
    model.train()
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ── Evaluate every EVAL_EVERY steps ──
    if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
        model.eval()
        with torch.no_grad():
            _, val_loss = model(*get_batch("val"))
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        print(f"Step {step:4d} | train loss: {loss.item():.4f} | val loss: {val_loss.item():.4f}")

# ── Save the trained model ───────────────────────────────
torch.save(model.state_dict(), "model/arabic_lm.pt")
print("Model saved to model/arabic_lm.pt")

# ── Save losses for plotting ─────────────────────────────
torch.save({"train": train_losses, "val": val_losses},
           "model/losses.pt")
print("Losses saved to model/losses.pt")