import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

# Load saved losses
losses = torch.load("model/losses.pt")
train_l = losses["train"]
val_l   = losses["val"]

steps = [i * 300 for i in range(len(train_l))]

plt.figure(figsize=(9, 5))
plt.plot(steps, train_l, label="Train loss",
         color="#534AB7", linewidth=2)
plt.plot(steps, val_l,   label="Val loss",
         color="#1D9E75", linewidth=2, linestyle="--")

plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Arabic LLM — training progress")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("model/loss_curve.png", dpi=150)
plt.show()
print("Saved to model/loss_curve.png")