import sentencepiece as spm
import torch
from tqdm import tqdm

# ── Part 1: Train the tokenizer ──────────────────────────
print("Training tokenizer...")

spm.SentencePieceTrainer.train(
    input="data/arabic_clean.txt",
    model_prefix="tokenizer/arabic",   # saves arabic.model + arabic.vocab
    vocab_size=4000,       # 4000 unique tokens
    character_coverage=0.9999,
    model_type="bpe",
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    pad_piece="<PAD>", unk_piece="<UNK>",
    bos_piece="<BOS>", eos_piece="<EOS>",
)
print("Tokenizer saved to tokenizer/arabic.model")

# ── Part 2: Test the tokenizer ───────────────────────────
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/arabic.model")

test = "مرحبا بالعالم"
tokens = sp.encode(test, out_type=str)
ids    = sp.encode(test)
back   = sp.decode(ids)

print(f"\nTest sentence:  {test}")
print(f"Tokens:         {tokens}")
print(f"IDs (numbers):  {ids}")
print(f"Decoded back:   {back}")

# ── Part 3: Encode the full dataset ──────────────────────
print("\nEncoding full dataset...")

all_ids = []
with open("data/arabic_clean.txt", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines):
    ids = sp.encode(line.strip())
    if len(ids) > 5:
        all_ids.extend(ids)

print(f"Total tokens: {len(all_ids):,}")

# ── Part 4: Split and save as PyTorch tensors ─────────────
data = torch.tensor(all_ids, dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

torch.save(train_data, "data/train.pt")
torch.save(val_data,   "data/val.pt")

print(f"Train tokens: {len(train_data):,}  → saved to data/train.pt")
print(f"Val tokens:   {len(val_data):,}  → saved to data/val.pt")
print("\nPhase 1 complete!")