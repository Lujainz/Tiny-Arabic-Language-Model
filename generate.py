import torch
import re
import sentencepiece as spm
from model import ArabicLM, BLOCK_SIZE

# ── Load tokenizer (from Phase 1) ────────────────────────
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/arabic.model")

# ── Load trained model ───────────────────────────────────
model = ArabicLM()
model.load_state_dict(torch.load("model/arabic_lm.pt"))
model.eval()
print("Model loaded.")


def clean_prompt(text):
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Generate function ────────────────────────────────────
def complete(prompt, max_new_tokens=60, temperature=0.9):
    ids = sp.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -BLOCK_SIZE:]
            logits, _ = model(idx_crop)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty — make recently used tokens less likely
            generated = idx[0].tolist()
            for token_id in set(generated[-20:]):  # last 20 tokens
                logits[0, token_id] /= 1.8          # reduce their probability

            probs   = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx     = torch.cat([idx, next_id], dim=1)

    return sp.decode(idx[0].tolist())

# ── Try it ───────────────────────────────────────────────
prompts = [
    clean_prompt("السلام علي"),
    clean_prompt("في يوم من الأيام"), 
    clean_prompt("المملكة العربية السعودية"),
]

for p in prompts:
    print(f"Input:  {p}")
    print(f"Output: {complete(p)}")
    print("-" * 50)
