from datasets import load_dataset

print("Downloading Arabic Wikipedia data...")

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.ar",        # Arabic Wikipedia snapshot
    split="train",
    streaming=True
)

texts = []
for i, sample in enumerate(dataset):
    texts.append(sample["text"])
    if i % 5000 == 0:
        print(f"  Collected {i} texts so far...")
    if i >= 50_000:
        break

print(f"Done! Collected {len(texts)} texts.")
print("\nFirst example:")
print(texts[0][:300])

# Save to disk
with open("data/arabic_raw.txt", "w", encoding="utf-8") as f:
    for line in texts:
        f.write(line.strip() + "\n")

print("Saved to data/arabic_raw.txt")