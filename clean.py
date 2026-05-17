import re

def clean_arabic(text):
    # Remove URLs  (http://... or www....)
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove diacritics — the tiny marks above/below Arabic letters
    # Example: كَتَبَ  becomes  كتب
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', text)

    # Normalize: all alef variants → bare alef
    # أ إ آ  all become  ا
    text = re.sub(r'[أإآٱ]', 'ا', text)

    # Keep only Arabic characters and spaces (remove emoji, English, etc.)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- Read the raw file ---
with open("data/arabic_raw.txt", encoding="utf-8") as f:
    raw_lines = f.readlines()

print(f"Raw lines: {len(raw_lines)}")

# --- Clean each line ---
clean_lines = []
for line in raw_lines:
    cleaned = clean_arabic(line)
    if len(cleaned) > 20:   # skip very short lines
        clean_lines.append(cleaned)

print(f"Clean lines: {len(clean_lines)}")

# --- Show before / after for 3 examples ---
print("\nBefore / After examples:")
for i in range(3):
    print(f"  Before: {raw_lines[i].strip()[:80]}")
    print(f"  After:  {clean_arabic(raw_lines[i])[:80]}")
    print()

# --- Save ---
with open("data/arabic_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(clean_lines))

print("Saved to data/arabic_clean.txt")