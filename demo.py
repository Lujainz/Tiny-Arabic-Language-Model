import torch
import gradio as gr
import sentencepiece as spm
import matplotlib
matplotlib.use("Agg")   # no GUI window — Gradio handles display
import matplotlib.pyplot as plt
import seaborn as sns
import re, io
from PIL import Image
from model import ArabicLM, BLOCK_SIZE

# ── Load model + tokenizer once at startup ────────────────
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/arabic.model")

model = ArabicLM()
model.load_state_dict(torch.load("model/arabic_lm.pt",
                     map_location="cpu"))
model.eval()
print("Model ready.")

def clean(text):
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'[^؀-ۿs]', ' ', text)
    return re.sub(r's+', ' ', text).strip()

# ── Core function: generate + attention map ───────────────
def generate_and_visualize(prompt, max_tokens, temperature, rep_penalty):
    if not prompt.strip():
        return "اكتب نصاً عربياً", None   # "write Arabic text"

    ids  = sp.encode(clean(prompt))
    idx  = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(int(max_tokens)):
            crop   = idx[:, -BLOCK_SIZE:]
            logits, _ = model(crop)
            logits = logits[:, -1, :] / temperature
            for tid in set(idx[0].tolist()[-20:]):
                logits[0, tid] /= rep_penalty
            probs  = torch.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1)
            idx    = torch.cat([idx, nxt], dim=1)

    # decode output
    output = sp.decode(idx[0].tolist())

    # get attention weights
    final_ids = idx[:, -BLOCK_SIZE:]
    with torch.no_grad():
        model(final_ids)

    weights = model.all_attn_weights
    tokens  = sp.encode(
                sp.decode(final_ids[0].tolist()),
                out_type=str)[:24]

    # draw heatmaps
    n_layers = len(weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
    fig.patch.set_facecolor('#0f0f0f')

    for i, w in enumerate(weights):
        avg = w[0].mean(0).cpu().numpy()[:24, :24]
        ax  = axes[i]
        ax.set_facecolor('#1a1a1a')
        sns.heatmap(avg, ax=ax, cmap="YlOrRd",
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cbar=i==n_layers-1,
                    vmin=0, vmax=avg.max())
        ax.set_title(f"Layer {i+1}", color="white", fontsize=10)
        ax.tick_params(colors='white', labelsize=6)

    plt.tight_layout()

    # convert to PIL image for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor='#0f0f0f')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)

    return output, img

# ── Build the Gradio UI ───────────────────────────────────
with gr.Blocks(title="Arabic LLM Demo") as app:
    gr.Markdown("## نموذج اللغة العربية الصغيرTiny Arabic Language Model — built from scratch")

    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(
                label="Arabic prompt (النص العربي)",
                placeholder="السلام علي...",
                rtl=True
            )
            with gr.Row():
                max_tok = gr.Slider(10, 80, value=40,
                                    step=5, label="Tokens to generate")
                temp    = gr.Slider(0.5, 1.5, value=0.9,
                                    step=0.1, label="Temperature")
                rep_pen = gr.Slider(1.0, 3.0, value=1.8,
                                    step=0.1, label="Repetition penalty")
            btn = gr.Button("Generate ✨", variant="primary")

        with gr.Column():
            output_box = gr.Textbox(label="Completion", rtl=True)
            attn_img   = gr.Image(label="Attention maps")

    btn.click(
        fn=generate_and_visualize,
        inputs=[prompt_box, max_tok, temp, rep_pen],
        outputs=[output_box, attn_img]
    )

    gr.Examples(
        examples=[
            ["السلام علي"],
            ["في يوم من الأيام"],
            ["المملكة العربية السعودية"],
            ["كان يا ما كان"],
        ],
        inputs=prompt_box
    )

app.launch()