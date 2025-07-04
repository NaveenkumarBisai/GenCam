import streamlit as st
from PIL import Image
import torch
import itertools
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from modules.isp.isp_transformer import adjust_brightness, adjust_contrast, adjust_sharpness
from modules.eval.scorer import compute_bleu, compute_bertscore

# Load BLIP model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_model()

# Title
st.title("📸 GenCam: AI-Guided ISP Auto-Tuning")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Human reference caption input (optional)
    ref_caption = st.text_input("📝 Enter a human (reference) caption (optional):")

    # Define ISP grid
    brightness_vals = [0.8, 1.0, 1.2]
    contrast_vals = [0.9, 1.0, 1.1]
    sharpness_vals = [1.0, 1.5]
    combos = list(itertools.product(brightness_vals, contrast_vals, sharpness_vals))

    # Run auto-tuning
    st.info("🧪 Generating captions and scores...")
    results = []
    for b, c, s in combos:
        isp_image = adjust_brightness(image, b)
        isp_image = adjust_contrast(isp_image, c)
        isp_image = adjust_sharpness(isp_image, s)

        inputs = processor(isp_image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Score if reference is given
        bleu = compute_bleu(caption, ref_caption) if ref_caption else 0
        meteor = compute_bertscore(caption, ref_caption) if ref_caption else 0

        results.append({
            "brightness": b,
            "contrast": c,
            "sharpness": s,
            "caption": caption,
            "bleu": round(bleu, 4),
            "bert": round(meteor, 4),
            "image": isp_image
        })

    # Rank by BLEU+METEOR
    best = sorted(results, key=lambda x: (x["bleu"] + x["bert"]), reverse=True)[0]

    # Display best result
    st.subheader("✅ Best Tuned Output")
    st.image(best["image"], caption="Best Tuned Image", use_column_width=True)
    st.markdown(f"**📷 Caption:** _{best['caption']}_")
    if ref_caption:
        st.markdown(f"- **BLEU:** {best['bleu']}, **BERT:** {best['bert']}")
    st.markdown(f"- **ISP Settings →** Brightness: `{best['brightness']}`, Contrast: `{best['contrast']}`, Sharpness: `{best['sharpness']}`")

    # Expand to view all combinations
    with st.expander("🔍 View All ISP Variants and Captions"):
        for res in results:
            st.image(res["image"], caption=f"Caption: {res['caption']}\nISP: B={res['brightness']}, C={res['contrast']}, S={res['sharpness']}", use_column_width=True)
