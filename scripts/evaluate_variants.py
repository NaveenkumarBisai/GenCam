# scripts/evaluate_variants.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from modules.eval.scorer import compute_bleu, compute_bertscore

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Load annotations
with open("data/coco/annotations/captions_val2017.json") as f:
    coco_data = json.load(f)

filename_to_caption = {img["file_name"]: [] for img in coco_data["images"]}
for ann in coco_data["annotations"]:
    img_id = ann["image_id"]
    fname = next((img["file_name"] for img in coco_data["images"] if img["id"] == img_id), None)
    if fname in filename_to_caption:
        filename_to_caption[fname].append(ann["caption"])

# Evaluate all variants of a sample image
img_base = "000000573626.jpg"  # choose your test image
variants = ["original.jpg", "dark.jpg", "bright.jpg", "low_contrast.jpg", "sharper.jpg"]
variants_dir = "results/"
human_caption = filename_to_caption.get(img_base, [""])[0]  # use first caption as ground truth
print(f"Captions for {img_base}: {filename_to_caption.get(img_base, [])}")
results = []
for variant in variants:
    path = os.path.join(variants_dir, variant)
    image = Image.open(path).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    blip_caption = processor.decode(out[0], skip_special_tokens=True)

    bleu_score = compute_bleu(blip_caption, human_caption)
    bert_score = compute_bertscore(blip_caption, human_caption)

    results.append({
        "variant": variant,
        "blip_caption": blip_caption,
        "human_caption": human_caption,
        "bleu": round(bleu_score, 4),
        "bertscore": round(bert_score, 4)
    })

df = pd.DataFrame(results)
df.to_csv("results/captions/isp_variant_scores.csv", index=False)
print("✅ Scores saved to: results/captions/isp_variant_scores.csv")
