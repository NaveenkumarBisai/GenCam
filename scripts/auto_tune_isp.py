import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import itertools
import json
import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from modules.eval.scorer import compute_bleu, compute_bertscore
from modules.isp.isp_transformer import adjust_brightness, adjust_contrast, adjust_sharpness

# Model setup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Image and COCO captions
image_name = "000000573626.jpg"
image_path = f"data/coco/val2017/{image_name}"
annotations_path = "data/coco/annotations/captions_val2017.json"

with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

# Get ground-truth human caption
img_id = next((img["id"] for img in coco_data["images"] if img["file_name"] == image_name), None)
human_captions = [ann["caption"] for ann in coco_data["annotations"] if ann["image_id"] == img_id]
human_caption = human_captions[0] if human_captions else ""

# ISP parameter grid
brightness_values = [0.8, 1.0, 1.2]
contrast_values   = [0.9, 1.0, 1.1]
sharpness_values  = [1.0, 1.5]

combos = list(itertools.product(brightness_values, contrast_values, sharpness_values))
image = Image.open(image_path).convert("RGB")

results = []

for b, c, s in combos:
    # Apply ISP
    transformed = adjust_brightness(image, b)
    transformed = adjust_contrast(transformed, c)
    transformed = adjust_sharpness(transformed, s)

    # Caption with BLIP
    inputs = processor(transformed, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Score
    bleu = compute_bleu(caption, human_caption)
    bert = compute_bertscore(caption, human_caption)

    results.append({
        "brightness": b,
        "contrast": c,
        "sharpness": s,
        "caption": caption,
        "bleu": round(bleu, 4),
        "bert": round(bert, 4)
    })

# Save best config
df = pd.DataFrame(results)
best_row = df.sort_values(by=["bleu", "bert"], ascending=False).iloc[0]
df.to_csv("results/captions/auto_tuned_isp_scores.csv", index=False)

print("\n🎯 Best ISP Parameters:")
print(best_row)
