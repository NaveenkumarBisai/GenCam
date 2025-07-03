import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import itertools
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from modules.isp.isp_transformer import adjust_brightness, adjust_contrast, adjust_sharpness
from modules.eval.scorer import compute_bleu, compute_bertscore

# === Config ===
SAMPLE_DIR = "data/coco/val2017"
ANNOT_FILE = "data/coco/annotations/captions_val2017.json"
OUTPUT_CSV = "results/captions/auto_tuned_batch_results.csv"
NUM_IMAGES = 5  # You can increase this later

# === ISP Parameter Grid ===
brightness_vals = [0.8, 1.0, 1.2]
contrast_vals   = [0.9, 1.0, 1.1]
sharpness_vals  = [1.0, 1.5]
combo_grid = list(itertools.product(brightness_vals, contrast_vals, sharpness_vals))

# === Load BLIP Model ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# === Load COCO Captions ===
with open(ANNOT_FILE) as f:
    coco = json.load(f)

filename_to_caption = {img["file_name"]: [] for img in coco["images"]}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    fname = next((img["file_name"] for img in coco["images"] if img["id"] == img_id), None)
    if fname in filename_to_caption:
        filename_to_caption[fname].append(ann["caption"])

# === Select Sample Images ===
image_files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".jpg")][:NUM_IMAGES]

all_results = []

# === Main Loop ===
for image_name in tqdm(image_files, desc="Tuning Images"):
    image_path = os.path.join(SAMPLE_DIR, image_name)
    image = Image.open(image_path).convert("RGB")

    human_captions = filename_to_caption.get(image_name, [])
    if not human_captions:
        continue
    human_caption = human_captions[0]  # Take first as ground truth

    best_bleu = -1
    best_result = {}

    for b, c, s in combo_grid:
        # Apply ISP
        mod_img = adjust_brightness(image, b)
        mod_img = adjust_contrast(mod_img, c)
        mod_img = adjust_sharpness(mod_img, s)

        # Generate Caption
        inputs = processor(mod_img, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Score
        bleu = compute_bleu(caption, human_caption)
        bert = compute_bertscore(caption, human_caption)

        if bleu > best_bleu:
            best_bleu = bleu
            best_result = {
                "image": image_name,
                "brightness": b,
                "contrast": c,
                "sharpness": s,
                "caption": caption,
                "human_caption": human_caption,
                "bleu": round(bleu, 4),
                "bert": round(bert, 4)
            }

    all_results.append(best_result)

# Save Results
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Batch tuning complete. Results saved to: {OUTPUT_CSV}")
